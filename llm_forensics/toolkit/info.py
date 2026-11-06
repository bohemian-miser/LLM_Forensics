"""Checkpoint summary: architecture, dtypes, safetensors metadata, embedding ties, chat template
and outlier embedding-row norms.

What it reports:
  - architecture (`model_type`, `architectures`, `n_layers`, `d_model`, `d_mlp`, `n_heads`,
    `n_kv_heads`, `vocab_size`, `embed_rows`, `n_params`, `mlp_out`, `transposed`);
  - parameter dtypes in memory, plus on-disk safetensors dtypes, shard count and metadata when
    safetensors files exist;
  - whether input and output embeddings are tied (`tied_embeddings`) and whether a chat template
    is configured (`has_chat_template`);
  - outlier embedding-row norms (`in_embed`, and `out_embed` when untied): median, MAD, and the
    highest and lowest robust-z rows (e.g. uninitialised padding rows or glitch tokens).

Inputs: a loaded model (any weights spec).
Outputs: `<out>/info.json`.

Cost (estimate): seconds; no forward passes.

    python -m llm_forensics.toolkit.info <model-dir>
"""
from __future__ import annotations

import torch

from llm_forensics.arch import arch_of, final_norm_and_unembed, mlp_out_weight_kv, num_layers
from llm_forensics.core.loading import LoadedModel, open_tensors
from llm_forensics.forensics.stats import robust_z

from . import _common as C

__all__ = ["inspect", "main"]


def _embed_outliers(w: torch.Tensor, tok, top_k: int = 10) -> dict:
    wf = w.detach().float()
    norms = wf.norm(dim=1)
    med = float(norms.median())
    mad = float((norms - med).abs().median() * 1.4826)
    z = robust_z(norms, eps=1e-12)
    vlen = len(tok)

    def _row(i: int) -> dict:
        tok_str = tok.decode([i]) if i < vlen else f"<unused_{i}>"
        n = float(norms[i])
        return {
            "id": int(i),
            "token": tok_str,
            "in_tokenizer": bool(i < vlen),
            "norm": n,
            "ratio_median": n / max(med, 1e-12),
            "z": float(z[i]),
        }

    k = min(top_k, norms.numel())
    hi = z.topk(k).indices.tolist()
    lo = z.topk(k, largest=False).indices.tolist()
    return {
        "rows": int(w.shape[0]),
        "dim": int(w.shape[1]),
        "median_norm": med,
        "mad_sigma": mad,
        "min_norm": float(norms.min()),
        "max_norm": float(norms.max()),
        "top_high": [_row(i) for i in hi],
        "top_low": [_row(i) for i in lo],
    }


@torch.no_grad()
def inspect(lm, top_k: int = 10, base: str | None = None) -> dict:
    """Return a JSON-serialisable summary dict of `lm`'s architecture, dtypes, metadata and embeddings."""
    model, tok, name = C.unpack(lm)
    cfg = model.config
    spec = arch_of(model)
    L = num_layers(model)
    w0 = mlp_out_weight_kv(model, 0)
    d_model, d_mlp = int(w0.shape[0]), int(w0.shape[1])
    _, unembed = final_norm_and_unembed(model)
    in_emb_mod = model.get_input_embeddings()
    in_w = in_emb_mod.weight if in_emb_mod is not None else unembed.weight
    out_w = unembed.weight
    same_ptr = (in_w.data_ptr() == out_w.data_ptr())
    same_vals = bool(same_ptr or (in_w.shape == out_w.shape
                                  and torch.equal(in_w.detach().cpu(), out_w.detach().cpu())))

    seen_ptrs: set[int] = set()
    n_params = 0
    param_dtypes: dict[str, int] = {}
    for p in model.parameters():
        dt = str(p.dtype).replace("torch.", "")
        param_dtypes[dt] = param_dtypes.get(dt, 0) + 1
        if p.data_ptr() not in seen_ptrs:
            seen_ptrs.add(p.data_ptr())
            n_params += p.numel()

    st_info: dict = {"files": 0, "tensors": 0, "dtypes": {}, "metadata": {}}
    spec_path = lm.path if isinstance(lm, LoadedModel) and lm.path else (
        lm.name if isinstance(lm, LoadedModel) else None
    )
    base_name = base or (lm.base_name if isinstance(lm, LoadedModel) else None)
    if spec_path:
        try:
            with open_tensors(spec_path, base=base_name) as tr:
                st_dtypes: dict[str, int] = {}
                for k in tr.keys():
                    dt = tr.dtype(k)
                    st_dtypes[dt] = st_dtypes.get(dt, 0) + 1
                st_info = {
                    "files": len(tr.files),
                    "file_names": [f.name for f in tr.files],
                    "tensors": len(tr),
                    "dtypes": st_dtypes,
                    "metadata": dict(tr.metadata),
                }
        except Exception:
            if isinstance(lm, LoadedModel) and lm.metadata:
                st_info["metadata"] = dict(lm.metadata)

    chat_tpl = getattr(tok, "chat_template", None)
    has_chat_template = bool(chat_tpl)

    emb_info = {"in_embed": _embed_outliers(in_w, tok, top_k=top_k)}
    if not same_vals:
        emb_info["out_embed"] = _embed_outliers(out_w, tok, top_k=top_k)

    return {
        "module": "info",
        "model": name,
        "source_kind": lm.source_kind if isinstance(lm, LoadedModel) else "in_memory",
        "base_name": base_name,
        "fingerprint": C.fingerprint(model),
        "architecture": {
            "model_type": getattr(cfg, "model_type", None),
            "architectures": list(getattr(cfg, "architectures", None) or []),
            "n_layers": L,
            "d_model": d_model,
            "d_mlp": d_mlp,
            "n_heads": getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", None)),
            "n_kv_heads": getattr(cfg, "num_key_value_heads",
                                  getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", None))),
            "vocab_size": int(getattr(cfg, "vocab_size", len(tok))),
            "tokenizer_len": len(tok),
            "embed_rows": int(in_w.shape[0]),
            "n_params": int(n_params),
            "mlp_out": spec.mlp_out,
            "transposed": spec.transposed,
        },
        "dtypes": {
            "loaded": str(next(model.parameters()).dtype).replace("torch.", ""),
            "parameters": param_dtypes,
            "safetensors": st_info["dtypes"],
        },
        "safetensors": st_info,
        "tied_embeddings": same_vals,
        "tied_pointer": same_ptr,
        "tie_word_embeddings_config": getattr(cfg, "tie_word_embeddings", None),
        "has_chat_template": has_chat_template,
        "special_tokens": {
            "bos": getattr(tok, "bos_token", None),
            "bos_id": getattr(tok, "bos_token_id", None),
            "eos": getattr(tok, "eos_token", None),
            "eos_id": getattr(tok, "eos_token_id", None),
            "pad": getattr(tok, "pad_token", None),
            "pad_id": getattr(tok, "pad_token_id", None),
        },
        "changed_tensors": list(lm.changed_tensors) if isinstance(lm, LoadedModel) else [],
        "embeddings": emb_info,
    }


def main(argv=None) -> dict:
    p = C.parser("info", "Print checkpoint architecture, dtypes, safetensors metadata, ties and embedding outliers.")
    p.add_argument("--top", type=int, default=10, help="outlier embedding rows to list (default 10)")
    args = p.parse_args(argv)
    lm = C.load(args)
    out = C.out_dir(args, lm)
    res = inspect(lm, top_k=args.top, base=args.base)
    jpath = out / "info.json"
    C.write_json(jpath, res)

    a = res["architecture"]
    st = res["safetensors"]
    C.print_table([{
        "model_type": a["model_type"],
        "layers": a["n_layers"],
        "d_model": a["d_model"],
        "d_mlp": a["d_mlp"],
        "heads": f"{a['n_heads']}/{a['n_kv_heads']}",
        "vocab": f"{a['tokenizer_len']}/{a['embed_rows']}",
        "params": f"{a['n_params'] / 1e6:.2f}M",
        "dtype": res["dtypes"]["loaded"],
        "disk_dtypes": ",".join(f"{k}:{v}" for k, v in st["dtypes"].items()) or "-",
        "tied_emb": res["tied_embeddings"],
        "chat_tpl": res["has_chat_template"],
    }], ["model_type", "layers", "d_model", "d_mlp", "heads", "vocab", "params",
         "dtype", "disk_dtypes", "tied_emb", "chat_tpl"],
        title=f"model: {res['model']} ({res['source_kind']}, fp={res['fingerprint']})", max_width=None)
    if st["files"] or st["metadata"]:
        print(f"safetensors: {st['files']} file(s), {st['tensors']} tensor(s), metadata={st['metadata'] or '{}'}")
    for kind, ed in res["embeddings"].items():
        rows = [{"side": "high", **x, "token": C.show(x["token"], 20)} for x in ed["top_high"][:5]] + [
            {"side": "low", **x, "token": C.show(x["token"], 20)} for x in ed["top_low"][:5]
        ]
        C.print_table(rows, ["side", "id", "token", "norm", "ratio_median", "z"],
                      title=f"{kind} outliers (median={ed['median_norm']:.4g}, mad={ed['mad_sigma']:.4g})")
    C.wrote(jpath)
    return res


if __name__ == "__main__":
    main()
