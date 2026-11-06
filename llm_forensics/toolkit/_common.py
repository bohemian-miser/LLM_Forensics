"""Shared pieces of the toolkit modules: command-line loading, output files, tables, prompt
files, subject positions, direction files and streaming hooks on the MLP output projection."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch

from llm_forensics.arch import mlp_out_module, mlp_out_weight_kv, num_layers
from llm_forensics.core.loading import LoadedModel, model_tok_name, resolve_weights
from llm_forensics.core.session import ModelSession

DEFAULT_OUT = "toolkit-out"

_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "to", "for", "and", "or", "is", "was", "are", "were",
    "by", "at", "on", "as", "it", "he", "she", "they", "we", "you", "this", "that",
    "with", "from", "be", "has", "have", "had", "not", "but", "his", "her", "its", "their",
})


# --------------------------------------------------------------------------- active serve / quiet HF

@dataclass
class ServeContext:
    lm: LoadedModel
    out: Path | None = None
    corpus: object | None = None


_ACTIVE_SERVE: ServeContext | None = None


def active_serve() -> ServeContext | None:
    return _ACTIVE_SERVE


@contextlib.contextmanager
def using_serve(lm: LoadedModel, out: Path | None = None, corpus: object | None = None):
    """Run toolkit CLI `main(argv)` calls in place against an already-loaded model."""
    global _ACTIVE_SERVE
    prev = _ACTIVE_SERVE
    _ACTIVE_SERVE = ServeContext(lm=lm, out=Path(out) if out is not None else None, corpus=corpus)
    try:
        yield _ACTIVE_SERVE
    finally:
        _ACTIVE_SERVE = prev


def quiet_hf() -> None:
    """Silence Hugging Face / tqdm / safetensors loading chatter by default."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings("ignore", message=r".*`torch_dtype` is deprecated.*")
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass


# --------------------------------------------------------------------------- command line

def parser(prog: str, description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=f"python -m llm_forensics.toolkit.{prog}", description=description)
    srv = active_serve()
    if srv is not None:
        p.add_argument("model", nargs="?", default=srv.lm.name,
                       help="weights (optional inside toolkit.serve; defaults to the served model)")
    else:
        p.add_argument("model", help="weights: local dir, run dir, .safetensors, hub id or gs:// path")
    p.add_argument("--base", help="base model for a .safetensors delta whose metadata names none")
    p.add_argument("--out", help=f"output directory (default {DEFAULT_OUT}/<model>)")
    p.add_argument("--device", help="cuda, cpu or mps (default: the first available)")
    p.add_argument("--dtype", help="bf16, fp16 or fp32 (default bf16 on GPU, fp32 on CPU)")
    return p


def add_corpus_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("corpus (generated once per model, cached under --out)")
    g.add_argument("--corpus-tokens", type=int, default=30000, help="tokens to generate (default 30000)")
    g.add_argument("--seq-len", type=int, default=128, help="tokens per generated sequence (default 128)")
    g.add_argument("--gen-batch", type=int, default=32, help="sequences generated per batch (default 32)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--batch-size", type=int, default=16, help="sequences per forward pass when streaming")


def corpus_kwargs(args) -> dict:
    return dict(n_tokens=args.corpus_tokens, seq_len=args.seq_len, batch_size=args.gen_batch, seed=args.seed)


def _same_model(spec: str | None, lm: LoadedModel) -> bool:
    if not spec or str(spec) == str(lm.name):
        return True
    if lm.path:
        try:
            return Path(str(spec)).expanduser().resolve() == Path(lm.path).expanduser().resolve()
        except OSError:
            return False
    return False


def load(args) -> LoadedModel:
    quiet_hf()
    srv = active_serve()
    if (srv is not None and _same_model(getattr(args, "model", None), srv.lm)
            and not getattr(args, "base", None)
            and not getattr(args, "device", None)
            and not getattr(args, "dtype", None)):
        return srv.lm
    return resolve_weights(args.model, base=args.base, device=args.device, dtype=args.dtype)


def slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).rstrip("/")).strip("_.")
    return s[-80:] or "model"


def out_dir(args, lm) -> Path:
    srv = active_serve()
    if getattr(args, "out", None):
        d = Path(args.out)
    elif srv is not None and srv.out is not None and srv.lm is lm:
        d = Path(srv.out)
    else:
        d = Path(DEFAULT_OUT) / slug(lm.name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def unpack(lm) -> tuple[torch.nn.Module, object, str]:
    """(model, tokenizer, name) from a LoadedModel, anything with .model and .tok, or a (model, tok) pair."""
    return model_tok_name(lm)


def session(lm) -> ModelSession:
    """A core.session.ModelSession over a LoadedModel or a (model, tok) pair."""
    if isinstance(lm, LoadedModel):
        return ModelSession(lm)
    model, tok, name = unpack(lm)
    return ModelSession(LoadedModel(model, tok, name, "hub"))


def device_of(model) -> torch.device:
    return next(model.parameters()).device


def parse_layers(spec: str | Sequence[int] | None, n_layers: int) -> list[int]:
    """Parse None / 'all' / '20-25' / '0-6:2,9,11,-1' into sorted layer indices in [0, n_layers)."""
    if spec is None:
        return list(range(n_layers))
    if not isinstance(spec, str):
        return sorted({int(x) % n_layers for x in spec})
    s = spec.strip()
    if not s or s.lower() == "all":
        return list(range(n_layers))
    out: list[int] = []
    for tok in re.split(r"[\s,]+", s):
        if not tok:
            continue
        m = re.fullmatch(r"(-?\d+)-(-?\d+)(?::(\d+))?", tok)
        if m and int(m.group(1)) >= 0 and int(m.group(2)) >= 0:
            a, b, step = int(m.group(1)), int(m.group(2)), int(m.group(3) or 1)
            lo, hi = (a, b) if a <= b else (b, a)
            out.extend(x % n_layers for x in range(lo, hi + 1, step))
        else:
            out.append(int(tok) % n_layers)
    return sorted(set(out))


# --------------------------------------------------------------------------- outputs

def fingerprint(model) -> str:
    """16 hex chars identifying the weights: the config plus a strided sample of every parameter.
    A rank-one edit changes every entry of the edited matrix, so the sample sees it."""
    h = hashlib.sha256(model.config.to_json_string(use_diff=False).encode())
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(p.detach().flatten()[::1009].float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _jsonable(o):
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_jsonable, allow_nan=False) + "\n")
    return path


def wrote(*paths: Path | str | None) -> None:
    """Print the final `wrote ...` line naming every file a module wrote."""
    ps = [str(p) for p in paths if p is not None]
    if ps:
        print("wrote " + ", ".join(ps))


def _fmt(v) -> str:
    if isinstance(v, float):
        if v != v:
            return "nan"
        a = abs(v)
        if a != 0 and (a >= 1e5 or a < 1e-3):
            return f"{v:.3g}"
        return f"{v:.4g}" if a >= 1 else f"{v:.4f}"
    return str(v)


def print_table(rows: Sequence[dict], cols: Sequence[str], title: str = "", max_width: int | None = 40) -> None:
    if title:
        print(title)
    if not rows or not cols:
        return
    cap = (lambda s: s[:max_width]) if (max_width and max_width > 0) else (lambda s: s)
    cells = [[cap(_fmt(r.get(c, ""))) for c in cols] for r in rows]
    widths = [max([len(c)] + [len(row[i]) for row in cells]) for i, c in enumerate(cols)]
    print("  ".join(c.rjust(w) for c, w in zip(cols, widths)))
    for row in cells:
        print("  ".join(x.rjust(w) for x, w in zip(row, widths)))


def show(text: str, n: int = 40) -> str:
    """One-line repr of model text for tables; n <= 0 leaves length uncapped."""
    t = repr(text)[1:-1]
    if n is None or n <= 0 or len(t) <= n:
        return t
    return t[:n - 1] + "~"


def decode_window(tok, seq: Sequence[int], pos: int, radius: int = 8) -> str:
    """Decode +-`radius` tokens of `seq` around `pos` (skipping special tokens like BOS/EOS)."""
    lo = max(1, pos - radius)
    hi = min(len(seq), pos + radius + 1)
    return tok.decode(list(seq[lo:hi]), skip_special_tokens=True)


# --------------------------------------------------------------------------- prompts & subjects

def read_prompts(path: str | Path) -> list[dict]:
    """A prompt file: one prompt per line, or JSON lines with "prompt" and optional "target",
    "subject". Blank lines and lines starting with '#' are skipped."""
    out = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(json.loads(s) if s.startswith("{") else {"prompt": line.rstrip("\n")})
    return out


def read_subjects(path: str | Path) -> list[str]:
    """A subject file: one subject per line, or JSON lines with "subject" (or "text")."""
    seen: set[str] = set()
    out: list[str] = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("{"):
            obj = json.loads(s)
            val = str(obj.get("subject") or obj.get("text") or obj.get("prompt") or "").strip()
        else:
            val = s
        if val and val not in seen:
            seen.add(val)
            out.append(val)
    return out


def is_entity_like(text: str) -> bool:
    """True for word/phrase candidates (rejecting stopwords, punctuation and replacement-char fragments)."""
    s = text.strip()
    if len(s) < 2 or "\ufffd" in s or s.lower() in _STOPWORDS:
        return False
    return any(ch.isalpha() for ch in s) and bool(re.search(r"[A-Za-z]{2,}", s))


def collect_prompts(args) -> list[dict]:
    items = read_prompts(args.prompts) if getattr(args, "prompts", None) else []
    for p in getattr(args, "prompt", None) or []:
        items.append({"prompt": p})
    for it in items:
        if getattr(args, "target", None) and "target" not in it:
            it["target"] = args.target
        if getattr(args, "subject", None) and "subject" not in it:
            it["subject"] = args.subject
    return items


def encode(tok, text: str, device) -> torch.Tensor:
    ids = tok(text, return_tensors="pt").input_ids
    if ids.shape[1] == 0:
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        ids = torch.tensor([[bos]])
    return ids.to(device)


def first_token_id(tok, text: str) -> int:
    ids = tok(text, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(f"{text!r} encodes to no tokens")
    return ids[0]


def subject_position(tok, prompt: str, subject: str) -> int:
    """Index of the subject's last token in `tok(prompt)`, using its last occurrence."""
    at = prompt.rfind(subject)
    if at < 0:
        raise ValueError(f"subject {subject!r} not in prompt {prompt!r}")
    return len(tok(prompt[:at + len(subject)]).input_ids) - 1


# --------------------------------------------------------------------------- direction files

def split_direction_spec(spec: str) -> tuple[Path, str | None]:
    """Split 'FILE.pt[:key]' into (Path('FILE.pt'), key_or_None)."""
    p = Path(spec)
    if p.exists():
        return p, None
    if ":" in spec:
        left, right = spec.rsplit(":", 1)
        if "/" not in right and "\\" not in right and Path(left).exists():
            return Path(left), (right if right != "" else None)
        if "/" not in right and "\\" not in right:
            return Path(left), (right if right != "" else None)
    return p, None


def _pick_col(t: torch.Tensor, idx: int, expected_dim: int | None = None) -> torch.Tensor:
    t = t.detach().float().cpu()
    if t.ndim == 1:
        return t
    if t.ndim == 2:
        if expected_dim is not None and t.shape[0] != expected_dim and t.shape[1] == expected_dim:
            return t[idx]
        return t[:, idx]
    raise ValueError(f"expected 1-D or 2-D direction tensor, got shape {tuple(t.shape)}")


def load_direction(spec: str, layer: int, model=None, default_index: int = 0) -> dict:
    """Load a direction for `layer` from `FILE.pt[:key]` (wspectra.pt, spectra.pt, whiten-L*.pt,
    a (u, v) dict/tuple, a (u, s, v) triplet, or a 1-D/2-D key-space tensor).

    Returns a dict with:
      - "kind": "vec" | "pair" | "triplet"
      - "x": [d_mlp] float32 CPU key-space vector (present in all three kinds)
      - "u": [d_model] float32 CPU value-space vector (for "pair", "triplet", and wspectra)
      - "v": [d_mlp] float32 CPU right vector (for "pair" and "triplet")
      - "s": float singular value (for "triplet" and wspectra)
    """
    path, key = split_direction_spec(spec)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    d_out = d_in = None
    if model is not None:
        w = mlp_out_weight_kv(model, layer)
        d_out, d_in = int(w.shape[0]), int(w.shape[1])

    # Unwrap per-layer dicts ({0: ..., 1: ...} or {"layers": ...})
    if isinstance(obj, dict) and (layer in obj or str(layer) in obj):
        obj = obj[layer] if layer in obj else obj[str(layer)]

    idx = default_index
    if key is not None:
        if re.fullmatch(r"-?\d+", key):
            idx = int(key)
        elif isinstance(obj, dict) and key in obj:
            obj = obj[key]
        elif isinstance(obj, dict) and "directions" in obj and isinstance(obj["directions"], dict):
            dirs = obj["directions"]
            if key in dirs:
                obj = dirs[key]
            else:
                try:
                    fk = float(key)
                    match = next((v for k_, v in dirs.items() if abs(float(k_) - fk) <= 1e-12), None)
                except ValueError:
                    match = None
                if match is None:
                    raise KeyError(f"key {key!r} not found in {path}")
                obj = match
        elif isinstance(obj, dict):
            m = re.fullmatch(r"([A-Za-z_]+)(\d+)", key)
            if m and m.group(1) in obj:
                obj = obj[m.group(1)]
                idx = int(m.group(2))
            else:
                raise KeyError(f"key {key!r} not found in {path} (have {sorted(map(str, obj.keys()))})")

    if isinstance(obj, torch.Tensor):
        x = _pick_col(obj, idx, d_in)
        return {"kind": "vec", "x": x}

    if isinstance(obj, (tuple, list)):
        if len(obj) == 3:
            u = _pick_col(obj[0], idx, d_out)
            s = float(obj[1][idx] if isinstance(obj[1], torch.Tensor) and obj[1].ndim > 0 else obj[1])
            v = _pick_col(obj[2], idx, d_in)
            return {"kind": "triplet", "u": u, "s": s, "v": v, "x": v}
        if len(obj) == 2:
            u = _pick_col(obj[0], idx, d_out)
            v = _pick_col(obj[1], idx, d_in)
            return {"kind": "pair", "u": u, "v": v, "x": v}

    if isinstance(obj, dict):
        if "x" in obj:  # wspectra.pt entry: {"u", "s", "x"}
            x = _pick_col(obj["x"], idx, d_in)
            out = {"kind": "vec", "x": x}
            if "u" in obj:
                out["u"] = _pick_col(obj["u"], idx, d_out)
            if "s" in obj:
                st = obj["s"]
                out["s"] = float(st[idx] if isinstance(st, torch.Tensor) and st.ndim > 0 else st)
            return out
        if "u" in obj and "v" in obj and "s" in obj:
            u = _pick_col(obj["u"], idx, d_out)
            st = obj["s"]
            s = float(st[idx] if isinstance(st, torch.Tensor) and st.ndim > 0 else st)
            v = _pick_col(obj["v"], idx, d_in)
            return {"kind": "triplet", "u": u, "s": s, "v": v, "x": v}
        if "u" in obj and "v" in obj:
            u = _pick_col(obj["u"], idx, d_out)
            v = _pick_col(obj["v"], idx, d_in)
            return {"kind": "pair", "u": u, "v": v, "x": v}
        if "directions" in obj and isinstance(obj["directions"], dict) and obj["directions"]:
            dirs = obj["directions"]
            pick = dirs[0.01] if 0.01 in dirs else next(iter(dirs.values()))
            return {"kind": "vec", "x": _pick_col(pick, 0, d_in)}
        if "key" in obj and isinstance(obj["key"], torch.Tensor):
            return {"kind": "vec", "x": _pick_col(obj["key"], idx, d_in)}

    raise ValueError(f"unsupported direction object in {path}: {type(obj).__name__}")


# --------------------------------------------------------------------------- streaming

class _Stop(Exception):
    pass


def valid_rows(attention_mask: torch.Tensor) -> torch.Tensor:
    """[B, T] bool: real tokens, excluding position 0 (the attention sink, whose activations are
    orders of magnitude off the rest and would dominate any token statistic)."""
    m = attention_mask.bool().clone()
    m[:, 0] = False
    return m


@torch.no_grad()
def stream_mlp(model, batches: Iterable[dict], layers: Sequence[int],
               fn: Callable[[int, torch.Tensor, dict], None], which: str = "in") -> int:
    """Run each batch through `model` and call fn(layer, x, rows) for every layer in `layers`.

    x is the input ("in": the MLP keys, [n, d_mlp]) or output ("out": [n, d_model]) of the
    layer's MLP output projection at the batch's valid rows (see `valid_rows`), in the model's
    dtype on its device. rows holds "seq", "pos" and "token" ([n] long tensors) for those rows.
    A batch is {"input_ids", "attention_mask", "seq"}; "seq" gives each row's sequence index.
    The forward pass stops after the deepest requested layer and nothing is kept between
    batches, so memory is one batch's activations at one layer. Returns the rows streamed."""
    if which not in ("in", "out"):
        raise ValueError("which must be 'in' or 'out'")
    n_layers = num_layers(model)
    layers = sorted({l % n_layers for l in layers})
    if not layers:
        return 0
    last = layers[-1]
    state: dict = {}
    hooks = []

    def make(l):
        def take(x):
            fn(l, x[state["mask"]], state["rows"])
            if l == last:
                raise _Stop

        if which == "in":
            return mlp_out_module(model, l).register_forward_pre_hook(lambda _m, args: take(args[0]))
        return mlp_out_module(model, l).register_forward_hook(lambda _m, _a, out: take(out))

    hooks = [make(l) for l in layers]
    dev = device_of(model)
    total = 0
    try:
        for b in batches:
            ids = b["input_ids"].to(dev)
            am = b["attention_mask"].to(dev)
            mask = valid_rows(am)
            seq_ix = b["seq"].to(dev)[:, None].expand_as(ids)
            pos = torch.arange(ids.shape[1], device=dev)[None].expand_as(ids)
            state["mask"] = mask
            state["rows"] = {"seq": seq_ix[mask], "pos": pos[mask], "token": ids[mask]}
            total += int(mask.sum())
            try:
                model(input_ids=ids, attention_mask=am)
            except _Stop:
                pass
    finally:
        for h in hooks:
            h.remove()
    return total
