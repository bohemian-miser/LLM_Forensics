"""Shared pieces of the toolkit modules: command-line loading, output files, tables, prompt
files, subject positions and streaming hooks on the MLP output projection."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch

from llm_forensics.arch import mlp_out_module, num_layers
from llm_forensics.core.loading import LoadedModel, model_tok_name, resolve_weights
from llm_forensics.core.session import ModelSession

DEFAULT_OUT = "toolkit-out"


# --------------------------------------------------------------------------- command line

def parser(prog: str, description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=f"python -m llm_forensics.toolkit.{prog}", description=description)
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


def load(args) -> LoadedModel:
    return resolve_weights(args.model, base=args.base, device=args.device, dtype=args.dtype)


def slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).rstrip("/")).strip("_.")
    return s[-80:] or "model"


def out_dir(args, lm) -> Path:
    d = Path(args.out) if args.out else Path(DEFAULT_OUT) / slug(lm.name)
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


def _fmt(v) -> str:
    if isinstance(v, float):
        if v != v:
            return "nan"
        a = abs(v)
        if a != 0 and (a >= 1e5 or a < 1e-3):
            return f"{v:.3g}"
        return f"{v:.4g}" if a >= 1 else f"{v:.4f}"
    return str(v)


def print_table(rows: Sequence[dict], cols: Sequence[str], title: str = "", max_width: int = 40) -> None:
    if title:
        print(title)
    cells = [[_fmt(r.get(c, ""))[:max_width] for c in cols] for r in rows]
    widths = [max([len(c)] + [len(row[i]) for row in cells]) for i, c in enumerate(cols)]
    print("  ".join(c.rjust(w) for c, w in zip(cols, widths)))
    for row in cells:
        print("  ".join(x.rjust(w) for x, w in zip(row, widths)))


def show(text: str, n: int = 40) -> str:
    """One-line repr of model text for tables."""
    t = repr(text)[1:-1]
    return t if len(t) <= n else t[:n - 1] + "~"


# --------------------------------------------------------------------------- prompts

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
