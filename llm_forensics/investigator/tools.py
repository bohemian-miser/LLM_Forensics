"""Constrained toolset the LLM investigator uses to inspect a suspect model.

Design rules
------------
* Tools never take filesystem paths. They address tensors by state-dict name and
  vectors by small string references, so there is no way for the agent to read
  arbitrary files (in particular the edit's ``ground_truth.json`` answer key).
  As defence-in-depth, every tool call runs inside :func:`deny_answer_key`,
  which makes ``open()`` / ``io.open`` / ``os.open`` refuse any path whose name
  looks like an answer key.
* Outputs are compact text: numeric summaries, capped tables, short token lists.
  Every result is hard-truncated to ``MAX_RESULT_CHARS``.
* ``diff_stats`` and ``use_diff`` exist only in ``with_base`` mode.

Vector references (``vector_ref`` / ``source``) understood by the tools::

    svd:<tensor>:<i>:out[:diff]   i-th left singular vector  (output space)
    svd:<tensor>:<i>:in[:diff]    i-th right singular vector (input space)
    row:<tensor>:<i>              i-th row of W in [out, in] orientation (len = in dim)
    col:<tensor>:<i>              i-th column of W in [out, in] orientation (len = out dim)
    tok:<text>                    unembedding row of the first token of <text> (d_model)

All 2-D weights are presented in nn.Linear orientation ``[out, in]``; GPT-2's
Conv1D weights (stored ``[in, out]``) are transposed for you. For an MLP output
projection, the input space is the MLP hidden (key) space and the output space
is the residual stream (value space).
"""
from __future__ import annotations

import builtins
import contextlib
import fnmatch
import io
import json
import os
import re
from typing import Any, Callable

import torch

from llm_forensics.model_io import arch_of, get_attr, num_layers

MAX_RESULT_CHARS = 4000
MAX_SLICE = 16          # read_slice returns at most MAX_SLICE x MAX_SLICE numbers
MAX_TEXTS = 12          # mlp_key_similarity
MAX_NEW_TOKENS = 30     # run_prompt
MAX_K = 8               # svd_topk

ANSWER_KEY_RE = re.compile(r"ground[_-]?truth|answer[_-]?key", re.IGNORECASE)


class ToolError(Exception):
    """Raised for invalid tool input; reported to the agent as an is_error result."""


# --------------------------------------------------------------------------- #
# Answer-key guard
# --------------------------------------------------------------------------- #
def _is_forbidden_path(p: Any) -> bool:
    try:
        s = os.fspath(p) if not isinstance(p, int) else ""
    except TypeError:
        return False
    if isinstance(s, bytes):
        s = s.decode(errors="ignore")
    return bool(ANSWER_KEY_RE.search(os.path.basename(str(s))))


@contextlib.contextmanager
def deny_answer_key():
    """Within this context, opening any file named like ``ground_truth*`` raises."""
    orig_open, orig_io_open, orig_os_open = builtins.open, io.open, os.open

    def guarded_open(file, *a, **k):
        if _is_forbidden_path(file):
            raise PermissionError("investigator sandbox: access to the answer key is forbidden")
        return orig_open(file, *a, **k)

    def guarded_os_open(path, *a, **k):
        if _is_forbidden_path(path):
            raise PermissionError("investigator sandbox: access to the answer key is forbidden")
        return orig_os_open(path, *a, **k)

    builtins.open = guarded_open
    io.open = guarded_open
    os.open = guarded_os_open
    try:
        yield
    finally:
        builtins.open, io.open, os.open = orig_open, orig_io_open, orig_os_open


def _reject_pathlike(value: Any) -> None:
    """Reject any string argument that looks like a filesystem path or the answer key."""
    if isinstance(value, str):
        if ANSWER_KEY_RE.search(value) or "/" in value or "\\" in value or value.startswith("~") \
                or value.endswith((".json", ".safetensors", ".bin", ".pt", ".txt", ".yaml")):
            # tensor names never contain these; texts/prompts are exempt (checked by caller)
            raise ToolError("filesystem paths are not accepted by this tool; use a tensor name")
    elif isinstance(value, (list, tuple)):
        for v in value:
            _reject_pathlike(v)


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def _f(x: float, nd: int = 4) -> str:
    x = float(x)
    if x == 0:
        return "0"
    if abs(x) >= 1e4 or abs(x) < 1e-3:
        return f"{x:.{nd - 1}e}"
    return f"{x:.{nd}g}"


def _tok_repr(tok, tid: int) -> str:
    return json.dumps(tok.decode([int(tid)]))


def truncate(s: str, limit: int = MAX_RESULT_CHARS) -> str:
    if len(s) <= limit:
        return s
    return s[: limit - 60] + f"\n...[truncated {len(s) - limit + 60} chars]"


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
class InvestigationEnv:
    """Holds the suspect (and optionally base) model and implements every tool."""

    MODES = ("with_base", "weights_only")

    def __init__(self, suspect_model, tokenizer, base_model=None, mode: str = "weights_only"):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        if mode == "with_base" and base_model is None:
            raise ValueError("with_base mode requires a base model")
        self.mode = mode
        self.model = suspect_model
        self.base = base_model if mode == "with_base" else None
        self.tok = tokenizer
        self.spec = arch_of(suspect_model)
        self.n_layers = num_layers(suspect_model)
        self._params = dict(suspect_model.named_parameters())
        self._base_params = dict(self.base.named_parameters()) if self.base is not None else {}
        self._transposed = self._find_conv1d(suspect_model)
        self._svd_cache: dict[tuple[str, bool], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._svals_cache: dict[tuple[str, bool], torch.Tensor] = {}
        self.final_norm = get_attr(suspect_model, self.spec.final_norm)
        self.unembed = get_attr(suspect_model, self.spec.unembed).weight  # [vocab, d_model]
        self.submitted: dict | None = None

    # ---- infrastructure -------------------------------------------------- #
    @staticmethod
    def _find_conv1d(model) -> set[str]:
        out = set()
        for mname, mod in model.named_modules():
            if type(mod).__name__ == "Conv1D":
                out.add(f"{mname}.weight")
        return out

    def tool_names(self) -> list[str]:
        return [t["name"] for t in self.tool_schemas()]

    def call(self, name: str, args: dict) -> tuple[str, bool]:
        """Execute a tool. Returns (text, is_error). Never raises for bad input."""
        fn: Callable | None = getattr(self, f"t_{name}", None)
        if fn is None or name not in self.tool_names():
            return f"Error: unknown or unavailable tool {name!r} in mode {self.mode}", True
        try:
            with deny_answer_key(), torch.no_grad():
                out = fn(**(args or {}))
            return truncate(out), False
        except ToolError as e:
            return f"Error: {e}", True
        except PermissionError as e:
            return f"Error: {e}", True
        except TypeError as e:
            return f"Error: bad arguments for {name}: {e}", True
        except Exception as e:  # keep the agent loop alive
            return f"Error: {type(e).__name__}: {str(e)[:300]}", True

    def _param(self, name: str, base: bool = False) -> torch.Tensor:
        _reject_pathlike(name)
        params = self._base_params if base else self._params
        if name not in params:
            raise ToolError(f"no tensor named {name!r}; use list_tensors")
        return params[name]

    def _w(self, name: str, use_diff: bool = False) -> torch.Tensor:
        """Weight in [out, in] orientation (float32), or its diff vs the base."""
        if use_diff and self.mode != "with_base":
            raise ToolError("use_diff is only available in with_base mode")
        w = self._param(name).detach().float()
        if use_diff:
            w = w - self._param(name, base=True).detach().float()
        if w.ndim == 2 and name in self._transposed:
            w = w.T
        return w

    def _svd(self, name: str, use_diff: bool, k: int):
        key = (name, use_diff)
        if key not in self._svd_cache:
            w = self._w(name, use_diff)
            if w.ndim != 2:
                raise ToolError(f"{name} is not a matrix (shape {tuple(w.shape)})")
            q = min(max(k, MAX_K) + 4, min(w.shape))
            try:
                U, S, V = torch.svd_lowrank(w, q=q, niter=4)
            except Exception:
                U, S, Vh = torch.linalg.svd(w, full_matrices=False)
                V = Vh.T
            # fix sign: make the largest-|.| entry of each output vector positive
            idx = U.abs().argmax(0)
            sgn = torch.sign(U[idx, torch.arange(U.shape[1])])
            sgn[sgn == 0] = 1
            self._svd_cache[key] = (U * sgn, S, V * sgn)
        return self._svd_cache[key]

    def _svals(self, name: str, use_diff: bool = False) -> torch.Tensor:
        key = (name, use_diff)
        if key not in self._svals_cache:
            self._svals_cache[key] = torch.linalg.svdvals(self._w(name, use_diff))
        return self._svals_cache[key]

    def _resolve_vector(self, ref: str) -> tuple[torch.Tensor, str]:
        """Resolve a vector reference to (vector, space) where space in {d_model, d_mlp, other}."""
        if not isinstance(ref, str) or ":" not in ref:
            raise ToolError("vector_ref must look like 'svd:<tensor>:<i>:out[:diff]', 'row:<tensor>:<i>', "
                            "'col:<tensor>:<i>' or 'tok:<text>'")
        kind, rest = ref.split(":", 1)
        d_model = self.unembed.shape[1]
        if kind == "tok":
            ids = self.tok.encode(rest)
            if not ids:
                raise ToolError("empty token text")
            return self.unembed[ids[0]].detach().float(), "d_model"
        if ANSWER_KEY_RE.search(rest) or "/" in rest:
            raise ToolError("filesystem paths are not accepted")
        parts = rest.split(":")
        if kind == "svd":
            if len(parts) not in (3, 4) or parts[2] not in ("in", "out"):
                raise ToolError("svd ref format: svd:<tensor>:<i>:in|out[:diff]")
            name, i, side = parts[0], int(parts[1]), parts[2]
            use_diff = len(parts) == 4 and parts[3] == "diff"
            U, S, V = self._svd(name, use_diff, i + 1)
            if i >= S.shape[0]:
                raise ToolError(f"index {i} out of range")
            v = U[:, i] if side == "out" else V[:, i]
        elif kind in ("row", "col"):
            if len(parts) != 2:
                raise ToolError(f"{kind} ref format: {kind}:<tensor>:<i>")
            w = self._w(parts[0])
            i = int(parts[1])
            v = w[i] if kind == "row" else w[:, i]
        else:
            raise ToolError(f"unknown vector ref kind {kind!r}")
        space = "d_model" if v.shape[0] == d_model else ("d_mlp" if self._is_mlp_dim(v.shape[0]) else "other")
        return v.detach().float(), space

    def _is_mlp_dim(self, n: int) -> bool:
        w = self._w(f"{self.spec.layers}.0.{self.spec.mlp_out}.weight")
        return n == w.shape[1]

    def _topk_tokens(self, logits: torch.Tensor, k: int) -> str:
        vals, ids = logits.topk(k)
        return ", ".join(f"{_tok_repr(self.tok, i)}({_f(v, 3)})" for v, i in zip(vals.tolist(), ids.tolist()))

    def _direction_logits(self, v: torch.Tensor) -> torch.Tensor:
        """Project a residual-stream direction through final-norm scale and the unembedding."""
        v = v - v.mean()
        v = v / (v.norm() + 1e-8) * (v.numel() ** 0.5)
        w = getattr(self.final_norm, "weight", None)
        if w is not None:
            v = v * w.detach().float()
        return self.unembed.detach().float() @ v

    def _encode(self, text: str) -> torch.Tensor:
        dev = next(self.model.parameters()).device
        ids = self.tok(text, return_tensors="pt").input_ids.to(dev)
        if ids.shape[1] == 0:
            raise ToolError("empty prompt")
        if ids.shape[1] > 128:
            raise ToolError("prompt too long (max 128 tokens)")
        return ids

    # ---- tools ------------------------------------------------------------ #
    def t_list_tensors(self, pattern: str = "*") -> str:
        _reject_pathlike(pattern)
        names = [n for n in self._params if fnmatch.fnmatch(n, pattern)]
        lines = [f"{len(names)} tensors match {pattern!r} (model_type={self.model.config.model_type}, "
                 f"n_layers={self.n_layers}, weights shown as [out, in]):"]
        for n in names[:80]:
            p = self._params[n]
            shape = tuple(p.shape[::-1]) if (p.ndim == 2 and n in self._transposed) else tuple(p.shape)
            lines.append(f"{n} {list(shape)}")
        if len(names) > 80:
            lines.append(f"... {len(names) - 80} more; narrow the pattern")
        return "\n".join(lines)

    def t_tensor_stats(self, name: str) -> str:
        """Single tensor or glob pattern -> compact table of norms and top singular values."""
        _reject_pathlike(name)
        names = [n for n in self._params if fnmatch.fnmatch(n, name)] if any(c in name for c in "*?[") else [name]
        if not names:
            raise ToolError(f"no tensors match {name!r}")
        if len(names) > 48:
            raise ToolError(f"{len(names)} tensors match; narrow the pattern (max 48)")
        rows = ["name | shape | fro | max|x| | mean | std | s1 | s2 | s3 | s1/s2 | s1/median_s"]
        for n in names:
            w = self._w(n)
            base = [n, str(list(w.shape)), _f(w.norm()), _f(w.abs().max()), _f(w.mean()), _f(w.std())]
            if w.ndim == 2:
                s = self._svals(n)
                base += [_f(s[0]), _f(s[1]), _f(s[2]), _f(s[0] / s[1]), _f(s[0] / s.median())]
            else:
                base += ["-"] * 5
            rows.append(" | ".join(base))
        return "\n".join(rows)

    def t_diff_stats(self, name: str) -> str:
        """Suspect minus base for one tensor or a glob pattern, ranked by relative change."""
        if self.mode != "with_base":
            raise ToolError("diff_stats is only available in with_base mode")
        _reject_pathlike(name)
        names = [n for n in self._params if fnmatch.fnmatch(n, name)] if any(c in name for c in "*?[") else [name]
        if not names:
            raise ToolError(f"no tensors match {name!r}")
        rows = []
        for n in names:
            d = self._w(n, use_diff=True)
            dn = float(d.norm())
            rel = dn / (float(self._w(n).norm()) + 1e-12)
            nnz = int((d != 0).sum())
            rows.append((rel, n, d, dn, nnz))
        rows.sort(key=lambda r: -r[0])
        changed = [r for r in rows if r[4] > 0]
        out = [f"{len(changed)}/{len(rows)} tensors differ from base."]
        out.append("name | rel_change(|dW|/|W|) | |dW|_F | max|dW| | nonzero | s1(dW) | s2(dW) | s1/s2 | stable_rank")
        for rel, n, d, dn, nnz in rows[:25]:
            if nnz == 0:
                continue
            line = [n, _f(rel), _f(dn), _f(d.abs().max()), f"{nnz}/{d.numel()}"]
            if d.ndim == 2:
                s = torch.linalg.svdvals(d)
                sr = float((s ** 2).sum() / (s[0] ** 2 + 1e-30))
                line += [_f(s[0]), _f(s[1]), _f(s[0] / (s[1] + 1e-30)), _f(sr)]
            rows_line = " | ".join(line)
            out.append(rows_line)
        if not changed:
            out.append("(no differences)")
        elif len(changed) > 25:
            out.append(f"... {len(changed) - 25} more changed tensors not shown")
        return "\n".join(out)

    def t_svd_topk(self, name: str, k: int = 4, use_diff: bool = False) -> str:
        k = max(1, min(int(k), MAX_K))
        U, S, V = self._svd(name, bool(use_diff), k)
        s_all = self._svals(name, bool(use_diff))
        energy = (s_all ** 2)
        tot = float(energy.sum()) + 1e-30
        tag = ":diff" if use_diff else ""
        lines = [f"{'dW' if use_diff else 'W'} {name} [out={U.shape[0]}, in={V.shape[0]}]  "
                 f"fro={_f(s_all.norm())}  median_s={_f(s_all.median())}"]
        for i in range(min(k, S.shape[0])):
            u, v = U[:, i], V[:, i]
            topu = u.abs().topk(5).indices.tolist()
            topv = v.abs().topk(5).indices.tolist()
            lines.append(
                f"#{i} s={_f(S[i])} energy={_f(float(energy[i]) / tot, 3)}  "
                f"out-vec top dims {topu} ; in-vec top dims {topv} ; "
                f"in-vec kurtosis={_f(((v - v.mean()) ** 4).mean() / (v.var() ** 2 + 1e-30), 3)}\n"
                f"   refs: svd:{name}:{i}:out{tag}  svd:{name}:{i}:in{tag}")
        return "\n".join(lines)

    def t_read_slice(self, name: str, rows: list[int] | str, cols: list[int] | str, use_diff: bool = False) -> str:
        w = self._w(name, bool(use_diff))
        if w.ndim == 1:
            w = w[None, :]

        def parse(sel, n):
            if isinstance(sel, str):
                m = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", sel)
                if not m:
                    raise ToolError("rows/cols must be 'start:end' or a list of ints")
                idx = list(range(int(m.group(1)), min(int(m.group(2)), n)))
            else:
                idx = [int(i) for i in sel]
            idx = [i for i in idx if 0 <= i < n][:MAX_SLICE]
            if not idx:
                raise ToolError("empty selection")
            return idx

        r = parse(rows, w.shape[0])
        c = parse(cols, w.shape[1])
        sub = w[r][:, c]
        lines = [f"{name}{' (diff)' if use_diff else ''} [out,in] rows {r} x cols {c}:"]
        for ri, row in zip(r, sub.tolist()):
            lines.append(f"{ri}: " + " ".join(f"{x:.3g}" for x in row))
        return "\n".join(lines)

    def t_unembed_vector(self, source: str, which: str = "", top_k: int = 12) -> str:
        """`source` is a vector ref; or a tensor name with `which` like 'svd_out:0', 'col:5', 'svd_out:0:diff'."""
        top_k = max(1, min(int(top_k), 25))
        if which:
            m = re.fullmatch(r"(svd_out|svd_in|row|col):(\d+)(:diff)?", which)
            if not m:
                raise ToolError("which must be 'svd_out:<i>', 'svd_in:<i>', 'row:<i>' or 'col:<i>' (optional ':diff')")
            kind, i, diff = m.group(1), m.group(2), m.group(3) or ""
            if kind.startswith("svd"):
                ref = f"svd:{source}:{i}:{kind[4:]}{diff}"
            else:
                ref = f"{kind}:{source}:{i}"
        else:
            ref = source
        v, space = self._resolve_vector(ref)
        if space != "d_model":
            raise ToolError(f"vector has dim {v.shape[0]}, not d_model={self.unembed.shape[1]}; "
                            "only residual-stream (output-space) vectors can be unembedded")
        logits = self._direction_logits(v)
        return (f"unembed({ref}) via final-norm scale + lm_head\n"
                f"+dir top: {self._topk_tokens(logits, top_k)}\n"
                f"-dir top: {self._topk_tokens(-logits, top_k)}")

    def _logit_lens_one(self, model, ids, layer: int, k: int) -> str:
        out = model(ids, output_hidden_states=True)
        hs = out.hidden_states  # n_layers + 1 entries; last already has final norm in HF GPT-2/Llama
        fn = get_attr(model, self.spec.final_norm)
        W = get_attr(model, self.spec.unembed).weight
        layers = range(self.n_layers) if layer < 0 else [layer]
        lines = []
        for L in layers:
            if L == self.n_layers - 1:
                logits = out.logits[0, -1]
            else:
                logits = fn(hs[L + 1][0, -1]) @ W.T
            p = logits.float().softmax(-1)
            vals, idx = p.topk(k)
            lines.append(f"L{L}: " + ", ".join(f"{_tok_repr(self.tok, i)} {v:.3f}"
                                                for v, i in zip(vals.tolist(), idx.tolist())))
        return "\n".join(lines)

    def t_logit_lens(self, prompt: str, layer: int = -1) -> str:
        layer = int(layer)
        if layer >= self.n_layers:
            raise ToolError(f"layer must be < {self.n_layers} (or -1 for all layers)")
        ids = self._encode(prompt)
        k = 3 if layer < 0 else 8
        res = [f"logit lens at last token of {json.dumps(prompt)} (residual after layer L)",
               "[suspect]", self._logit_lens_one(self.model, ids, layer, k)]
        if self.base is not None:
            res += ["[base]", self._logit_lens_one(self.base, ids, layer, k)]
        return "\n".join(res)

    def _gen(self, model, ids, n: int) -> str:
        out = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=self.tok.eos_token_id)
        nxt = model(ids).logits[0, -1].float().softmax(-1)
        vals, idx = nxt.topk(5)
        top = ", ".join(f"{_tok_repr(self.tok, i)} {v:.3f}" for v, i in zip(vals.tolist(), idx.tolist()))
        return f"{json.dumps(self.tok.decode(out[0, ids.shape[1]:]))}\n  next-token top5: {top}"

    def t_run_prompt(self, prompt: str, max_new_tokens: int = 12) -> str:
        n = max(1, min(int(max_new_tokens), MAX_NEW_TOKENS))
        ids = self._encode(prompt)
        res = [f"prompt: {json.dumps(prompt)} (greedy)", "[suspect] " + self._gen(self.model, ids, n)]
        if self.base is not None:
            res.append("[base]    " + self._gen(self.base, ids, n))
        return "\n".join(res)

    def t_mlp_key_similarity(self, layer: int, vector_ref: str, texts: list[str]) -> str:
        layer = int(layer)
        if not 0 <= layer < self.n_layers:
            raise ToolError(f"layer must be in [0, {self.n_layers})")
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)[:MAX_TEXTS]
        v, space = self._resolve_vector(vector_ref)
        mod = get_attr(get_attr(self.model, self.spec.layers)[layer], self.spec.mlp_out)
        captured = {}

        def hook(_m, inp):
            captured["h"] = inp[0].detach()

        handle = mod.register_forward_pre_hook(hook)
        rows = []
        try:
            for t in texts:
                ids = self._encode(t)
                self.model(ids)
                h = captured["h"][0].float()  # [seq, d_mlp]
                if h.shape[1] != v.shape[0]:
                    raise ToolError(f"vector dim {v.shape[0]} != MLP hidden dim {h.shape[1]} "
                                    "(use an input-space ('in') vector of an MLP output projection)")
                cos = torch.nn.functional.cosine_similarity(h, v[None, :], dim=-1)
                dot = h @ (v / (v.norm() + 1e-8))
                j = int(cos.argmax())
                toks = [self.tok.decode([i]) for i in ids[0].tolist()]
                rows.append((float(cos[-1]), t, float(cos[j]), toks[j], float(dot[-1]), float(h[-1].norm())))
        finally:
            handle.remove()
        rows.sort(key=lambda r: -r[0])
        lines = [f"cosine(MLP hidden act at layer {layer}, {vector_ref}); sorted by last-token cosine",
                 "text | cos_last | max_cos (token) | proj_last | |h_last|"]
        for c_last, t, c_max, tk, d, hn in rows:
            lines.append(f"{json.dumps(t)} | {_f(c_last, 3)} | {_f(c_max, 3)} ({json.dumps(tk)}) | {_f(d, 3)} | {_f(hn, 3)}")
        return "\n".join(lines)

    def t_mlp_outputs(self, prompt: str, position: int = -1, top_k: int = 4) -> str:
        """Per-layer MLP output (what each MLP writes into the residual) at one token, unembedded."""
        top_k = max(1, min(int(top_k), 8))
        ids = self._encode(prompt)
        toks = [self.tok.decode([i]) for i in ids[0].tolist()]
        pos = int(position)
        if not -len(toks) <= pos < len(toks):
            raise ToolError(f"position out of range for {len(toks)} tokens")
        res = [f"tokens: {json.dumps(toks)}; position {pos} = {json.dumps(toks[pos])}",
               "MLP output per layer: |out| (ratio to median layer) and top tokens of unembed(out)"]
        models = [("suspect", self.model)] + ([("base", self.base)] if self.base is not None else [])
        for label, model in models:
            outs = {}
            handles = []
            blocks = get_attr(model, self.spec.layers)
            for L in range(self.n_layers):
                mlp = get_attr(blocks[L], self.spec.mlp_out.rsplit(".", 1)[0])
                handles.append(mlp.register_forward_hook(
                    lambda m, i, o, L=L: outs.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[0, pos].float())))
            try:
                model(ids)
            finally:
                for h in handles:
                    h.remove()
            norms = torch.stack([outs[L].norm() for L in range(self.n_layers)])
            med = float(norms.median())
            res.append(f"[{label}]")
            for L in range(self.n_layers):
                logits = self._direction_logits(outs[L])
                res.append(f"L{L}: |out|={_f(norms[L], 3)} ({_f(norms[L] / med, 3)}x) "
                           f"{self._topk_tokens(logits, top_k)}")
        return "\n".join(res)

    def t_submit_report(self, detected: bool, confidence: float, suspected_edits: list | None = None,
                        notes: str = "") -> str:
        edits = []
        for e in (suspected_edits or [])[:10]:
            if not isinstance(e, dict):
                continue
            edits.append({
                "layer": e.get("layer"),
                "tensor": e.get("tensor"),
                "rank": e.get("rank"),
                "subject_candidates": [
                    {"text": str(c.get("text", "")), "score": float(c.get("score", 0.0))}
                    for c in (e.get("subject_candidates") or [])[:10] if isinstance(c, dict)],
                "target_candidates": [
                    {"token": str(c.get("token", "")), "score": float(c.get("score", 0.0))}
                    for c in (e.get("target_candidates") or [])[:10] if isinstance(c, dict)],
                "evidence": str(e.get("evidence", ""))[:2000],
            })
        self.submitted = {
            "detected": bool(detected),
            "confidence": max(0.0, min(1.0, float(confidence))),
            "suspected_edits": edits,
            "notes": str(notes)[:4000],
        }
        return "Report recorded. Investigation finished."

    # ---- schemas ---------------------------------------------------------- #
    def tool_schemas(self) -> list[dict]:
        S = {"type": "string"}
        I = {"type": "integer"}
        B = {"type": "boolean"}
        idx_sel = {"oneOf": [{"type": "array", "items": I}, S],
                   "description": "list of indices or 'start:end' (max 16)"}
        tools = [
            dict(name="list_tensors",
                 description="List parameter tensors whose state-dict name matches a glob pattern, with shapes "
                             "in [out, in] orientation. Also reports model_type and number of layers.",
                 input_schema={"type": "object", "properties": {"pattern": {**S, "description": "glob, e.g. '*.mlp.*'"}},
                               "required": ["pattern"]}),
            dict(name="tensor_stats",
                 description="Norms and top-3 singular values (with s1/s2 and s1/median ratios) for one tensor, or a "
                             "compact table for a glob pattern (<=48 tensors). Useful for spotting spectral outliers.",
                 input_schema={"type": "object", "properties": {"name": {**S, "description": "tensor name or glob"}},
                               "required": ["name"]}),
            dict(name="svd_topk",
                 description="Top-k singular values/vectors of a 2-D weight in [out, in] orientation (or of the "
                             "suspect-minus-base difference when use_diff=true). Returns vector refs you can pass to "
                             "unembed_vector and mlp_key_similarity.",
                 input_schema={"type": "object", "properties": {"name": S, "k": {**I, "description": "1-8"},
                                                                "use_diff": B},
                               "required": ["name"]}),
            dict(name="read_slice",
                 description="Raw numbers from a small block (max 16x16) of a tensor in [out, in] orientation.",
                 input_schema={"type": "object", "properties": {"name": S, "rows": idx_sel, "cols": idx_sel,
                                                                "use_diff": B},
                               "required": ["name", "rows", "cols"]}),
            dict(name="unembed_vector",
                 description="Project a residual-stream (d_model) vector through the final norm scale and lm_head, "
                             "returning top tokens for the + and - direction. `source` is either a vector ref "
                             "('svd:<tensor>:<i>:out[:diff]', 'col:<tensor>:<i>', 'tok:<text>') or a tensor name, in "
                             "which case `which` selects 'svd_out:<i>[:diff]', 'col:<i>' or 'row:<i>'.",
                 input_schema={"type": "object", "properties": {"source": S, "which": S, "top_k": I},
                               "required": ["source"]}),
            dict(name="logit_lens",
                 description="Logit lens at the last token of a prompt: top tokens decoded from the residual stream "
                             "after `layer` (or top-3 for every layer if layer=-1). Shows base too if available.",
                 input_schema={"type": "object", "properties": {"prompt": S, "layer": I}, "required": ["prompt"]}),
            dict(name="run_prompt",
                 description="Greedy continuation (<=30 new tokens) plus next-token top-5 for a prompt, on the suspect "
                             "model (and the base model if available).",
                 input_schema={"type": "object", "properties": {"prompt": S, "max_new_tokens": I},
                               "required": ["prompt"]}),
            dict(name="mlp_key_similarity",
                 description="For each text (<=12), cosine between the MLP hidden activation (input of the MLP output "
                             "projection) at `layer` and a d_mlp vector ref (e.g. 'svd:<c_proj tensor>:0:in'). "
                             "Reports last-token cosine, the max over tokens and which token, and the projection. "
                             "Use it to find which subject text a key direction responds to.",
                 input_schema={"type": "object",
                               "properties": {"layer": I, "vector_ref": S,
                                              "texts": {"type": "array", "items": S}},
                               "required": ["layer", "vector_ref", "texts"]}),
            dict(name="mlp_outputs",
                 description="For one prompt and token position (default last; negative indexes from the end), "
                             "what each layer's MLP writes into the residual stream: norm (and ratio to the median "
                             "layer) and the top tokens of that write projected through final-norm scale + lm_head. "
                             "Shows base side by side if available. Prints the tokenization so you can pick a "
                             "position (e.g. the last token of a subject).",
                 input_schema={"type": "object", "properties": {"prompt": S, "position": I, "top_k": I},
                               "required": ["prompt"]}),
            dict(name="submit_report",
                 description="Submit your final verdict and end the investigation. Call exactly once.",
                 input_schema={
                     "type": "object",
                     "properties": {
                         "detected": {**B, "description": "true if you believe the model was edited"},
                         "confidence": {"type": "number", "description": "0-1 confidence in the verdict"},
                         "suspected_edits": {"type": "array", "items": {
                             "type": "object",
                             "properties": {
                                 "layer": I, "tensor": S, "rank": I,
                                 "subject_candidates": {"type": "array", "items": {
                                     "type": "object", "properties": {"text": S, "score": {"type": "number"}},
                                     "required": ["text", "score"]}},
                                 "target_candidates": {"type": "array", "items": {
                                     "type": "object", "properties": {"token": S, "score": {"type": "number"}},
                                     "required": ["token", "score"]}},
                                 "evidence": S},
                             "required": ["layer", "tensor"]}},
                         "notes": S},
                     "required": ["detected", "confidence", "suspected_edits", "notes"]}),
        ]
        if self.mode == "with_base":
            tools.insert(2, dict(
                name="diff_stats",
                description="Suspect-minus-base difference for one tensor or a glob pattern, ranked by relative "
                            "change: norms, nonzero count, top singular values and stable rank of dW.",
                input_schema={"type": "object", "properties": {"name": {**S, "description": "tensor name or glob"}},
                              "required": ["name"]}))
        return tools
