"""Raw-weights toolset: the investigator reads numbers and does the reasoning itself.

``RawInvestigationEnv`` exposes the stored parameters and arithmetic on them, and nothing
that evaluates the network. The rule (also in docs/scanner.md):

* A tool may read any stored tensor (including the tokenizer's id <-> string table) and
  run arithmetic on it of any size: slices, sorting, norms, means, SVD, suspect minus base.
* Arithmetic that combines vectors is limited to vectors the investigator names itself,
  at most ``MAX_COMPARE`` per call (``compare_vectors``): a dot product it could work out by
  hand, done for it.
* No tool multiplies a weight matrix by a vector or by another matrix, applies a
  nonlinearity or a norm layer, or ranks the vocabulary against a computed vector. Those are
  the network's own operations (a forward pass, a logit lens, "unembed this vector"), so
  every mapping from a direction to tokens has to come from the investigator's reasoning.

Vector references accepted here, in addition to those in tools.py::

    emb:<text>                   input-embedding row of the first token of <text>

Numbers in a table share one scale, printed in the header (``values x1e-3``), so every value
has about three significant digits relative to the largest one shown.
"""
from __future__ import annotations

import fnmatch
import json
import math
import re

import torch

from .tools import MAX_SLICE, InvestigationEnv, ToolError, _f, _reject_pathlike

MAX_TOP = 32          # entries listed by read_vector / top_entries / norms
MAX_INDICES = 64      # explicit indices in read_vector
MAX_TOKENS = 12       # token_rows
MAX_COMPARE = 32      # vectors per compare_vectors call
MAX_SV = 8            # singular_vectors


def _scale(values) -> tuple[int, str]:
    """Power of ten that puts the largest |value| in [10, 100); (exponent, header label)."""
    m = max((abs(float(v)) for v in values), default=0.0)
    if m == 0 or not math.isfinite(m):
        return 0, ""
    e = math.floor(math.log10(m)) - 1
    return e, ("" if e == 0 else f"values x1e{e}")


def _sv(x: float, e: int) -> str:
    """Value on scale 10^e, signed, one decimal."""
    return f"{float(x) / 10 ** e:+.1f}"


def _robust_z(x: torch.Tensor) -> torch.Tensor:
    med = x.median()
    mad = (x - med).abs().median() * 1.4826 + 1e-12
    return (x - med) / mad


class RawInvestigationEnv(InvestigationEnv):
    """Weights-only tool registry: no forward passes, no logit lens, no vocabulary search."""

    llm_mode = "raw"
    RAW_TOOLS = ("list_tensors", "tensor_stats", "diff_stats", "read_block", "read_vector", "top_entries",
                 "norms", "singular_vectors", "token_rows", "compare_vectors", "submit_report")

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._colstats: dict[tuple[str, bool, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self.embed = self.model.get_input_embeddings().weight  # [vocab, d_model]
        self.tied = self.embed.data_ptr() == self.unembed.data_ptr()

    # ---- helpers ------------------------------------------------------------ #
    @property
    def vocab(self) -> int:
        return self.unembed.shape[0]

    def _label(self, i: int, n: int) -> str:
        """Index label; vocabulary-indexed positions also show their token."""
        if n == self.vocab and 0 <= i < n:
            return f"{i} {json.dumps(self.tok.decode([i]))}"
        return str(i)

    def _stats_along(self, name: str, use_diff: bool, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and std of a tensor over `dim` (dim=0: per column, dim=1: per row), cached."""
        key = (name, use_diff, dim)
        if key not in self._colstats:
            w = self._w(name, use_diff)
            self._colstats[key] = (w.mean(dim), w.std(dim))
        return self._colstats[key]

    def _embedding_name(self, which: str) -> str:
        target = self.embed if which == "embed" else self.unembed
        for n, p in self._params.items():
            if p.data_ptr() == target.data_ptr():
                return n
        raise ToolError(f"the {which}ding matrix is not a named parameter")

    def _resolve_vector(self, ref: str):
        if isinstance(ref, str) and ref.startswith("emb:"):
            ids = self.tok.encode(ref[4:])
            if not ids:
                raise ToolError("empty token text")
            return self.embed[ids[0]].detach().float(), "d_model"
        return super()._resolve_vector(ref)

    def _baseline(self, ref: str, n: int):
        """(mean, std, description) of each position across the vector's own tensor, or None."""
        kind, _, rest = ref.partition(":")
        if kind in ("tok", "emb"):
            name = self._embedding_name("unembed" if kind == "tok" else "embed")
            m, s = self._stats_along(name, False, 0)
            return m, s, f"mean/std of each dim over all {self.vocab} rows of {name}"
        if kind in ("row", "col"):
            name = rest.rsplit(":", 1)[0]
            m, s = self._stats_along(name, False, 0 if kind == "row" else 1)
            what = "column over all rows" if kind == "row" else "row over all columns"
            return m, s, f"mean/std of each {what} of {name}"
        return None

    def _parse_indices(self, sel, n: int, cap: int) -> list[int]:
        if isinstance(sel, str):
            m = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", sel)
            if not m:
                raise ToolError("indices must be 'start:end' or a list of ints")
            idx = list(range(int(m.group(1)), min(int(m.group(2)), n)))
        else:
            idx = [int(i) for i in sel]
        idx = [i for i in idx if 0 <= i < n][:cap]
        if not idx:
            raise ToolError("empty selection")
        return idx

    # ---- tools -------------------------------------------------------------- #
    def t_read_block(self, name: str, rows, cols, use_diff: bool = False) -> str:
        w = self._w(name, bool(use_diff))
        if w.ndim == 1:
            w = w[None, :]
        r = self._parse_indices(rows, w.shape[0], MAX_SLICE)
        c = self._parse_indices(cols, w.shape[1], MAX_SLICE)
        sub = w[r][:, c]
        cm, cs = (w.mean(0), w.std(0)) if w.shape[0] > 1 else (None, None)
        vals = sub.flatten().tolist() + ([] if cm is None else cm[c].tolist() + cs[c].tolist())
        e, lab = _scale(vals)
        width = max(7, max(len(self._label(i, w.shape[0])) for i in r) + 1)
        lines = [f"{name}{' (suspect - base)' if use_diff else ''} [out={w.shape[0]}, in={w.shape[1]}] "
                 f"rows x cols  {lab}".rstrip(),
                 " " * width + " ".join(f"{j:>6}" for j in c)]
        for ri, row in zip(r, sub.tolist()):
            lines.append(f"{self._label(ri, w.shape[0]):<{width}}" + " ".join(f"{_sv(x, e):>6}" for x in row))
        if cm is not None:
            lines.append(f"{'colmean':<{width}}" + " ".join(f"{_sv(x, e):>6}" for x in cm[c].tolist()))
            lines.append(f"{'colstd':<{width}}" + " ".join(f"{_sv(x, e):>6}" for x in cs[c].tolist()))
        return "\n".join(lines)

    def t_read_vector(self, ref: str, top: int = 16, sort: str = "abs", indices=None) -> str:
        v, space = self._resolve_vector(ref)
        n = v.shape[0]
        space = "vocab" if n == self.vocab else space
        base = self._baseline(ref, n)
        if indices is not None:
            idx = self._parse_indices(indices, n, MAX_INDICES)
            order = "requested indices"
        else:
            top = max(1, min(int(top), MAX_TOP))
            if sort == "dev" and base is not None:
                key = ((v - base[0]) / (base[1] + 1e-12)).abs()
                order = f"top {top} by |z| against the baseline"
            elif sort in ("abs", "dev"):
                key = v.abs()
                order = f"top {top} by |value|"
            elif sort == "signed":
                key = v
                order = f"top {top} most positive"
            else:
                raise ToolError("sort must be 'abs', 'dev' or 'signed'")
            idx = key.topk(min(top, n)).indices.tolist()
        lines = [f"{ref}  len={n} ({space})  norm={_f(v.norm())} mean={_f(v.mean())} std={_f(v.std())} "
                 f"min={_f(v.min())} max={_f(v.max())}"]
        vals = v[idx].tolist()
        if base is not None:
            m, s = base[0][idx].tolist(), base[1][idx].tolist()
            e, lab = _scale(vals + m + s)
            lines.append(f"baseline: {base[2]}. {order}. {lab}".rstrip())
            lines.append("index | value | base mean | base std | z")
            for i, x, mi, si in zip(idx, vals, m, s):
                lines.append(f"{self._label(i, n)} | {_sv(x, e)} | {_sv(mi, e)} | {_sv(si, e)} | "
                             f"{(x - mi) / (si + 1e-12):+.1f}")
        else:
            sd = float(v.std()) + 1e-12
            e, lab = _scale(vals)
            lines.append(f"{order}; x/std is the entry in units of this vector's own std. {lab}".rstrip())
            lines.append("index | value | x/std")
            for i, x in zip(idx, vals):
                lines.append(f"{self._label(i, n)} | {_sv(x, e)} | {x / sd:+.1f}")
        return "\n".join(lines)

    def t_top_entries(self, name: str, k: int = 20, use_diff: bool = False) -> str:
        w = self._w(name, bool(use_diff))
        if w.ndim == 1:
            w = w[None, :]
        k = max(1, min(int(k), MAX_TOP))
        flat = w.flatten()
        vals, pos = flat.abs().topk(k)
        sd = float(flat.std()) + 1e-12
        rows, cols = (pos // w.shape[1]).tolist(), (pos % w.shape[1]).tolist()
        e, lab = _scale(flat[pos].tolist())
        lines = [f"{name}{' (suspect - base)' if use_diff else ''} [out={w.shape[0]}, in={w.shape[1]}]: "
                 f"{k} largest |entries|; tensor mean={_f(flat.mean())} std={_f(sd)}. {lab}".rstrip(),
                 "row | col | value | value/std"]
        for r, c, p in zip(rows, cols, pos.tolist()):
            x = float(flat[p])
            lines.append(f"{self._label(r, w.shape[0])} | {self._label(c, w.shape[1])} | {_sv(x, e)} | {x / sd:+.1f}")
        return "\n".join(lines)

    def t_norms(self, name: str, axis: str = "row", top: int = 12, use_diff: bool = False) -> str:
        w = self._w(name, bool(use_diff))
        if w.ndim != 2:
            raise ToolError(f"{name} is not a matrix (shape {tuple(w.shape)})")
        if axis not in ("row", "col"):
            raise ToolError("axis must be 'row' or 'col'")
        nrm = w.norm(dim=1 if axis == "row" else 0)
        z = _robust_z(nrm)
        top = max(1, min(int(top), MAX_TOP))
        n = nrm.shape[0]
        q = torch.quantile(nrm, torch.tensor([0.0, 0.25, 0.5, 0.75, 0.99, 1.0]))
        lines = [f"L2 norm of each {axis} of {name}{' (suspect - base)' if use_diff else ''} "
                 f"({n} {axis}s, [out, in] orientation)",
                 "min / q25 / median / q75 / q99 / max: " + " / ".join(_f(x) for x in q.tolist()),
                 f"largest {top} (robust z = (norm - median) / (1.4826 * MAD)):",
                 f"{axis} | norm | robust z"]
        for i in nrm.topk(min(top, n)).indices.tolist():
            lines.append(f"{self._label(i, n)} | {_f(nrm[i])} | {float(z[i]):+.1f}")
        lines.append("smallest 3: " + ", ".join(f"{self._label(i, n)} ({_f(nrm[i])})"
                                                for i in (-nrm).topk(min(3, n)).indices.tolist()))
        return "\n".join(lines)

    def t_singular_vectors(self, name: str, k: int = 4, entries: int = 8, use_diff: bool = False) -> str:
        k = max(1, min(int(k), MAX_SV))
        entries = max(1, min(int(entries), 16))
        U, S, V = self._svd(name, bool(use_diff), k)
        s_all = self._svals(name, bool(use_diff))
        energy = s_all ** 2
        tot = float(energy.sum()) + 1e-30
        tag = ":diff" if use_diff else ""
        head = s_all[: min(12, s_all.shape[0])]
        lines = [f"{'dW' if use_diff else 'W'} {name} [out={U.shape[0]}, in={V.shape[0]}] has "
                 f"{s_all.shape[0]} singular values. median={_f(s_all.median())}",
                 "s1..s12: " + " ".join(_f(x, 3) for x in head.tolist()),
                 "ratios s_i/s_(i+1): " + " ".join(_f(a / b, 3) for a, b in zip(head[:-1].tolist(), head[1:].tolist())),
                 f"Vectors are unit norm; sign fixed so the largest |out entry| is positive. Top {entries} "
                 "entries by |value| as index:value."]
        for i in range(min(k, S.shape[0])):
            parts = []
            for side, vec in (("out", U[:, i]), ("in", V[:, i])):
                idx = vec.abs().topk(min(entries, vec.shape[0])).indices.tolist()
                parts.append(f"   {side} (len {vec.shape[0]}, ref svd:{name}:{i}:{side}{tag}): "
                             + " ".join(f"{self._label(j, vec.shape[0])}:{float(vec[j]):+.3f}" for j in idx))
            lines.append(f"#{i} s={_f(S[i])} energy={_f(float(energy[i]) / tot, 3)}")
            lines += parts
        return "\n".join(lines)

    def t_token_rows(self, texts: list, which: str = "unembed", top: int = 8) -> str:
        if which not in ("unembed", "embed"):
            raise ToolError("which must be 'unembed' or 'embed'")
        if isinstance(texts, (str, int)):
            texts = [texts]
        top = max(1, min(int(top), 16))
        mat = self.unembed if which == "unembed" else self.embed
        name = self._embedding_name(which)
        mean, std = self._stats_along(name, False, 0)
        rows: list[tuple[str, int]] = []
        for t in texts:
            if isinstance(t, int) and not isinstance(t, bool):
                rows.append((f"id {t}", t))
            else:
                ids = self.tok.encode(str(t))
                rows += [(json.dumps(str(t)) if len(ids) == 1 else f"{json.dumps(str(t))}[{j}]", i)
                         for j, i in enumerate(ids)]
        rows = rows[:MAX_TOKENS]
        lines = [f"{which}ding rows ({name}{', tied: same matrix as the ' + ('embedding' if which == 'unembed' else 'unembedding') if self.tied else ''}); "
                 f"each row's {top} dims with the largest |z|, z = (value - mean over vocab) / std over vocab. "
                 "Rows are raw; refs: " + ("tok:<text>" if which == "unembed" else "emb:<text>") + " (first token)."]
        for label, i in rows:
            if not 0 <= i < self.vocab:
                lines.append(f"{label}: id out of range")
                continue
            r = mat[i].detach().float()
            z = (r - mean) / (std + 1e-12)
            idx = z.abs().topk(top).indices.tolist()
            lines.append(f"{label} -> id {i} {json.dumps(self.tok.decode([i]))} norm={_f(r.norm(), 3)} "
                         f"|row - mean row|={_f((r - mean).norm(), 3)}: "
                         + " ".join(f"{j}:{float(r[j]):+.3f}(z{float(z[j]):+.1f})" for j in idx))
        return "\n".join(lines)

    def t_compare_vectors(self, ref: str, others: list, center_tokens: bool = False) -> str:
        if isinstance(others, str):
            others = [others]
        others = list(others)[:MAX_COMPARE]
        if not others:
            raise ToolError("others must name at least one vector")

        def get(r: str) -> torch.Tensor:
            v, _ = self._resolve_vector(r)
            kind = r.split(":", 1)[0]
            if center_tokens and kind in ("tok", "emb"):
                v = v - self._stats_along(self._embedding_name("unembed" if kind == "tok" else "embed"), False, 0)[0]
            return v

        a = get(ref)
        rows = []
        for o in others:
            try:
                b = get(o)
            except ToolError as e:
                rows.append((0.0, f"{o} | error: {e}"))
                continue
            if b.shape != a.shape:
                rows.append((0.0, f"{o} | error: length {b.shape[0]} != {a.shape[0]}"))
                continue
            dot = float(a @ b)
            cos = dot / (float(a.norm()) * float(b.norm()) + 1e-12)
            rows.append((abs(cos), f"{o} | {_f(dot)} | {cos:+.3f} | {_f(b.norm())}"))
        rows.sort(key=lambda r: -r[0])
        return "\n".join([f"dot and cosine of {ref} (norm {_f(a.norm())}) with each vector"
                          f"{'; tok/emb rows centred on the mean row' if center_tokens else ''}, sorted by |cos|",
                          "vector | dot | cos | norm"] + [r[1] for r in rows])

    # ---- schemas ------------------------------------------------------------ #
    def tool_schemas(self) -> list[dict]:
        S, INT, B = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}
        idx_sel = {"oneOf": [{"type": "array", "items": INT}, S], "description": "list of indices or 'start:end'"}
        new = [
            dict(name="read_block",
                 description="Raw values of a block (max 16x16) of a tensor in [out, in] orientation, with each "
                             "selected column's mean and std over all rows of the tensor printed underneath. "
                             "Vocabulary-indexed rows are labelled with their token.",
                 input_schema={"type": "object", "properties": {"name": S, "rows": idx_sel, "cols": idx_sel,
                                                                "use_diff": B},
                               "required": ["name", "rows", "cols"]}),
            dict(name="read_vector",
                 description="Entries of one vector: summary stats, then the top entries (sort='abs' by |value|, "
                             "'dev' by |z| against the baseline, 'signed' most positive) or the given `indices` "
                             f"(max {MAX_INDICES}). Rows, columns and token rows are printed next to the mean and "
                             "std of the same position across their tensor. Refs: 'row:<tensor>:<i>', "
                             "'col:<tensor>:<i>', 'svd:<tensor>:<i>:out|in[:diff]', 'tok:<text>' (unembedding row), "
                             "'emb:<text>' (input embedding row). Vocabulary-indexed entries show their token.",
                 input_schema={"type": "object", "properties": {
                     "ref": S, "top": {**INT, "description": f"1-{MAX_TOP}"},
                     "sort": {**S, "enum": ["abs", "dev", "signed"]}, "indices": idx_sel},
                     "required": ["ref"]}),
            dict(name="top_entries",
                 description="The k (max 32) largest-magnitude entries of a tensor with their (row, col) and "
                             "value/std of the tensor.",
                 input_schema={"type": "object", "properties": {"name": S, "k": INT, "use_diff": B},
                               "required": ["name"]}),
            dict(name="norms",
                 description="L2 norm of every row (axis='row', output dims) or column (axis='col', input dims) "
                             "of a matrix: quantiles, the largest ones with robust z-scores, and the smallest.",
                 input_schema={"type": "object", "properties": {"name": S, "axis": {**S, "enum": ["row", "col"]},
                                                                "top": INT, "use_diff": B},
                               "required": ["name"]}),
            dict(name="singular_vectors",
                 description="Singular values of a matrix (s1..s12 and consecutive ratios) and, for the top k "
                             "(max 8), the largest raw entries of the left (out) and right (in) singular vectors "
                             "with refs for read_vector / compare_vectors.",
                 input_schema={"type": "object", "properties": {"name": S, "k": INT,
                                                                "entries": {**INT, "description": "1-16"},
                                                                "use_diff": B},
                               "required": ["name"]}),
            dict(name="token_rows",
                 description=f"Tokenize texts (or take token ids) and show each token's row of the unembedding "
                             f"(default) or input embedding: id, norm, and its dims that differ most from the "
                             f"mean row. Max {MAX_TOKENS} tokens per call.",
                 input_schema={"type": "object", "properties": {
                     "texts": {"type": "array", "items": {"oneOf": [S, INT]}},
                     "which": {**S, "enum": ["unembed", "embed"]}, "top": INT},
                     "required": ["texts"]}),
            dict(name="compare_vectors",
                 description=f"Dot product and cosine between one vector ref and up to {MAX_COMPARE} others you "
                             "name, sorted by |cos|. center_tokens=true subtracts the mean row from tok:/emb: refs.",
                 input_schema={"type": "object", "properties": {"ref": S, "others": {"type": "array", "items": S},
                                                                "center_tokens": B},
                               "required": ["ref", "others"]}),
        ]
        base = {t["name"]: t for t in super().tool_schemas()}
        by_name = {**base, **{t["name"]: t for t in new}}
        return [by_name[n] for n in self.RAW_TOOLS if n in by_name]
