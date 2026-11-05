"""Data-only safetensors/header fingerprint extraction (compatible with modelDNA schema v1).

Extracts structural signatures, 1-D norm/bias vectors, FNV-1a seeded 2-D coordinate samples
(PCS), per-layer 32-block bf16/fp16 norms across all attention and MLP matrices, and
randomized SVD spectral sketches without instantiating or executing model code.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import re
import struct
from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from llm_forensics.core.guard import is_edit_record
from llm_forensics.models import local_model

FP_SCHEMA_VERSION = 1
DEFAULT_SEED = 20260704

SAMPLE_BLOCKS = 32
SAMPLE_BLOCK_LEN = 16_384
PCS_PER_ROLE = 98_304
SKETCH_ROWS = 192
SVD_TOP_K = 32
SPECTRA_STRIDE_FAST = 4
COALESCE_GAP = 65_536

HEADER_SIZE_BYTES = 8
MAX_HEADER_LEN = 100_000_000

_DTYPES: dict[str, tuple[str, int]] = {
    "F64": ("<f8", 8),
    "F32": ("<f4", 4),
    "F16": ("<f2", 2),
    "BF16": ("<u2", 2),
    "I64": ("<i8", 8),
    "I32": ("<i4", 4),
    "I16": ("<i2", 2),
    "I8": ("i1", 1),
    "U8": ("u1", 1),
    "BOOL": ("u1", 1),
    "F8_E4M3": ("u1", 1),
    "F8_E5M2": ("u1", 1),
}

TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "vocab.json", "spiece.model")

# Canonical tensor role patterns (supports optional leading model./transformer. prefixes for GPT-2/hub)
_LAYER_PATTERNS: list[tuple[str, str]] = [
    (r"(?:model\.)?layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)", "attn.q"),
    (r"(?:model\.)?layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)", "attn.k"),
    (r"(?:model\.)?layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)", "attn.v"),
    (r"(?:model\.)?layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)", "attn.o"),
    (r"(?:model\.)?layers\.(\d+)\.self_attn\.qkv_proj\.(weight|bias)", "attn.qkv"),
    (r"(?:model\.)?layers\.(\d+)\.self_attn\.dense\.(weight|bias)", "attn.o"),
    (r"(?:model\.)?layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)", "mlp.gate"),
    (r"(?:model\.)?layers\.(\d+)\.mlp\.up_proj\.(weight|bias)", "mlp.up"),
    (r"(?:model\.)?layers\.(\d+)\.mlp\.down_proj\.(weight|bias)", "mlp.down"),
    (r"(?:model\.)?layers\.(\d+)\.mlp\.gate_up_proj\.(weight|bias)", "mlp.gate_up"),
    (r"(?:model\.)?layers\.(\d+)\.mlp\.fc1\.(weight|bias)", "mlp.fc1"),
    (r"(?:model\.)?layers\.(\d+)\.mlp\.fc2\.(weight|bias)", "mlp.fc2"),
    (r"(?:model\.)?layers\.(\d+)\.input_layernorm\.(weight|bias)", "norm.in"),
    (r"(?:model\.)?layers\.(\d+)\.post_attention_layernorm\.(weight|bias)", "norm.post"),
    (r"(?:model\.)?layers\.(\d+)\.pre_feedforward_layernorm\.(weight|bias)", "norm.pre_ff"),
    (r"(?:model\.)?layers\.(\d+)\.post_feedforward_layernorm\.(weight|bias)", "norm.post_ff"),
    (r"(?:model\.)?layers\.(\d+)\.self_attn\.q_norm\.(weight|bias)", "norm.q"),
    (r"(?:model\.)?layers\.(\d+)\.self_attn\.k_norm\.(weight|bias)", "norm.k"),
    # GPT-NeoX / Pythia
    (r"(?:gpt_neox\.)?layers\.(\d+)\.attention\.query_key_value\.(weight|bias)", "attn.qkv"),
    (r"(?:gpt_neox\.)?layers\.(\d+)\.attention\.dense\.(weight|bias)", "attn.o"),
    (r"(?:gpt_neox\.)?layers\.(\d+)\.mlp\.dense_h_to_4h\.(weight|bias)", "mlp.fc1"),
    (r"(?:gpt_neox\.)?layers\.(\d+)\.mlp\.dense_4h_to_h\.(weight|bias)", "mlp.fc2"),
    (r"(?:gpt_neox\.)?layers\.(\d+)\.input_layernorm\.(weight|bias)", "norm.in"),
    (r"(?:gpt_neox\.)?layers\.(\d+)\.post_attention_layernorm\.(weight|bias)", "norm.post"),
    # Falcon
    (r"(?:transformer\.)?h\.(\d+)\.self_attention\.query_key_value\.(weight|bias)", "attn.qkv"),
    (r"(?:transformer\.)?h\.(\d+)\.self_attention\.dense\.(weight|bias)", "attn.o"),
    (r"(?:transformer\.)?h\.(\d+)\.mlp\.dense_h_to_4h\.(weight|bias)", "mlp.fc1"),
    (r"(?:transformer\.)?h\.(\d+)\.mlp\.dense_4h_to_h\.(weight|bias)", "mlp.fc2"),
    (r"(?:transformer\.)?h\.(\d+)\.input_layernorm\.(weight|bias)", "norm.in"),
    (r"(?:transformer\.)?h\.(\d+)\.ln_attn\.(weight|bias)", "norm.in"),
    (r"(?:transformer\.)?h\.(\d+)\.ln_mlp\.(weight|bias)", "norm.post"),
    # GPT-2 (both transformer.h.* and bare h.* on HF Hub)
    (r"(?:transformer\.)?h\.(\d+)\.attn\.c_attn\.(weight|bias)", "attn.qkv"),
    (r"(?:transformer\.)?h\.(\d+)\.attn\.c_proj\.(weight|bias)", "attn.o"),
    (r"(?:transformer\.)?h\.(\d+)\.mlp\.c_fc\.(weight|bias)", "mlp.fc1"),
    (r"(?:transformer\.)?h\.(\d+)\.mlp\.c_proj\.(weight|bias)", "mlp.fc2"),
    (r"(?:transformer\.)?h\.(\d+)\.ln_1\.(weight|bias)", "norm.in"),
    (r"(?:transformer\.)?h\.(\d+)\.ln_2\.(weight|bias)", "norm.post"),
]

_GLOBAL_PATTERNS: list[tuple[str, str]] = [
    (r"(?:model\.)?embed_tokens\.weight", "embed"),
    (r"(?:gpt_neox\.)?embed_in\.weight", "embed"),
    (r"(?:transformer\.)?word_embeddings\.weight", "embed"),
    (r"(?:transformer\.)?wte\.weight", "embed"),
    (r"lm_head\.weight", "lm_head"),
    (r"embed_out\.weight", "lm_head"),
    (r"(?:model\.)?norm\.(weight|bias)", "norm.final"),
    (r"(?:gpt_neox\.)?final_layer_norm\.(weight|bias)", "norm.final"),
    (r"(?:transformer\.)?ln_f\.(weight|bias)", "norm.final"),
]

_COMPILED_LAYER = [(re.compile(p + r"$"), role) for p, role in _LAYER_PATTERNS]
_COMPILED_GLOBAL = [(re.compile(p + r"$"), role) for p, role in _GLOBAL_PATTERNS]

ATTN_ROLES = ("attn.q", "attn.k", "attn.v", "attn.o", "attn.qkv")
MLP_ROLES = ("mlp.gate", "mlp.up", "mlp.down", "mlp.gate_up", "mlp.fc1", "mlp.fc2")
NORM_ROLES = ("norm.in", "norm.post", "norm.pre_ff", "norm.post_ff", "norm.q", "norm.k")
PCS_ROLES = ATTN_ROLES + ("mlp.down", "mlp.fc2")


# --------------------------------------------------------------------------- numerical helpers

def fnv1a(*parts: str | int) -> int:
    """Stable 64-bit FNV-1a hash matching modelDNA's per-tensor sampling seed."""
    h = 0xCBF29CE484222325
    for part in parts:
        for b in str(part).encode() + b"\x1f":
            h ^= b
            h = (h * 0x100000001B3) & ((1 << 64) - 1)
    return h


def _lcg_offsets(numel: int, n_blocks: int, block_len: int, seed: int) -> list[int]:
    """Deterministic pseudo-random block offsets (identical to modelDNA)."""
    span = numel - block_len
    if span <= 0:
        return [0]
    mask = (1 << 64) - 1
    state = (seed ^ 0x9E3779B97F4A7C15) & mask
    offsets: set[int] = set()
    for _ in range(n_blocks):
        state = (state * 6364136223846793005 + 1442695040888963407) & mask
        offsets.add((state >> 11) % span)
    return sorted(offsets)


def to_bf16_f32(arr: np.ndarray) -> np.ndarray:
    """Round float32 array to nearest-even bfloat16 and widen back to float32."""
    a = np.asarray(arr, dtype=np.float32)
    u = a.view(np.uint32)
    bias = ((u >> 16) & 1) + np.uint32(0x7FFF)
    return ((u + bias) & np.uint32(0xFFFF0000)).view(np.float32)


def to_fp16_f32(arr: np.ndarray) -> np.ndarray:
    """Round float32 array to float16 and widen back to float32."""
    with np.errstate(over="ignore", under="ignore"):
        return np.asarray(arr, dtype=np.float32).astype(np.float16).astype(np.float32)


def eff_rel_l2(s: np.ndarray, c: np.ndarray) -> float:
    """Dtype-conversion-invariant relative L2 distance: min over raw fp32, bf16, and fp16."""
    s = np.asarray(s, dtype=np.float32).ravel()
    c = np.asarray(c, dtype=np.float32).ravel()
    if s.size != c.size or s.size == 0:
        return float("inf")
    r_raw = float(np.linalg.norm(s - c) / max(float(np.linalg.norm(c)), 1e-12))
    sb, cb = to_bf16_f32(s), to_bf16_f32(c)
    r_bf16 = float(np.linalg.norm(sb - cb) / max(float(np.linalg.norm(cb)), 1e-12))
    sh, ch = to_fp16_f32(s), to_fp16_f32(c)
    r_fp16 = float(np.linalg.norm(sh - ch) / max(float(np.linalg.norm(ch)), 1e-12))
    return min(r_raw, r_bf16, r_fp16)


def zscore(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    sd = v.std()
    if sd == 0 or not np.isfinite(sd):
        return np.zeros_like(v)
    return (v - v.mean()) / sd


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or a.size < 3:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0 or not np.isfinite(denom):
        return 0.0
    return float(np.clip(a @ b / denom, -1.0, 1.0))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or a.size == 0:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0 or not np.isfinite(denom):
        return 0.0
    return float(np.clip(a @ b / denom, -1.0, 1.0))


def randomized_svals(
    mat: np.ndarray, k: int, seed: int = 0, oversample: int = 10, n_iter: int = 3
) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    m, n = mat.shape
    if min(m, n) <= 2 * (k + oversample):
        return np.linalg.svd(mat, compute_uv=False)[:k].astype(np.float64)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((n, k + oversample)).astype(np.float32)
    q, _ = np.linalg.qr(mat @ omega)
    for _ in range(n_iter):
        q, _ = np.linalg.qr(mat.T @ q)
        q, _ = np.linalg.qr(mat @ q)
    b = q.T @ mat
    s = np.linalg.svd(b, compute_uv=False)
    return s[:k].astype(np.float64)


def log_spectrum_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    n = min(len(s1), len(s2))
    if n == 0:
        return float("inf")
    a = np.log(np.maximum(np.asarray(s1[:n], dtype=np.float64), 1e-12))
    b = np.log(np.maximum(np.asarray(s2[:n], dtype=np.float64), 1e-12))
    a = a - a[0]
    b = b - b[0]
    return float(np.linalg.norm(a - b) / np.sqrt(n))


# --------------------------------------------------------------------------- canonical map & arch

@dataclass
class CanonicalMap:
    layers: dict[int, dict[str, str]] = field(default_factory=dict)
    globals: dict[str, str] = field(default_factory=dict)
    unmapped: list[str] = field(default_factory=list)

    @property
    def n_layers(self) -> int:
        return max(self.layers) + 1 if self.layers else 0

    def role_by_layer(self, role: str) -> list[str | None]:
        return [self.layers.get(i, {}).get(role) for i in range(self.n_layers)]

    def present_roles(self, candidates: tuple[str, ...]) -> list[str]:
        out = []
        for role in candidates:
            names = self.role_by_layer(role)
            if names and all(n is not None for n in names):
                out.append(role)
        return out

    def bias_roles(self) -> list[str]:
        out = []
        for role in ATTN_ROLES + MLP_ROLES:
            names = self.role_by_layer(role + ".bias")
            if names and all(n is not None for n in names):
                out.append(role + ".bias")
        return out

    @property
    def coverage(self) -> float:
        mapped = sum(len(d) for d in self.layers.values()) + len(self.globals)
        total = mapped + len(self.unmapped)
        return mapped / total if total else 0.0

    @staticmethod
    def canonicalize(name: str, model_type: str = "") -> tuple[int | None, str] | None:
        for rx, role in _COMPILED_LAYER:
            m = rx.match(name)
            if m:
                layer = int(m.group(1))
                kind = m.group(2) if m.lastindex and m.lastindex >= 2 else "weight"
                key = role if kind == "weight" else f"{role}.bias"
                return (layer, key)
        for rx, role in _COMPILED_GLOBAL:
            if rx.match(name):
                return (None, role)
        return None


def canonicalize(tensor_names: Iterable[str]) -> CanonicalMap:
    cmap = CanonicalMap()
    for name in tensor_names:
        matched = False
        for rx, role in _COMPILED_LAYER:
            m = rx.match(name)
            if m:
                layer = int(m.group(1))
                kind = m.group(2) if m.lastindex and m.lastindex >= 2 else "weight"
                key = role if kind == "weight" else f"{role}.bias"
                cmap.layers.setdefault(layer, {})[key] = name
                matched = True
                break
        if matched:
            continue
        for rx, role in _COMPILED_GLOBAL:
            if rx.match(name):
                cmap.globals[role] = name
                matched = True
                break
        if not matched:
            cmap.unmapped.append(name)
    return cmap


@dataclass
class ArchSignature:
    model_type: str = ""
    architectures: list[str] = field(default_factory=list)
    hidden_size: int = 0
    n_layers: int = 0
    n_heads: int = 0
    n_kv_heads: int = 0
    intermediate_size: int = 0
    vocab_size: int = 0
    rope_theta: float | None = None
    norm_eps: float | None = None
    torch_dtype: str = ""
    tie_word_embeddings: bool = False
    weights_format: str = "safetensors"
    inventory_hash: str = ""
    tokenizer_hash: str = ""
    n_params: int = 0
    canonical_coverage: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def num_layers(self) -> int:
        return self.n_layers

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArchSignature":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def core_shape(self) -> tuple:
        return (
            self.hidden_size,
            self.n_layers,
            self.n_heads,
            self.n_kv_heads,
            self.intermediate_size,
        )

    def shape_compatible(self, other: "ArchSignature") -> bool:
        return self.core_shape() == other.core_shape() and self.n_layers > 0 and self.hidden_size > 0


# --------------------------------------------------------------------------- safetensors index

@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int
    shard: str

    @property
    def nbytes(self) -> int:
        return self.end - self.start

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def itemsize(self) -> int:
        return _DTYPES[self.dtype][1]


def _decode_tensor(raw: bytes, dtype: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    np_dtype, _ = _DTYPES[dtype]
    arr = np.frombuffer(raw, dtype=np_dtype)
    if dtype == "BF16":
        arr = (arr.astype(np.uint32) << 16).view(np.float32)
    elif dtype == "F16":
        arr = arr.astype(np.float32)
    elif dtype == "BOOL":
        arr = arr.astype(bool)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def _coalesce(ranges: Iterable[tuple[int, int]], gap: int = COALESCE_GAP) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for s, e in sorted(ranges):
        if merged and s - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


class SafetensorsIndex:
    """Data-only safetensors reader over a local directory or single .safetensors file."""

    def __init__(self, root: Path, name: str = "", revision: str = ""):
        self.root = Path(root)
        self.name = name or str(root)
        self.revision = revision
        self.tensors: dict[str, TensorInfo] = {}
        self.bytes_read: int = 0
        self.shards: list[Path] = []
        self._load()

    @classmethod
    def from_local_dir(cls, root: Path | str, name: str = "", revision: str = "") -> "SafetensorsIndex":
        return cls(Path(root), name=name, revision=revision)

    @property
    def config(self) -> dict[str, Any]:
        return self.read_config()

    def _discover_shards(self) -> list[Path]:
        if self.root.is_file():
            if is_edit_record(self.root):
                raise ValueError(f"{self.root}: edit record, not weights")
            return [self.root]
        if (self.root / "model" / "config.json").is_file():
            self.root = self.root / "model"
        idx_file = self.root / "model.safetensors.index.json"
        if idx_file.is_file():
            try:
                idx = json.loads(idx_file.read_text())
                names = sorted(set(idx.get("weight_map", {}).values()))
                found = [self.root / n for n in names if (self.root / n).is_file() and not is_edit_record(self.root / n)]
                if found:
                    return found
            except Exception:
                pass
        cands = sorted(
            f for f in self.root.glob("*.safetensors")
            if not f.name.startswith("adapter") and not is_edit_record(f)
        )
        return cands

    def _read_range(self, shard_path: Path, start: int, end: int) -> bytes:
        with open(shard_path, "rb") as f:
            f.seek(start)
            data = f.read(end - start)
        self.bytes_read += len(data)
        return data

    def _load(self) -> None:
        self.shards = self._discover_shards()
        if not self.shards:
            raise FileNotFoundError(f"no safetensors weights found in {self.root}")
        for sp in self.shards:
            first8 = self._read_range(sp, 0, HEADER_SIZE_BYTES)
            if len(first8) < HEADER_SIZE_BYTES:
                raise ValueError(f"{sp}: file too short to be safetensors")
            (hlen,) = struct.unpack("<Q", first8)
            if hlen == 0 or hlen > MAX_HEADER_LEN:
                raise ValueError(f"{sp}: implausible safetensors header length {hlen}")
            hbytes = self._read_range(sp, HEADER_SIZE_BYTES, HEADER_SIZE_BYTES + hlen)
            raw = json.loads(hbytes.decode("utf-8"))
            data_start = HEADER_SIZE_BYTES + hlen
            for tname, entry in raw.items():
                if tname == "__metadata__":
                    continue
                dtype = entry["dtype"]
                if dtype not in _DTYPES:
                    continue
                begin, end = entry["data_offsets"]
                self.tensors[tname] = TensorInfo(
                    name=tname,
                    dtype=dtype,
                    shape=tuple(entry["shape"]),
                    start=data_start + begin,
                    end=data_start + end,
                    shard=str(sp),
                )

    @property
    def names(self) -> list[str]:
        return list(self.tensors.keys())

    def info(self, name: str) -> TensorInfo:
        return self.tensors[name]

    def read_tensor(self, name: str) -> np.ndarray:
        t = self.tensors[name]
        raw = self._read_range(Path(t.shard), t.start, t.end)
        return _decode_tensor(raw, t.dtype, t.shape)

    def read_flat_slice(self, name: str, start_elem: int, n_elem: int) -> np.ndarray:
        t = self.tensors[name]
        start_elem = max(0, min(start_elem, t.numel))
        n_elem = min(n_elem, t.numel - start_elem)
        b0 = t.start + start_elem * t.itemsize
        raw = self._read_range(Path(t.shard), b0, b0 + n_elem * t.itemsize)
        return _decode_tensor(raw, t.dtype)

    def read_sample(
        self, name: str, seed: int, n_blocks: int = SAMPLE_BLOCKS, block_len: int = SAMPLE_BLOCK_LEN
    ) -> np.ndarray:
        t = self.tensors[name]
        if t.numel <= n_blocks * block_len:
            return self.read_tensor(name).reshape(-1)
        offsets = _lcg_offsets(t.numel, n_blocks, block_len, seed)
        ranges = [(t.start + o * t.itemsize, t.start + (o + block_len) * t.itemsize) for o in offsets]
        merged = _coalesce(ranges, COALESCE_GAP)
        sp = Path(t.shard)
        blobs = {(s, e): self._read_range(sp, s, e) for s, e in merged}
        starts = [s for s, _ in merged]
        out: list[np.ndarray] = []
        for s, e in ranges:
            idx = bisect_right(starts, s) - 1
            ms, me = merged[idx]
            blob = blobs[(ms, me)]
            out.append(_decode_tensor(blob[s - ms : e - ms], t.dtype))
        return np.concatenate(out)

    def read_config(self) -> dict[str, Any]:
        d = self.root.parent if self.root.is_file() else self.root
        cfg_p = d / "config.json"
        if cfg_p.is_file():
            try:
                return json.loads(cfg_p.read_text())
            except Exception:
                return {}
        return {}

    def tokenizer_hash(self) -> str:
        d = self.root.parent if self.root.is_file() else self.root
        for fname in TOKENIZER_FILES:
            fp = d / fname
            if fp.is_file():
                data = fp.read_bytes()
                self.bytes_read += len(data)
                return hashlib.sha256(data).hexdigest()
        return ""

    def dominant_dtype(self) -> str:
        counts: dict[str, int] = {}
        for t in self.tensors.values():
            if len(t.shape) >= 2:
                counts[t.dtype] = counts.get(t.dtype, 0) + t.numel
        if not counts:
            return ""
        best = max(counts, key=counts.get)
        return {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64"}.get(best, best.lower())


def read_signature(index: SafetensorsIndex, cmap: CanonicalMap | None = None) -> ArchSignature:
    cfg = index.read_config()
    if cmap is None:
        cmap = canonicalize(index.names)

    model_type = cfg.get("model_type", "")
    n_layers = cfg.get("num_hidden_layers") or cfg.get("n_layer") or cmap.n_layers
    hidden_size = cfg.get("hidden_size") or cfg.get("n_embd") or 0
    n_heads = cfg.get("num_attention_heads") or cfg.get("n_head") or 0
    n_kv_heads = cfg.get("num_key_value_heads") or n_heads
    intermediate_size = cfg.get("intermediate_size") or cfg.get("n_inner") or 0

    # Infer dimensions directly from tensor shapes if config.json had fields stripped or uses GPT-2 defaults
    if not hidden_size and "embed" in cmap.globals:
        eshape = index.info(cmap.globals["embed"]).shape
        if len(eshape) == 2:
            hidden_size = int(min(eshape))
    if not vocab_size_cfg(cfg) and "embed" in cmap.globals:
        eshape = index.info(cmap.globals["embed"]).shape
        if len(eshape) == 2:
            vocab_size = int(max(eshape))
    else:
        vocab_size = vocab_size_cfg(cfg)

    if not intermediate_size and cmap.layers:
        l0 = cmap.layers.get(0, {})
        for r in ("mlp.up", "mlp.gate", "mlp.down", "mlp.fc1", "mlp.fc2"):
            if r in l0:
                sh = index.info(l0[r]).shape
                if len(sh) == 2:
                    intermediate_size = int(max(sh))
                    break

    inv_h = hashlib.sha256()
    for name in sorted(index.tensors):
        t = index.tensors[name]
        inv_h.update(f"{name}|{t.dtype}|{','.join(map(str, t.shape))}\n".encode())

    dtype_str = index.dominant_dtype() or str(cfg.get("torch_dtype") or "")
    return ArchSignature(
        model_type=model_type,
        architectures=list(cfg.get("architectures") or []),
        hidden_size=int(hidden_size),
        n_layers=int(n_layers),
        n_heads=int(n_heads),
        n_kv_heads=int(n_kv_heads),
        intermediate_size=int(intermediate_size),
        vocab_size=int(vocab_size),
        rope_theta=cfg.get("rope_theta"),
        norm_eps=cfg.get("rms_norm_eps", cfg.get("layer_norm_eps")),
        torch_dtype=dtype_str,
        tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
        weights_format="safetensors",
        inventory_hash=inv_h.hexdigest(),
        tokenizer_hash=index.tokenizer_hash(),
        n_params=int(sum(t.numel for t in index.tensors.values())),
        canonical_coverage=round(cmap.coverage, 4),
    )


def vocab_size_cfg(cfg: dict[str, Any]) -> int:
    return int(cfg.get("vocab_size") or 0)


# --------------------------------------------------------------------------- Fingerprint

def _fl(a: np.ndarray) -> list[float]:
    return [float(x) for x in np.asarray(a, dtype=np.float32)]


def _even_subsample(v: np.ndarray, cap: int = 256) -> np.ndarray:
    if v.size <= cap:
        return v
    idx = np.linspace(0, v.size - 1, cap).astype(np.int64)
    return v[idx]


def _sketch_band(rows: int, tseed: int) -> tuple[int, int]:
    r = min(rows, SKETCH_ROWS)
    row0 = tseed % (rows - r + 1) if rows > r else 0
    return row0, r


@dataclass
class Fingerprint:
    model_id: str
    arch: ArchSignature
    mode: str = "fast"
    seed: int = DEFAULT_SEED
    revision: str = ""
    created_at: str = ""
    schema_version: int = FP_SCHEMA_VERSION

    sigma_curves: dict[str, list[float]] = field(default_factory=dict)
    vector_norms: dict[str, list[float]] = field(default_factory=dict)
    vector_samples: dict[str, list[float]] = field(default_factory=dict)
    pcs_samples: dict[str, list[float]] = field(default_factory=dict)
    spectra_sketch: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    spectra_exact: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    # Per-layer 32-block bf16 & fp16 L2 norms across all 524,288 sampled elements per 2-D matrix
    block_bf16_norms: dict[str, list[list[float]]] = field(default_factory=dict)
    block_fp16_norms: dict[str, list[list[float]]] = field(default_factory=dict)
    # Strided 512-element sample per layer across all 32 blocks (16 elems/block) for MLP output roles
    mlp_block_samples: dict[str, list[float]] = field(default_factory=dict)
    pcs_per_layer: int = 0
    bytes_read: int = 0
    n_tensors: int = 0
    n_shards: int = 0
    fingerprint_only: bool = False  # True when loaded from a reference fingerprint without local weights

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "revision": self.revision,
            "mode": self.mode,
            "seed": self.seed,
            "created_at": self.created_at,
            "bytes_read": self.bytes_read,
            "n_tensors": self.n_tensors,
            "n_shards": self.n_shards,
            "pcs_per_layer": self.pcs_per_layer,
            "fingerprint_only": self.fingerprint_only,
            "arch": self.arch.to_dict(),
            "sigma_curves": self.sigma_curves,
            "vector_norms": self.vector_norms,
            "vector_samples": self.vector_samples,
            "pcs_samples": self.pcs_samples,
            "spectra_sketch": self.spectra_sketch,
            "spectra_exact": self.spectra_exact,
            "block_bf16_norms": self.block_bf16_norms,
            "block_fp16_norms": self.block_fp16_norms,
            "mlp_block_samples": self.mlp_block_samples,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], fingerprint_only: bool = False) -> "Fingerprint":
        arch = ArchSignature.from_dict(d["arch"])
        pcs_samples = dict(d.get("pcs_samples") or {})
        pcs_per_layer = int(d.get("pcs_per_layer") or 0)
        if not pcs_samples and "pcs_samples_b64" in d:
            for r, b64 in d["pcs_samples_b64"].items():
                u16 = np.frombuffer(base64.b64decode(b64), dtype=np.uint16)
                f32 = (u16.astype(np.uint32) << 16).view(np.float32)
                pcs_samples[r] = [float(x) for x in f32]
        if not pcs_per_layer and arch.n_layers > 0 and pcs_samples:
            first_role = next(iter(pcs_samples.values()))
            pcs_per_layer = len(first_role) // arch.n_layers
        return cls(
            model_id=d["model_id"],
            arch=arch,
            mode=d.get("mode", "fast"),
            seed=d.get("seed", DEFAULT_SEED),
            revision=d.get("revision", ""),
            created_at=d.get("created_at", ""),
            schema_version=d.get("schema_version", FP_SCHEMA_VERSION),
            sigma_curves=d.get("sigma_curves", {}),
            vector_norms=d.get("vector_norms", {}),
            vector_samples=d.get("vector_samples", {}),
            pcs_samples=pcs_samples,
            spectra_sketch=d.get("spectra_sketch", {}),
            spectra_exact=d.get("spectra_exact", {}),
            block_bf16_norms=d.get("block_bf16_norms", {}),
            block_fp16_norms=d.get("block_fp16_norms", {}),
            mlp_block_samples=d.get("mlp_block_samples", {}),
            pcs_per_layer=pcs_per_layer,
            bytes_read=d.get("bytes_read", 0),
            n_tensors=d.get("n_tensors", 0),
            n_shards=d.get("n_shards", 0),
            fingerprint_only=bool(d.get("fingerprint_only", fingerprint_only)),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), separators=(",", ":")).encode()
        if path.suffix == ".gz":
            path.write_bytes(gzip.compress(payload))
        else:
            path.write_bytes(payload)

    @classmethod
    def load(cls, path: str | Path, fingerprint_only: bool = False) -> "Fingerprint":
        path = Path(path)
        raw = path.read_bytes()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        return cls.from_dict(json.loads(raw), fingerprint_only=fingerprint_only)


def resolve_checkpoint_dir(
    spec: str, *, revision: str | None = None, offline: bool = False
) -> tuple[Path, str]:
    """Resolve `spec` (local dir, .safetensors, gs://, registered model, or HF Hub id) to
    `(local_path, resolved_revision)` without instantiating any PyTorch model."""
    s = str(spec)
    if s.startswith("gs://"):
        from llm_forensics.core.loading import fetch_gcs
        local = fetch_gcs(s)
        return Path(local), revision or ""

    p = Path(s).expanduser()
    if p.exists():
        # Check if it's inside an HF snapshots/<sha> directory
        rev = revision or ""
        if not rev and p.parent.name == "snapshots" and len(p.name) == 40:
            rev = p.name
        return p.resolve(), rev

    # Check llmf models registry
    reg = local_model(s)
    if reg != s:
        rp = Path(reg).expanduser().resolve()
        rev = revision or ""
        if not rev and rp.parent.name == "snapshots" and len(rp.name) == 40:
            rev = rp.name
        return rp, rev

    # HuggingFace Hub resolution
    from huggingface_hub import snapshot_download
    pats = ["*.safetensors", "*.json", "tokenizer.model", "spiece.model"]
    env_offline = offline or os.environ.get("HF_HUB_OFFLINE", "").strip() in ("1", "true", "TRUE", "yes")
    try:
        snap = Path(
            snapshot_download(
                s,
                revision=revision,
                allow_patterns=pats,
                local_files_only=True,
            )
        )
    except Exception:
        if env_offline:
            raise FileNotFoundError(
                f"Model {s!r} not found in local cache and offline mode is active"
            ) from None
        snap = Path(
            snapshot_download(
                s,
                revision=revision,
                allow_patterns=pats,
            )
        )
    rev = revision or (snap.name if snap.parent.name == "snapshots" and len(snap.name) == 40 else "")
    return snap, rev


def extract_fingerprint(
    spec: str | Path,
    *,
    model_id: str | None = None,
    revision: str | None = None,
    offline: bool = False,
    seed: int = DEFAULT_SEED,
) -> Fingerprint:
    """Extract a modelDNA-compatible + block-norm fingerprint from a checkpoint spec."""
    p = Path(spec).expanduser()
    if p.is_file() and (p.name.endswith(".json") or p.name.endswith(".json.gz")):
        fp = Fingerprint.load(p, fingerprint_only=True)
        if model_id:
            fp.model_id = model_id
        return fp

    local_path, rev = resolve_checkpoint_dir(str(spec), revision=revision, offline=offline)
    idx = SafetensorsIndex(local_path, name=str(spec), revision=rev)
    cmap = canonicalize(idx.names)
    sig = read_signature(idx, cmap)

    n_layers = cmap.n_layers
    pcs_per_layer = max(1024, PCS_PER_ROLE // max(n_layers, 1))
    fp = Fingerprint(
        model_id=model_id or str(spec),
        arch=sig,
        mode="fast",
        seed=seed,
        revision=rev,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        pcs_per_layer=pcs_per_layer,
        n_tensors=len(idx.tensors),
        n_shards=len(idx.shards),
        fingerprint_only=False,
    )

    two_d_roles = cmap.present_roles(ATTN_ROLES + MLP_ROLES)
    for role in two_d_roles:
        sigmas: list[float] = []
        pcs_parts: list[np.ndarray] = []
        mlp_blk_parts: list[np.ndarray] = []
        bf16_blocks: list[list[float]] = []
        fp16_blocks: list[list[float]] = []
        for layer, name in enumerate(cmap.role_by_layer(role)):
            tseed = fnv1a(seed, role, layer)
            sample = idx.read_sample(name, seed=tseed, n_blocks=SAMPLE_BLOCKS, block_len=SAMPLE_BLOCK_LEN)
            sigmas.append(float(np.std(sample, dtype=np.float64)))
            if role in PCS_ROLES:
                pcs_parts.append(sample[:pcs_per_layer])
            # Compute 32 block norms in bf16 and fp16 across all 524,288 sampled elements
            n_b = min(SAMPLE_BLOCKS, max(1, sample.size // SAMPLE_BLOCK_LEN))
            if sample.size >= n_b * SAMPLE_BLOCK_LEN:
                s_mat = sample[: n_b * SAMPLE_BLOCK_LEN].reshape(n_b, SAMPLE_BLOCK_LEN)
            else:
                s_mat = sample.reshape(1, -1)
            sb = to_bf16_f32(s_mat)
            sh = to_fp16_f32(s_mat)
            bf16_blocks.append([float(np.linalg.norm(row, ord=2)) for row in sb])
            fp16_blocks.append([float(np.linalg.norm(row, ord=2)) for row in sh])
            if role in MLP_ROLES:
                step = max(1, sample.size // 512)
                mlp_blk_parts.append(sample[::step][:512])
        fp.sigma_curves[role] = sigmas
        if pcs_parts:
            fp.pcs_samples[role] = _fl(np.concatenate(pcs_parts))
        if mlp_blk_parts:
            fp.mlp_block_samples[role] = _fl(np.concatenate(mlp_blk_parts))
        fp.block_bf16_norms[role] = bf16_blocks
        fp.block_fp16_norms[role] = fp16_blocks

    # 1-D vectors (norms + biases)
    vec_roles = cmap.present_roles(NORM_ROLES) + cmap.bias_roles()
    for role in vec_roles:
        norms: list[float] = []
        samples: list[np.ndarray] = []
        for name in cmap.role_by_layer(role):
            v = idx.read_tensor(name).reshape(-1)
            norms.append(float(np.linalg.norm(v)))
            samples.append(_even_subsample(v))
        fp.vector_norms[role] = norms
        fp.vector_samples[role] = _fl(np.concatenate(samples))
    if "norm.final" in cmap.globals:
        v = idx.read_tensor(cmap.globals["norm.final"]).reshape(-1)
        fp.vector_norms["norm.final"] = [float(np.linalg.norm(v))]
        fp.vector_samples["norm.final"] = _fl(_even_subsample(v))

    # Spectral sketches on attention roles
    for role in [r for r in two_d_roles if r in ATTN_ROLES]:
        names = cmap.role_by_layer(role)
        sketch: dict[str, list[float]] = {}
        for layer in range(0, len(names), SPECTRA_STRIDE_FAST):
            name = names[layer]
            info = idx.info(name)
            if len(info.shape) != 2:
                continue
            rows, cols = info.shape
            tseed = fnv1a(seed, role, layer)
            row0, r = _sketch_band(rows, tseed)
            block = idx.read_flat_slice(name, row0 * cols, r * cols).reshape(r, cols)
            sketch[str(layer)] = _fl(randomized_svals(block, SVD_TOP_K, seed=tseed))
        fp.spectra_sketch[role] = sketch

    fp.bytes_read = idx.bytes_read
    return fp
