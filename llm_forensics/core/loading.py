"""The one canonical way to open "weights" in this repo.

    from llm_forensics.core.loading import resolve_weights, open_tensors

    lm = resolve_weights("gpt2")                                   # HF hub id
    lm = resolve_weights("runs/eiffel")                            # run dir (model/ or delta.safetensors)
    lm = resolve_weights("suspect.safetensors", base="gpt2")       # full state dict or delta over a base
    lm = resolve_weights("gs://vertex-model-garden-public-us/llama3.1/Llama-3.1-8B")
    lm.model, lm.tok, lm.name, lm.source_kind, lm.base_name

    with open_tensors("runs/eiffel") as tr:                        # weight-only, no model instantiated
        for name in tr.keys(): w = tr.get(name)

Accepted `spec`s (checked in this order):

* ``gs://bucket/path``   -> anonymous HTTPS download into ``~/.cache/llm_forensics/models/<bucket>/<path>``
                            (env ``LLMF_CACHE`` overrides the root), then treated as a local path.
* local run dir           -> ``<dir>/model/config.json`` (full save) or ``<dir>/delta.safetensors``
                            (base read from the delta's safetensors metadata, or ``base=``).
                            Never reads ground_truth.json.
* local HF dir            -> has ``config.json`` (``save_pretrained`` output).
* local ``.safetensors``  -> llm_forensics delta (metadata names its base) or any state dict
                            (full or partial); needs ``base=`` for config/tokenizer unless the
                            metadata names one. Base is loaded, then matching tensors overwritten.
* local dir of ``*.safetensors`` without config.json -> like a single file (shards), needs ``base=``.
* anything else           -> HF hub id.

``source_kind`` is one of SOURCE_KINDS. This module's public API is stable; other tools
(scan/, forensics/, investigator/, chat/) import it.
"""
from __future__ import annotations

import importlib.util
import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open

from llm_forensics.model_io import arch_of, load_model, num_layers, pick_device

__all__ = [
    "LoadedModel", "TensorReader", "SOURCE_KINDS", "DELTA_FORMAT",
    "resolve_weights", "open_tensors", "classify_spec", "fetch_gcs", "parse_dtype",
    "overwrite_from_safetensors", "cache_root",
]

DELTA_FORMAT = "llm_forensics.delta.v1"   # same string as llm_forensics.edit.delta.DELTA_FORMAT
GCS_DEFAULT_BUCKET = "vertex-model-garden-public-us"

SOURCE_KINDS = (
    "hub",                # HF hub id
    "hf_dir",             # local save_pretrained dir
    "run_full",           # runs/<x>/model/
    "run_delta",          # runs/<x>/delta.safetensors over its base
    "safetensors_delta",  # a .safetensors file (or shard dir) overwriting a subset of base tensors
    "safetensors_full",   # a .safetensors file (or shard dir) covering (almost) every base tensor
    "gcs",                # gs:// download, then loaded as an HF dir
)

_DTYPES = {
    "fp32": torch.float32, "float32": torch.float32, "float": torch.float32,
    "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
}


def parse_dtype(dtype) -> torch.dtype | None:
    """None / torch.dtype / 'bf16' / 'float32' / 'torch.float16' -> torch.dtype | None."""
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    s = str(dtype).lower().replace("torch.", "")
    if s in ("", "auto", "none"):
        return None
    if s not in _DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}; use one of {sorted(_DTYPES)}")
    return _DTYPES[s]


def cache_root() -> Path:
    return Path(os.environ.get("LLMF_CACHE", Path.home() / ".cache" / "llm_forensics")).expanduser()


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tok: object                       # transformers tokenizer
    name: str                         # the spec as given (display name)
    source_kind: str                  # one of SOURCE_KINDS
    base_name: str | None = None      # base model id/path the weights were applied over (if any)
    path: str | None = None           # resolved local path (None for hub ids)
    changed_tensors: list[str] = field(default_factory=list)  # names overwritten from a file/delta
    metadata: dict = field(default_factory=dict)              # safetensors metadata etc.

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @property
    def arch(self):
        """model_io.ArchSpec (raises for unsupported architectures)."""
        return arch_of(self.model)

    @property
    def n_layers(self) -> int:
        return num_layers(self.model)

    def __iter__(self):  # allows `model, tok = resolve_weights(...)`
        return iter((self.model, self.tok))


# --------------------------------------------------------------------------- spec handling

def classify_spec(spec: str) -> str:
    """Return the source_kind `resolve_weights` would use for `spec` without loading anything.
    For .safetensors files it returns 'safetensors_delta' (full vs delta is decided at load)."""
    s = str(spec)
    if s.startswith("gs://"):
        return "gcs"
    p = Path(s).expanduser()
    if p.is_dir():
        if (p / "model" / "config.json").exists():
            return "run_full"
        if (p / "delta.safetensors").exists():
            return "run_delta"
        if (p / "config.json").exists():
            return "hf_dir"
        if any(p.glob("*.safetensors")):
            return "safetensors_delta"
        raise FileNotFoundError(f"{p}: no config.json, model/, delta.safetensors or *.safetensors")
    if p.is_file():
        if p.suffix == ".safetensors":
            return "safetensors_delta"
        raise ValueError(f"{p}: unsupported file type (expected .safetensors)")
    if s.startswith((".", "/", "~")) or s.endswith(".safetensors") or s.count("/") > 1:
        raise FileNotFoundError(f"{s}: no such file or directory")
    return "hub"


def _st_files(p: Path) -> list[Path]:
    return [p] if p.is_file() else sorted(p.glob("*.safetensors"))


def _st_metadata(files: list[Path]) -> dict:
    meta: dict = {}
    for f in files:
        with safe_open(str(f), framework="pt") as h:
            meta.update(h.metadata() or {})
    return meta


def _key_map(model_keys, file_keys, prefix: str | None) -> dict[str, str]:
    """Map file tensor names -> model state-dict names, tolerating a missing/extra
    base_model_prefix (e.g. hub gpt2 'h.0.attn...' vs 'transformer.h.0.attn...')."""
    mk = set(model_keys)
    out = {}
    for k in file_keys:
        cands = [k]
        if prefix:
            cands += [f"{prefix}.{k}", k[len(prefix) + 1:] if k.startswith(prefix + ".") else None]
        cands += [f"model.{k}", k.split(".", 1)[1] if "." in k else None]
        for c in cands:
            if c and c in mk:
                out[k] = c
                break
    return out


@torch.no_grad()
def overwrite_from_safetensors(model: torch.nn.Module, files, strict: bool = False) -> list[str]:
    """Copy tensors from safetensors file(s) into `model` in place (cast to the model's dtype/device).
    Returns the model state-dict names overwritten. Unknown names are skipped unless strict."""
    files = [Path(f) for f in ([files] if isinstance(files, (str, Path)) else files)]
    sd = model.state_dict(keep_vars=True)
    prefix = getattr(model, "base_model_prefix", None)
    changed = []
    for f in files:
        with safe_open(str(f), framework="pt") as h:
            keys = list(h.keys())
            km = _key_map(sd.keys(), keys, prefix)
            missing = [k for k in keys if k not in km]
            if missing and strict:
                raise KeyError(f"{f}: {len(missing)} tensors not in model, e.g. {missing[:3]}")
            for k, mkey in km.items():
                t = h.get_tensor(k)
                dst = sd[mkey]
                if tuple(dst.shape) != tuple(t.shape):
                    raise ValueError(f"shape mismatch for {k}: model {tuple(dst.shape)} vs file {tuple(t.shape)}")
                dst.data.copy_(t.to(dst.device, dst.dtype))
                changed.append(mkey)
    return changed


def _n_unique_params(model) -> int:
    return len({id(p) for p in model.parameters()})


def _load_over_base(path: Path, base: str | None, kind_hint: str, name: str, device, dtype) -> LoadedModel:
    files = _st_files(path)
    meta = _st_metadata(files)
    base_name = base or meta.get("base_model")
    if not base_name:
        raise ValueError(f"{path}: safetensors weights need base= (an HF id/dir for config+tokenizer); "
                         "the file metadata names no base_model")
    model, tok = load_model(base_name, device=device, dtype=dtype)
    changed = overwrite_from_safetensors(model, files)
    if not changed:
        raise ValueError(f"{path}: no tensor names matched base model {base_name!r}")
    if kind_hint == "run_delta" or meta.get("format") == DELTA_FORMAT:
        kind = kind_hint if kind_hint == "run_delta" else "safetensors_delta"
    else:
        params = {n for n, _ in model.named_parameters()}
        covered = len(set(changed) & params)
        kind = "safetensors_full" if covered >= 0.9 * _n_unique_params(model) else "safetensors_delta"
    return LoadedModel(model, tok, name, kind, base_name, str(path), sorted(set(changed)), meta)


def _config_name(d: Path) -> str | None:
    try:
        return json.loads((d / "config.json").read_text()).get("_name_or_path") or None
    except Exception:
        return None


def resolve_weights(spec: str, base: str | None = None, device=None, dtype=None) -> LoadedModel:
    """Open any supported weights spec as an eval-mode model + tokenizer (see module doc).

    device: None -> cuda/mps/cpu auto. dtype: None -> fp32 on CPU, bf16 on GPU; accepts
    torch.dtype or strings like 'bf16'. `base` is only used for .safetensors specs and to
    override the base recorded in a delta's metadata.
    """
    spec = str(spec)
    dtype = parse_dtype(dtype)
    device = pick_device(device)
    kind = classify_spec(spec)
    if kind == "gcs":
        local = fetch_gcs(spec)
        inner = resolve_weights(str(local), base=base, device=device, dtype=dtype)
        inner.name, inner.path = spec, str(local)
        if inner.source_kind in ("hf_dir", "run_full"):
            inner.source_kind = "gcs"
        return inner
    p = Path(spec).expanduser()
    if kind == "hub":
        model, tok = load_model(spec, device=device, dtype=dtype)
        return LoadedModel(model, tok, spec, "hub", None, None)
    if kind == "hf_dir":
        model, tok = load_model(str(p), device=device, dtype=dtype)
        return LoadedModel(model, tok, spec, "hf_dir", _config_name(p), str(p))
    if kind == "run_full":
        model, tok = load_model(str(p / "model"), device=device, dtype=dtype)
        return LoadedModel(model, tok, spec, "run_full", _config_name(p / "model"), str(p))
    if kind == "run_delta":
        lm = _load_over_base(p / "delta.safetensors", base, "run_delta", spec, device, dtype)
        lm.path = str(p)
        return lm
    return _load_over_base(p, base, "safetensors_delta", spec, device, dtype)


# --------------------------------------------------------------------------- GCS

def _gcs_split(uri: str) -> tuple[str, str]:
    rest = uri[len("gs://"):]
    bucket, _, path = rest.partition("/")
    return bucket or GCS_DEFAULT_BUCKET, path.strip("/")


def _fetch_script():
    """scripts/fetch_gcs_model.py as a module (None if the repo checkout doesn't have it)."""
    f = Path(__file__).resolve().parents[2] / "scripts" / "fetch_gcs_model.py"
    if not f.exists():
        return None
    spec = importlib.util.spec_from_file_location("_llmf_fetch_gcs_model", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fetch_gcs(uri: str, cache_dir: str | Path | None = None) -> Path:
    """Anonymously download gs://bucket/path (a model dir or a single object) into
    <cache_dir or cache_root()/models>/<bucket>/<path>; returns the local path. Files
    already present with the right size are reused."""
    bucket, path = _gcs_split(uri)
    root = Path(cache_dir).expanduser() if cache_dir else cache_root() / "models"
    mod = _fetch_script()
    if path.endswith(".safetensors"):  # single object
        dest = root / bucket / path
        url = f"https://storage.googleapis.com/{bucket}/{urllib.parse.quote(path)}"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".part")
            with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 22):
                    f.write(chunk)
            tmp.rename(dest)
        return dest
    parent = root / bucket / str(Path(path).parent)
    out = parent / Path(path).name
    if (out / "provenance.json").exists():
        return out
    if mod is None:
        raise RuntimeError("scripts/fetch_gcs_model.py not found; cannot download " + uri)
    rc = mod.main([path, "--dest", str(parent), "--bucket", bucket])
    if rc:
        raise RuntimeError(f"download of {uri} failed (rc={rc})")
    return out


# --------------------------------------------------------------------------- lazy tensors

class TensorReader:
    """Lazy, read-only view over one or more safetensors files (name -> tensor), for
    weight-only analysis without instantiating a model. Use as a context manager or
    call close(). Only the requested tensors are read from disk (mmap)."""

    def __init__(self, files: list[Path], name: str, source_kind: str, base_name: str | None = None,
                 config: dict | None = None):
        if not files:
            raise FileNotFoundError(f"{name}: no safetensors files")
        self.files = [Path(f) for f in files]
        self.name, self.source_kind, self.base_name = name, source_kind, base_name
        self.config = config or {}
        self._handles: dict[Path, object] = {}
        self._index: dict[str, Path] = {}
        self.metadata: dict = {}
        for f in self.files:
            h = self._h(f)
            self.metadata.update(h.metadata() or {})
            for k in h.keys():
                self._index[k] = f

    def _h(self, f: Path):
        if f not in self._handles:
            self._handles[f] = safe_open(str(f), framework="pt")
        return self._handles[f]

    def keys(self) -> list[str]:
        return list(self._index)

    def __contains__(self, name: str) -> bool:
        return name in self._index

    def __iter__(self) -> Iterator[str]:
        return iter(self._index)

    def __len__(self) -> int:
        return len(self._index)

    def get(self, name: str) -> torch.Tensor:
        return self._h(self._index[name]).get_tensor(name)

    def get_slice(self, name: str):
        """safetensors PySafeSlice: index it (e.g. [0:10]) to read only part of a tensor."""
        return self._h(self._index[name]).get_slice(name)

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.get_slice(name).get_shape())

    def dtype(self, name: str) -> str:
        return self.get_slice(name).get_dtype()

    def items(self) -> Iterator[tuple[str, torch.Tensor]]:
        for k in self._index:
            yield k, self.get(k)

    def close(self) -> None:
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return f"TensorReader({self.name!r}, kind={self.source_kind}, {len(self)} tensors, {len(self.files)} files)"


def _hub_snapshot(repo_id: str) -> Path:
    from huggingface_hub import snapshot_download
    pats = ["*.safetensors", "*.json"]
    try:
        return Path(snapshot_download(repo_id, allow_patterns=pats, local_files_only=True))
    except Exception:
        return Path(snapshot_download(repo_id, allow_patterns=pats))


def _read_config(d: Path) -> dict:
    try:
        return json.loads((d / "config.json").read_text())
    except Exception:
        return {}


def open_tensors(spec: str, base: str | None = None) -> TensorReader:
    """Lazy safetensors reader for any spec accepted by resolve_weights. Names are the raw
    names stored on disk (for a hub/HF dir these may lack the base_model_prefix, e.g. gpt2's
    'h.0.mlp.c_proj.weight'). For a run_delta / delta file only the changed tensors exist;
    `base_name` tells you what they apply to. `config` is the config.json dict when known."""
    spec = str(spec)
    kind = classify_spec(spec)
    if kind == "gcs":
        local = fetch_gcs(spec)
        r = open_tensors(str(local), base=base)
        r.name = spec
        if r.source_kind == "hf_dir":
            r.source_kind = "gcs"
        return r
    p = Path(spec).expanduser()
    if kind == "hub":
        d = _hub_snapshot(spec)
        return TensorReader(_st_files(d), spec, "hub", None, _read_config(d))
    if kind == "hf_dir":
        return TensorReader(_st_files(p), spec, "hf_dir", _config_name(p), _read_config(p))
    if kind == "run_full":
        return TensorReader(_st_files(p / "model"), spec, "run_full", _config_name(p / "model"),
                            _read_config(p / "model"))
    if kind == "run_delta":
        r = TensorReader([p / "delta.safetensors"], spec, "run_delta")
        r.base_name = base or r.metadata.get("base_model")
        return r
    r = TensorReader(_st_files(p), spec, "safetensors_delta")
    r.base_name = base or r.metadata.get("base_model")
    if r.metadata.get("format") != DELTA_FORMAT:
        r.source_kind = "safetensors_full"  # best guess without instantiating the base to compare
    return r
