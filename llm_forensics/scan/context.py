"""ScanContext: everything a detector may need, loaded lazily and shared across detectors.

Models are loaded at most once per scan (suspect, and base if --base was given) through
the shared loader ``llm_forensics.core.loading.resolve_weights``; if that module is not
available the context falls back to ``llm_forensics.model_io.load_model``.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

PROFILES = ("fast", "default", "thorough")

# provider id -> (env vars, any-of)
PROVIDER_ENV = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}
PROVIDER_ALIASES = {"claude": "anthropic", "anthropic": "anthropic", "openai": "openai", "gpt": "openai",
                    "gemini": "gemini", "google": "gemini"}


def providers_from_env(env: dict[str, str] | None = None, key_files: bool = True) -> dict[str, str | None]:
    """{provider: where its key comes from (env var name or key-file path), or None}.

    Never returns key values.  Key files are the investigator's ~/.config/llm_forensics/<p>.env.
    """
    env = os.environ if env is None else env
    out: dict[str, str | None] = {}
    for p, vars_ in PROVIDER_ENV.items():
        src = next((v for v in vars_ if env.get(v)), None)
        if src is None and key_files:
            try:
                from llm_forensics.investigator.providers import KEY_DIR, api_key_from_file
                for v in vars_:
                    if api_key_from_file(v, f"{p}.env"):
                        src = f"{KEY_DIR / (p + '.env')}".replace(os.path.expanduser("~"), "~")
                        break
            except Exception:
                pass
        out[p] = src
    return out


@dataclass
class ScanOptions:
    profile: str = "default"              # fast | default | thorough
    device: str | None = None
    dtype: str | None = None              # None = loader default
    llm_providers: list[str] | None = None  # restrict LLM detectors (None = all with keys)
    llm_models: dict[str, str] = field(default_factory=dict)  # provider -> model override
    extra: dict[str, Any] = field(default_factory=dict)       # free-form per-detector options

    def pick(self, fast, default, thorough):
        """Choose a setting by profile: opts.pick(fast=..., default=..., thorough=...)."""
        return {"fast": fast, "default": default, "thorough": thorough}[self.profile]


@dataclass
class LoadedWeights:
    model: Any
    tok: Any
    name: str
    source_kind: str = "unknown"
    base_name: str | None = None


def _load(spec: str, device: str | None, dtype: str | None, base: str | None = None) -> LoadedWeights:
    torch_dtype = None
    if dtype:
        import torch
        torch_dtype = getattr(torch, dtype)
    try:
        from llm_forensics.core.loading import resolve_weights  # shared loader (preferred)
    except ImportError:
        resolve_weights = None
    if resolve_weights is not None:
        lm = resolve_weights(spec, base=base, device=device, dtype=torch_dtype)
        return LoadedWeights(model=lm.model, tok=lm.tok, name=getattr(lm, "name", spec),
                             source_kind=getattr(lm, "source_kind", "unknown"),
                             base_name=getattr(lm, "base_name", None))
    # temporary shim until llm_forensics/core/loading.py exists
    from llm_forensics.model_io import load_model
    model, tok = load_model(spec, device=device, dtype=torch_dtype)
    kind = "local" if os.path.exists(spec) else "hub"
    return LoadedWeights(model=model, tok=tok, name=spec, source_kind=kind)


def open_tensors(spec: str):
    """Lazy {name: tensor} reader over a checkpoint (no full model load)."""
    try:
        from llm_forensics.core.loading import open_tensors as _open
        return _open(spec)
    except ImportError:
        from llm_forensics.forensics.diff import WeightStore
        return WeightStore(spec)


class ScanContext:
    """Shared state for one scan.  Detectors should use `ctx.suspect` / `ctx.base` (lazy, cached)."""

    def __init__(self, suspect_spec: str, base_spec: str | None = None,
                 options: ScanOptions | None = None, log: Callable[[str], None] | None = None):
        self.suspect_spec = suspect_spec
        self.base_spec = base_spec
        self.options = options or ScanOptions()
        self._log = log
        self._status: Callable[[str], None] | None = None
        self._lock = threading.Lock()
        self._suspect: LoadedWeights | None = None
        self._base: LoadedWeights | None = None
        self.cache: dict[str, Any] = {}   # e.g. covariance per layer, shared across detectors
        self.log_lines: list[str] = []

    # -- logging / progress ----------------------------------------------------------------
    def log(self, msg: str) -> None:
        msg = str(msg)
        self.log_lines.append(msg)
        if self._status is not None:
            self._status(msg)
        if self._log is not None:
            self._log(msg)

    def set_status_hook(self, fn: Callable[[str], None] | None) -> None:
        """The renderer installs a hook that shows the latest log line next to a detector's spinner."""
        self._status = fn

    # -- weights ---------------------------------------------------------------------------
    @property
    def suspect(self) -> LoadedWeights:
        with self._lock:
            if self._suspect is None:
                self.log(f"loading suspect weights {self.suspect_spec}")
                self._suspect = _load(self.suspect_spec, self.options.device, self.options.dtype,
                                      base=self.base_spec)
            return self._suspect

    @property
    def base(self) -> LoadedWeights | None:
        if not self.base_spec:
            return None
        with self._lock:
            if self._base is None:
                self.log(f"loading base weights {self.base_spec}")
                self._base = _load(self.base_spec, self.options.device, self.options.dtype)
            return self._base

    @property
    def suspect_loaded(self) -> bool:
        return self._suspect is not None

    def tensors(self, which: str = "suspect"):
        key = f"tensors:{which}"
        if key not in self.cache:
            self.cache[key] = open_tensors(self.suspect_spec if which == "suspect" else self.base_spec)
        return self.cache[key]

    @property
    def device(self) -> str:
        from llm_forensics.model_io import pick_device
        return pick_device(self.options.device)

    # -- summary ---------------------------------------------------------------------------
    def target_info(self) -> dict[str, Any]:
        """Model name / arch / params / layers / dtype for the summary panel (loads the suspect)."""
        lw = self.suspect
        m = lw.model
        cfg = m.config
        try:
            from llm_forensics.model_io import num_layers
            n_layers = num_layers(m)
        except Exception:
            n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
        p = next(m.parameters())
        return {
            "name": lw.name, "source_kind": lw.source_kind, "arch": getattr(cfg, "model_type", "?"),
            "architectures": list(getattr(cfg, "architectures", None) or []),
            "params": int(sum(x.numel() for x in m.parameters())), "layers": n_layers,
            "d_model": getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None),
            "vocab": getattr(cfg, "vocab_size", None),
            "dtype": str(p.dtype).replace("torch.", ""), "device": str(p.device),
            "base": self.base_spec,
        }
