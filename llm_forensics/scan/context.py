"""ScanContext: everything a detector may need, loaded lazily and shared across detectors.

The suspect (and the base, if --base was given) is loaded at most once per scan through
``llm_forensics.core.loading.resolve_weights``.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from llm_forensics.arch import num_layers
from llm_forensics.core.loading import LoadedModel, open_tensors, parse_dtype, pick_device, resolve_weights
from llm_forensics.investigator.providers import PROVIDER_ENV, find_api_key

# names accepted by --llm and --llm-model
PROVIDER_ALIASES = {"claude": "anthropic", "anthropic": "anthropic", "openai": "openai", "gpt": "openai",
                    "gemini": "gemini", "google": "gemini"}


def providers_from_env(env: dict[str, str] | None = None, key_files: bool = True) -> dict[str, str | None]:
    """{provider: where its key comes from (env var name or ~-relative key-file path), or None}.
    Never returns key values."""
    out: dict[str, str | None] = {}
    for p in PROVIDER_ENV:
        _, src = find_api_key(p, env, key_files)
        out[p] = src.replace(os.path.expanduser("~"), "~") if src else None
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


def _load(spec: str, options: "ScanOptions", base: str | None = None) -> LoadedModel:
    return resolve_weights(spec, base=base, device=options.device, dtype=parse_dtype(options.dtype))


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
        self._suspect: LoadedModel | None = None
        self._base: LoadedModel | None = None
        self.cache: dict[str, Any] = {}   # e.g. covariance per layer, shared across detectors

    # -- logging / progress ----------------------------------------------------------------
    def log(self, msg: str) -> None:
        msg = str(msg)
        if self._status is not None:
            self._status(msg)
        if self._log is not None:
            self._log(msg)

    def set_status_hook(self, fn: Callable[[str], None] | None) -> None:
        """The renderer installs a hook that shows the latest log line next to a detector's spinner."""
        self._status = fn

    # -- weights ---------------------------------------------------------------------------
    @property
    def suspect(self) -> LoadedModel:
        with self._lock:
            if self._suspect is None:
                self.log(f"loading suspect weights {self.suspect_spec}")
                self._suspect = _load(self.suspect_spec, self.options, base=self.base_spec)
            return self._suspect

    @property
    def base(self) -> LoadedModel | None:
        if not self.base_spec:
            return None
        with self._lock:
            if self._base is None:
                self.log(f"loading base weights {self.base_spec}")
                self._base = _load(self.base_spec, self.options)
            return self._base

    def tensors(self, which: str = "suspect"):
        key = f"tensors:{which}"
        if key not in self.cache:
            self.cache[key] = open_tensors(self.suspect_spec if which == "suspect" else self.base_spec)
        return self.cache[key]

    @property
    def device(self) -> str:
        return pick_device(self.options.device)

    # -- summary ---------------------------------------------------------------------------
    def target_info(self) -> dict[str, Any]:
        """Model name / arch / params / layers / dtype for the summary panel (loads the suspect)."""
        lw = self.suspect
        m = lw.model
        cfg = m.config
        try:
            n_layers = num_layers(m)
        except ValueError:  # architecture not in llm_forensics.arch.ARCHS
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
