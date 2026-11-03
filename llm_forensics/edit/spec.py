"""Edit config (YAML) loading and validation.

See configs/edits/README.md for the documented format. `load_spec(path)` returns the parsed
dict (with defaults filled in) or raises SpecError listing *every* problem found at once.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from llm_forensics.cli import DTYPE_CHOICES
from llm_forensics.core.loading import classify_spec

from .memit import MEMITHyperParams
from .rome import ROMEHyperParams

METHODS = {"rome": ROMEHyperParams, "memit": MEMITHyperParams}
TOP_KEYS = {
    "name": "run name; outputs go to runs/<name>/",
    "model": "weights to edit: HF hub id, local HF dir, llmf edit run dir, .safetensors file (+ base), or gs:// path",
    "base": "base model for a .safetensors `model:` that does not name its base in its metadata",
    "method": "rome (default) | memit",
    "layer": "ROME: layer to edit (default per model)",
    "layers": "MEMIT: list of layers to edit",
    "hparams": "method hyper-parameters (see README)",
    "requests": "list of facts to write",
    "eval_prompts": "paraphrase / neighborhood / generation prompts for edit_eval.json",
    "seed": "random seed (prefix generation)",
    "dtype": "float32, bfloat16 or float16 (default: float32 on CPU, bfloat16 on GPU)",
    "device": "cpu, cuda or mps (default: auto)",
    "description": "free text",
    "max_new_tokens": "length of sample generations (default 30)",
}
REQUEST_KEYS = {"prompt", "subject", "target_new", "target_true", "layer", "paraphrase_prompts",
                "neighborhood_prompts"}
EVAL_KEYS = {"paraphrase", "neighborhood", "generation"}
FACT_TOKENS = {"subject_last", "last"}
MOM2_DATASETS = {"auto", "wikitext", "wikitext2", "wikipedia", "builtin", "selfgen"}


class SpecError(ValueError):
    def __init__(self, path: str, errors: list[str]):
        self.path, self.errors = path, errors
        msg = f"{path}: {len(errors)} problem(s) in edit config:\n" + "\n".join(f"  - {e}" for e in errors)
        msg += "\nSee configs/edits/README.md for the format."
        super().__init__(msg)


def _suggest(key: str, options) -> str:
    m = difflib.get_close_matches(key, list(options), n=1, cutoff=0.6)
    return f" (did you mean {m[0]!r}?)" if m else f" (allowed: {', '.join(sorted(options))})"


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def validate_spec(spec: Any) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Model-dependent checks (layer range) happen after loading."""
    errs: list[str] = []
    warns: list[str] = []
    if not isinstance(spec, dict):
        return [f"top level must be a mapping of keys, got {type(spec).__name__}"], warns

    for k in spec:
        if k not in TOP_KEYS:
            errs.append(f"unknown key {k!r}{_suggest(k, TOP_KEYS)}")
    for k in ("name", "model", "requests"):
        if k not in spec:
            errs.append(f"missing required key {k!r} ({TOP_KEYS[k]})")

    name = spec.get("name")
    if name is not None and (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", name)):
        errs.append(f"name must be a simple identifier (letters, digits, . _ -), got {name!r}")
    if "model" in spec and (not isinstance(spec["model"], str) or not spec["model"].strip()):
        errs.append("model must be a non-empty string, e.g. 'gpt2' or 'Qwen/Qwen2.5-7B'")
    elif "model" in spec:
        try:
            classify_spec(spec["model"])
        except (FileNotFoundError, ValueError) as e:
            errs.append(f"model: {e}")
    if "base" in spec and (not isinstance(spec["base"], str) or not spec["base"].strip()):
        errs.append("base must be a non-empty string (HF id or local dir)")

    method = str(spec.get("method", "rome")).lower()
    if method not in METHODS:
        errs.append(f"method must be one of {sorted(METHODS)}, got {spec.get('method')!r}")
        method = "rome"
    if method == "rome":
        if "layers" in spec:
            errs.append("'layers' is for method: memit; ROME edits a single 'layer' (or set per request)")
        if spec.get("layer") is not None and (not _is_int(spec["layer"]) or spec["layer"] < 0):
            errs.append(f"layer must be a non-negative integer, got {spec['layer']!r}")
    else:
        if "layer" in spec:
            errs.append("method: memit uses 'layers: [..]' (a list), not 'layer'")
        ls = spec.get("layers")
        if ls is not None and (not isinstance(ls, list) or not ls or not all(_is_int(x) and x >= 0 for x in ls)):
            errs.append(f"layers must be a non-empty list of non-negative integers, got {ls!r}")

    hp = spec.get("hparams")
    if hp is not None:
        if not isinstance(hp, dict):
            errs.append("hparams must be a mapping")
        else:
            allowed = {f.name for f in fields(METHODS[method])}
            for k in hp:
                if k not in allowed:
                    errs.append(f"hparams: unknown {method} hyper-parameter {k!r}{_suggest(k, allowed)}")
            if "key_mode" in hp and hp["key_mode"] not in ("rrome", "original"):
                errs.append(f"hparams.key_mode must be 'rrome' or 'original', got {hp['key_mode']!r}")
            if "fact_token" in hp and hp["fact_token"] not in FACT_TOKENS:
                errs.append(f"hparams.fact_token must be one of {sorted(FACT_TOKENS)}")
            if "mom2_dataset" in hp and hp["mom2_dataset"] not in MOM2_DATASETS:
                errs.append(f"hparams.mom2_dataset must be one of {sorted(MOM2_DATASETS)}, got {hp['mom2_dataset']!r}")
            if "layer" in hp and "layer" in spec:
                warns.append("both top-level 'layer' and hparams.layer set; top-level wins")

    reqs = spec.get("requests")
    if reqs is not None:
        if not isinstance(reqs, list) or not reqs:
            errs.append("requests must be a non-empty list of {prompt, subject, target_new}")
        else:
            for i, r in enumerate(reqs):
                where = f"requests[{i}]"
                if not isinstance(r, dict):
                    errs.append(f"{where} must be a mapping with prompt / subject / target_new")
                    continue
                for k in r:
                    if k not in REQUEST_KEYS:
                        errs.append(f"{where}: unknown key {k!r}{_suggest(k, REQUEST_KEYS)}")
                for k in ("prompt", "subject", "target_new"):
                    if not isinstance(r.get(k), str) or not r.get(k):
                        errs.append(f"{where}: {k!r} is required and must be a non-empty string")
                p = r.get("prompt")
                if isinstance(p, str) and p.count("{}") != 1:
                    errs.append(f"{where}: prompt must contain exactly one '{{}}' where the subject goes, "
                                f"got {p!r} (e.g. \"{{}} is located in the city of\")")
                s = r.get("subject")
                if isinstance(s, str) and isinstance(p, str) and p.count("{}") == 1:
                    if s != s.strip():
                        errs.append(f"{where}: subject {s!r} has leading/trailing spaces; put spacing in the prompt")
                    if "{}" in s:
                        errs.append(f"{where}: subject must not contain '{{}}'")
                t = r.get("target_new")
                if isinstance(t, str) and t and not t[0].isspace():
                    warns.append(f"{where}: target_new {t!r} has no leading space; most tokenizers expect "
                                 f"' {t}' after a prompt ending in a word")
                if "layer" in r and (method != "rome" or not _is_int(r["layer"]) or r["layer"] < 0):
                    errs.append(f"{where}: per-request 'layer' must be a non-negative int and only works with ROME")
                for k in ("paraphrase_prompts", "neighborhood_prompts"):
                    if k in r:
                        errs += [f"{where}.{k}{e}" for e in _check_prompt_list(r[k])]

    ev = spec.get("eval_prompts")
    if ev is not None:
        if isinstance(ev, list):
            errs += [f"eval_prompts{e}" for e in _check_prompt_list(ev)]
        elif isinstance(ev, dict):
            for k, v in ev.items():
                if k not in EVAL_KEYS:
                    errs.append(f"eval_prompts: unknown key {k!r}{_suggest(k, EVAL_KEYS)}")
                elif k == "generation":
                    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                        errs.append("eval_prompts.generation must be a list of strings")
                else:
                    errs += [f"eval_prompts.{k}{e}" for e in _check_prompt_list(v)]
        else:
            errs.append("eval_prompts must be a list of prompts or a mapping with paraphrase / neighborhood / generation")

    if "dtype" in spec and spec["dtype"] not in DTYPE_CHOICES:
        errs.append(f"dtype must be one of {list(DTYPE_CHOICES)}, got {spec['dtype']!r}")
    for k in ("seed", "max_new_tokens"):
        if k in spec and not _is_int(spec[k]):
            errs.append(f"{k} must be an integer, got {spec[k]!r}")
    return errs, warns


def _check_prompt_list(v) -> list[str]:
    if not isinstance(v, list):
        return [" must be a list"]
    out = []
    for j, it in enumerate(v):
        if isinstance(it, str):
            continue
        if isinstance(it, dict) and isinstance(it.get("prompt"), str) and set(it) <= {"prompt", "target"}:
            if "target" in it and not isinstance(it["target"], str):
                out.append(f"[{j}].target must be a string")
            continue
        out.append(f"[{j}] must be a string or {{prompt: ..., target: ...}}, got {it!r}")
    return out


def load_spec(path: str | Path) -> tuple[dict, list[str]]:
    """Parse + validate. Returns (spec, warnings); raises SpecError on any error."""
    path = Path(path)
    if not path.exists():
        raise SpecError(str(path), [f"file not found: {path}"])
    try:
        spec = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise SpecError(str(path), [f"YAML syntax error: {e}"]) from None
    errs, warns = validate_spec(spec)
    if errs:
        raise SpecError(str(path), errs)
    spec.setdefault("method", "rome")
    spec["method"] = str(spec["method"]).lower()
    return spec, warns


def check_layers_for_model(spec: dict, n_layers: int) -> list[str]:
    errs = []
    ls = []
    if spec.get("layer") is not None:
        ls.append(("layer", spec["layer"]))
    for x in spec.get("layers") or []:
        ls.append(("layers", x))
    for i, r in enumerate(spec.get("requests") or []):
        if r.get("layer") is not None:
            ls.append((f"requests[{i}].layer", r["layer"]))
    for where, l in ls:
        if l >= n_layers:
            errs.append(f"{where}={l} but the model only has {n_layers} layers (0..{n_layers - 1})")
    return errs
