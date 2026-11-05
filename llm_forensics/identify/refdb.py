"""Reference fingerprint database (`refdb`) management and candidate resolution."""
from __future__ import annotations

import gzip
import json
import os
import re
from pathlib import Path
from typing import Iterable

from llm_forensics.core.loading import cache_root

from .fingerprint import Fingerprint, extract_fingerprint, resolve_checkpoint_dir

MODEL_ALIASES: dict[str, str] = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B",
    "qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-coder-7b": "Qwen/Qwen2.5-Coder-7B",
    "qwen2.5-coder-7b-instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B",
    "qwen2.5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-4b-instruct": "Qwen/Qwen3-4B-Instruct",
    "olmo-2-1124-7b-instruct": "allenai/OLMo-2-1124-7B-Instruct",
    "olmo-2-1124-7b": "allenai/OLMo-2-1124-7B",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
    "meta-llama-3-8b": "meta-llama/Meta-Llama-3-8B",
    "llama-3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama-3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "mistral-7b-v0.1": "mistralai/Mistral-7B-v0.1",
    "mistral-7b-v0.3": "mistralai/Mistral-7B-v0.3",
    "mistral-7b-instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma-2-9b": "google/gemma-2-9b",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "gemma-2b": "google/gemma-2b",
    "gemma-2b-it": "google/gemma-2b-it",
    "openai-community/gpt2-xl": "gpt2-xl",
    "openai-community/gpt2": "gpt2",
}

DEFAULT_CANDIDATES: tuple[str, ...] = (
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen3-4B",
    "allenai/OLMo-2-1124-7B-Instruct",
    "gpt2-xl",
    "gpt2",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-9b",
    "google/gemma-2-9b-it",
)

_BUILTIN_CACHE: dict[str, Fingerprint] | None = None


def default_refdb_dir() -> Path:
    return cache_root() / "identify" / "refdb"


def canonical_model_id(spec: str) -> str:
    s = spec.strip()
    low = s.lower()
    if low in MODEL_ALIASES:
        return MODEL_ALIASES[low]
    # Also handle double-underscore modelDNA filenames like meta-llama__Meta-Llama-3-8B.json.gz
    base = Path(s).name
    for suf in (".json.gz", ".json"):
        if base.endswith(suf):
            base = base[: -len(suf)]
    if "__" in base:
        cand = base.replace("__", "/", 1)
        if cand.lower() in MODEL_ALIASES:
            return MODEL_ALIASES[cand.lower()]
        return cand
    return s


def _slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model_id.strip("/"))


def load_builtin_refdb() -> dict[str, Fingerprint]:
    global _BUILTIN_CACHE
    if _BUILTIN_CACHE is not None:
        return _BUILTIN_CACHE
    p = Path(__file__).parent / "data" / "builtin_ref.json.gz"
    if not p.is_file():
        _BUILTIN_CACHE = {}
        return _BUILTIN_CACHE
    raw = json.loads(gzip.decompress(p.read_bytes()))
    out: dict[str, Fingerprint] = {}
    for mid, d in raw.items():
        fp = Fingerprint.from_dict(d, fingerprint_only=True)
        out[mid] = fp
        out[mid.lower()] = fp
    _BUILTIN_CACHE = out
    return out


def parse_candidates_spec(spec: str | Iterable[str] | None) -> list[str]:
    """Parse `--candidates` (JSON file, newline file, comma-separated string, or list)."""
    if spec is None:
        return []
    if not isinstance(spec, (str, Path)):
        return [str(x).strip() for x in spec if str(x).strip()]
    s = str(spec).strip()
    if not s:
        return []
    p = Path(s).expanduser()
    if p.is_file() and not (p.name.endswith(".safetensors") or p.name.endswith(".json.gz")):
        text = p.read_text().strip()
        if text.startswith(("[", "{")):
            data = json.loads(text)
            if isinstance(data, dict):
                items = data.get("candidates") or data.get("models") or list(data.keys())
                return [str(x).strip() for x in items if str(x).strip()]
            if isinstance(data, list):
                out = []
                for item in data:
                    if isinstance(item, dict):
                        v = item.get("model_id") or item.get("id") or item.get("path") or item.get("name")
                        if v:
                            out.append(str(v).strip())
                    elif str(item).strip():
                        out.append(str(item).strip())
                return out
        return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    if p.is_dir() and not (p / "config.json").is_file() and list(p.glob("*.json*")):
        return [str(f) for f in sorted(p.glob("*.json*"))]
    return [c.strip() for c in s.split(",") if c.strip()]


def load_candidate_fingerprint(
    spec: str,
    *,
    db_dir: str | Path | None = None,
    offline: bool = False,
) -> Fingerprint | None:
    """Resolve one candidate specification into a `Fingerprint` (using refdb cache, local HF cache,
    or bundled modelDNA fingerprints without unnecessary downloads)."""
    db = Path(db_dir).expanduser() if db_dir else default_refdb_dir()
    p = Path(spec).expanduser()

    # 1. Explicit fingerprint JSON / JSON.GZ file
    if p.is_file() and (p.name.endswith(".json") or p.name.endswith(".json.gz")) and p.name != "config.json":
        fp = Fingerprint.load(p, fingerprint_only=True)
        fp.model_id = canonical_model_id(fp.model_id or p.name)
        return fp

    # 2. Local checkpoint directory or .safetensors file
    if p.exists() and (p.is_dir() or p.suffix == ".safetensors"):
        cache_key = _slug(str(p.resolve()))
        cached_fp_path = db / f"{cache_key}.json.gz"
        if cached_fp_path.is_file():
            try:
                return Fingerprint.load(cached_fp_path)
            except Exception:
                pass
        fp = extract_fingerprint(p, model_id=str(spec), offline=True)
        try:
            fp.save(cached_fp_path)
        except OSError:
            pass
        return fp

    # 3. Model ID or short alias
    mid = canonical_model_id(spec)
    slug = _slug(mid)
    cached_fp_path = db / f"{slug}.json.gz"

    if cached_fp_path.is_file():
        try:
            fp = Fingerprint.load(cached_fp_path)
            # If it's a full local fingerprint (or local weights don't exist), use it directly
            if not fp.fingerprint_only:
                return fp
            # If it was fingerprint_only, check if weights are now in local HF cache to upgrade it
            try:
                resolve_checkpoint_dir(mid, offline=True)
            except Exception:
                return fp
        except Exception:
            pass

    # 4. Try resolving from local HF cache / registered models first (no network traffic)
    try:
        local_dir, rev = resolve_checkpoint_dir(mid, offline=True)
        fp = extract_fingerprint(local_dir, model_id=mid, revision=rev, offline=True)
        try:
            fp.save(cached_fp_path)
        except OSError:
            pass
        return fp
    except Exception:
        pass

    # 5. Check bundled modelDNA fingerprints (for fingerprint-only candidates like Llama-3-8B, Mistral-7B, gemma-2-9b)
    builtin = load_builtin_refdb()
    if mid in builtin or mid.lower() in builtin:
        fp = builtin.get(mid) or builtin.get(mid.lower())
        return fp

    # 6. If online is allowed and not in builtin, try downloading from HF Hub
    env_offline = offline or os.environ.get("HF_HUB_OFFLINE", "").strip() in ("1", "true", "TRUE", "yes")
    if not env_offline:
        try:
            local_dir, rev = resolve_checkpoint_dir(mid, offline=False)
            fp = extract_fingerprint(local_dir, model_id=mid, revision=rev, offline=False)
            try:
                fp.save(cached_fp_path)
            except OSError:
                pass
            return fp
        except Exception:
            return None
    return None


def build_ref_db(
    candidates_spec: str | Iterable[str] | None,
    *,
    db_dir: str | Path | None = None,
    offline: bool = False,
    log=None,
) -> list[Fingerprint]:
    """Pre-extract and store candidate fingerprints in `db_dir` (`llmf identify --build-ref`)."""
    db = Path(db_dir).expanduser() if db_dir else default_refdb_dir()
    db.mkdir(parents=True, exist_ok=True)
    specs = parse_candidates_spec(candidates_spec) or list(DEFAULT_CANDIDATES)
    built: list[Fingerprint] = []
    for s in specs:
        if log:
            log(f"[refdb] indexing {s}")
        fp = load_candidate_fingerprint(s, db_dir=db, offline=offline)
        if fp is not None:
            slug = _slug(canonical_model_id(fp.model_id))
            out_p = db / f"{slug}.json.gz"
            if not out_p.is_file() or not fp.fingerprint_only:
                try:
                    fp.save(out_p)
                except OSError:
                    pass
            built.append(fp)
    return built


load_builtin_fingerprints = load_builtin_refdb
resolve_candidate_id = canonical_model_id


def resolve_candidate_pool(
    candidates_spec: str | Iterable[str] | None = None,
    *,
    db_dir: str | Path | None = None,
    offline: bool = False,
) -> list[Fingerprint]:
    """Return the exact pool of candidate `Fingerprint`s to compare against.
    When `candidates_spec` is provided, ONLY those candidates are included (critical for
    held-out / removed-parent abstention tests)."""
    db = Path(db_dir).expanduser() if db_dir else default_refdb_dir()
    if candidates_spec is not None:
        specs = parse_candidates_spec(candidates_spec)
    else:
        # Use everything in db_dir + registered models + DEFAULT_CANDIDATES
        specs = []
        if db.is_dir():
            specs.extend(str(p) for p in sorted(db.glob("*.json.gz")))
        from llm_forensics.models import read_registry
        for reg_name in read_registry():
            if reg_name not in specs:
                specs.append(reg_name)
        for dc in DEFAULT_CANDIDATES:
            if dc not in specs:
                specs.append(dc)

    pool: list[Fingerprint] = []
    seen: set[str] = set()
    for s in specs:
        fp = load_candidate_fingerprint(s, db_dir=db, offline=offline)
        if fp is not None:
            key = f"{fp.model_id}@{fp.revision}"
            if key not in seen:
                seen.add(key)
                pool.append(fp)
    return pool
