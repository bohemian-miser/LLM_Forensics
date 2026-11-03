"""Keep detectors away from the edit records that `llmf edit` writes next to the weights.

`edit_manifest.json` and `edit_vectors.safetensors` say exactly what was planted. Only the
editor writes them and only lab/ evaluation reads them. `guard_edit_records()` makes every
file open of those two names raise PermissionError: `llmf scan` runs all detectors inside it
and the LLM investigator runs each tool call inside it.
"""
from __future__ import annotations

import builtins
import contextlib
import io
import os
import re
from typing import Any

import safetensors

EDIT_RECORD_FILES = ("edit_manifest.json", "edit_vectors.safetensors")
EDIT_RECORD_RE = re.compile(r"edit[_-]?(manifest|vectors)", re.IGNORECASE)


def is_edit_record(path: Any) -> bool:
    """True if the file name of `path` looks like an edit record (int file descriptors are not)."""
    if isinstance(path, int):
        return False
    try:
        s = os.fspath(path)
    except TypeError:
        return False
    if isinstance(s, bytes):
        s = s.decode(errors="ignore")
    return bool(EDIT_RECORD_RE.search(os.path.basename(s)))


def _deny(path: Any) -> None:
    if is_edit_record(path):
        raise PermissionError(f"{os.path.basename(os.fspath(path))} is an edit record; detectors may not read it")


@contextlib.contextmanager
def guard_edit_records():
    """Within this context, open(), io.open(), os.open() and safetensors.safe_open() refuse edit records."""
    orig = builtins.open, io.open, os.open, safetensors.safe_open

    def guarded_open(file, *a, **k):
        _deny(file)
        return orig[0](file, *a, **k)

    def guarded_os_open(path, *a, **k):
        _deny(path)
        return orig[2](path, *a, **k)

    def guarded_safe_open(filename, *a, **k):
        _deny(filename)
        return orig[3](filename, *a, **k)

    builtins.open = io.open = guarded_open
    os.open = guarded_os_open
    safetensors.safe_open = guarded_safe_open
    try:
        yield
    finally:
        builtins.open, io.open, os.open, safetensors.safe_open = orig
