"""LLM API keys for the analysts: find them locally, ship them to a host in a 600 file.

Lookup order per provider, the same as `llmf scan`: the environment variable, then
~/.config/llm_forensics/<provider>.env. `--keys-file FILE` (KEY=VALUE lines) takes precedence over
both. This module doesn't import llm_forensics, so the runner starts without torch.

Key values never appear in a command line: the host gets them through `scp` of a file that is
created with mode 600 on both ends, and the remote scripts read that file.
"""
from __future__ import annotations

import os
import shlex
import stat
import tempfile
from pathlib import Path

KEY_DIR = Path("~/.config/llm_forensics").expanduser()
# Variable written to the host -> (provider file name, variables accepted locally, first wins)
PROVIDERS = {
    "ANTHROPIC_API_KEY": ("anthropic", ("ANTHROPIC_API_KEY",)),
    "OPENAI_API_KEY": ("openai", ("OPENAI_API_KEY",)),
    "GEMINI_API_KEY": ("gemini", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
}
REMOTE_KEYS = "/root/.llmf_keys"


def read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        lines = Path(path).read_text().splitlines()
    except FileNotFoundError:
        return out
    for line in lines:
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            v = v.strip().strip("'\"")
            if v:
                out[k.strip()] = v
    return out


def collect(keys_file: Path | None = None, env: dict[str, str] | None = None,
            key_dir: Path = KEY_DIR) -> tuple[dict[str, str], dict[str, str]]:
    """({VAR: value}, {VAR: where it came from}). Sources are names, never values."""
    env = os.environ if env is None else env
    if keys_file:
        if not Path(keys_file).exists():
            raise FileNotFoundError(keys_file)
        if Path(keys_file).stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise PermissionError(f"{keys_file} is readable by other users; chmod 600 it")
    explicit = read_env_file(keys_file) if keys_file else {}
    keys, where = {}, {}
    for var, (prov, names) in PROVIDERS.items():
        for src_name, vals in ((str(keys_file), explicit), ("env", env),
                               (str(key_dir / f"{prov}.env"), read_env_file(key_dir / f"{prov}.env"))):
            hit = next((n for n in names if vals.get(n)), None)
            if hit:
                keys[var] = vals[hit]
                where[var] = f"${hit}" if src_name == "env" else src_name.replace(str(Path.home()), "~")
                break
    return keys, where


def file_text(keys: dict[str, str]) -> str:
    """Shell-sourceable `export VAR='value'` lines."""
    return "".join(f"export {k}={shlex.quote(v)}\n" for k, v in sorted(keys.items()))


class LocalKeyFile:
    """A temporary 600 file with the keys, in a 700 directory; removed on exit."""

    def __init__(self, keys: dict[str, str]):
        self.keys = keys

    def __enter__(self) -> Path:
        self.dir = Path(tempfile.mkdtemp(prefix="llmf-keys-"))
        os.chmod(self.dir, 0o700)
        self.path = self.dir / "keys.env"
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(file_text(self.keys))
        return self.path

    def __exit__(self, *exc) -> None:
        self.path.unlink(missing_ok=True)
        self.dir.rmdir()


def redact(text: str, keys: dict[str, str]) -> tuple[str, int]:
    """Replace every key value in `text`; return (text, number of replacements)."""
    n = 0
    for k, v in keys.items():
        if v and len(v) >= 8 and v in text:
            n += text.count(v)
            text = text.replace(v, f"[{k} redacted]")
    return text, n
