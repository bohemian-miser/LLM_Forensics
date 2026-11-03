"""Turn host specs into SSH targets.

    root@203.0.113.7:2222      any SSH server; root, or a user with passwordless sudo
    salad:llmf-cr1             a SaladCloud container group, via scripts/salad/salad.py

For Salad the endpoint and the pinned host key come from `salad._ssh_target` / `salad._ssh_opts`;
creating, starting and stopping groups goes through the `salad.py` command line.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading

from .batch import ROOT
from .remote import Host
from .commands import apt_install

SALAD_PY = ROOT / "scripts" / "salad" / "salad.py"
# Default image for cleanroom hosts: small, so it pulls fast on home connections. Torch comes from uv.
SALAD_IMAGE = "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04"
# salad.py create copies these into the container's environment, where everyone in the Salad
# project can read them. Cleanroom keys go to the host as a file instead.
NOT_IN_CONTAINER_ENV = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "HF_TOKEN")
_salad_lock = threading.Lock()


def is_salad(spec: str) -> bool:
    return spec.startswith("salad:")


def label(spec: str) -> str:
    """A file-name-safe label for logs and hosts.json."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", spec)


def _salad_module():
    sys.path.insert(0, str(SALAD_PY.parent))
    import salad  # noqa: E402
    return salad


def resolve(spec: str, ssh_key: str | None = None, wait: bool = True) -> Host:
    if not is_salad(spec):
        return Host.parse(spec, key=ssh_key, label=label(spec))
    name = spec.split(":", 1)[1]
    with _salad_lock:   # salad.py keeps the group name in a module global
        salad = _salad_module()
        salad.NAME = name
        host, port, fp = salad._ssh_target(wait=wait)
        opts = salad._ssh_opts(host, port, fp)
    return Host(host=host, port=port, user="root", opts=opts, label=label(spec))


def salad_cli(*args: str, env: dict[str, str] | None = None, timeout: float = 900) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SALAD_PY), *args], capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, env=env, timeout=timeout)


def salad_status(name: str) -> str:
    """'not found', or the group status ('running', 'stopped', 'deploying', ...)."""
    r = salad_cli("--name", name, "status")
    first = (r.stdout.strip().splitlines() or [""])[0]
    m = re.match(rf"{re.escape(name)}: (\S+(?: found)?)", first)
    if not m:
        raise RuntimeError(f"salad.py status {name}: {r.stdout.strip()[:300]} {r.stderr.strip()[:300]}")
    return m.group(1)


def container_command(repo: str, ref: str) -> str:
    """Keep the container alive for SSH, and run docker/watchdog.py so the group stops itself at the
    deadline salad.py sets (MAX_RUNTIME_HOURS)."""
    m = re.match(r"https://github\.com/([^/]+/[^/.]+)(?:\.git)?$", repo)
    lines = [f"({apt_install('python3', 'curl', 'git', 'ca-certificates')}) || true"]
    if m:
        raw = f"https://raw.githubusercontent.com/{m.group(1)}/{ref}/docker/watchdog.py"
        lines.append(f"curl -fsSL {raw} -o /root/watchdog.py && (python3 /root/watchdog.py loop &)")
    lines += ["echo '[cleanroom] host up'", "exec sleep infinity"]
    return "export DEBIAN_FRONTEND=noninteractive\n" + "\n".join(lines)


def _clean_env(extra: dict[str, str]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in NOT_IN_CONTAINER_ENV}
    env.update(extra)
    return env


def forwarded_secrets(env: dict[str, str]) -> list[str]:
    """Key variables salad.py would still forward with `env` (it also reads salad.env and the
    repo's .env)."""
    code = (f"import sys; sys.path.insert(0, {str(SALAD_PY.parent)!r}); import salad; "
            f"print(','.join(k for k in {NOT_IN_CONTAINER_ENV!r} if salad.cfg(k)))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                       stdin=subprocess.DEVNULL, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"could not check what salad.py forwards: {r.stderr.strip()[-300:]}")
    return [k for k in r.stdout.strip().split(",") if k]


def salad_up(name: str, repo: str, ref: str, max_hours: float, image: str | None = None,
             priority: str | None = None) -> str:
    """Create and start the group if it doesn't exist; use it if it's running. Returns what happened."""
    st = salad_status(name)
    if st in ("running", "deploying", "pending"):
        return f"using existing group ({st})"
    if st == "stopped":
        raise RuntimeError(
            f"Salad group {name} exists and is stopped. Its watchdog deadline was fixed when it was created, so "
            f"starting it may stop it again at once. Delete it (`salad.py --name {name} delete`) and use a new name, "
            f"or start it yourself (`salad.py --name {name} start`) if the deadline is still ahead.")
    if st != "not found":
        raise RuntimeError(f"Salad group {name} is {st}; wait for it or delete it")
    env = _clean_env({"MAX_RUNTIME_HOURS": f"{max_hours:.1f}", **({"PRIORITY": priority} if priority else {})})
    leaks = forwarded_secrets(env)
    if leaks:
        raise RuntimeError(
            f"salad.py would copy {', '.join(leaks)} into the container environment (from the repo's .env or "
            "~/.config/llm_forensics/salad.env), where anyone in the Salad project can read it. Move those keys to "
            "~/.config/llm_forensics/<provider>.env and try again.")
    r = salad_cli("--name", name, "create", "--start", "-y", "--image", image or SALAD_IMAGE,
                  "--command", container_command(repo, ref), env=env)
    if r.returncode != 0:
        raise RuntimeError(f"salad.py create {name} failed: {(r.stdout + r.stderr).strip()[-800:]}")
    return f"created, max {max_hours:.1f} h"


def salad_stop(name: str) -> str:
    r = salad_cli("--name", name, "stop")
    return (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else f"rc={r.returncode}"

