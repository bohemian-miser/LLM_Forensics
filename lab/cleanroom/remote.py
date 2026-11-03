"""SSH to a GPU host: run commands, copy files, run long jobs detached.

Works with any SSH target (`user@host:port` plus a key). Commands are wrapped so that they behave
the same on Salad's SSH gateway as on a normal sshd:

- The gateway exits 1 for every command, so the remote shell prints `__LLMF_RC=<status>` last and
  `parse_rc` reads the real status from stdout.
- One-off commands run without a shell, so everything goes through `bash -c '...'`.
- Remote stdin stays open with no TTY, so `git` and pagers can hang: the wrapper runs
  `exec </dev/null` and sets GIT_TERMINAL_PROMPT=0, GIT_PAGER=cat and PAGER=cat.

Non-root users get `sudo -n -H` in front of the wrapper, and file copies go through a private
staging directory in their home, because the cleanroom paths are under /root.
"""
from __future__ import annotations

import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

RC_MARK = "__LLMF_RC="
PRELUDE = ("exec </dev/null; export GIT_TERMINAL_PROMPT=0 GIT_PAGER=cat PAGER=cat LC_ALL=C.UTF-8 "
           "DEBIAN_FRONTEND=noninteractive; set -o pipefail; ")
SPEC_RE = re.compile(r"^(?:(?P<user>[^@\s]+)@)?(?P<host>[^:@\s]+)(?::(?P<port>\d+))?$")
STAGE = ".llmf-stage"


class RemoteError(RuntimeError):
    pass


def wrap(command: str, sudo: bool = False) -> str:
    """The string passed to `ssh ... -- <it>`: bash -c, stdin closed, status marker last. The command
    runs in a subshell, so `set -e` or `exit` in it can't skip the marker."""
    inner = PRELUDE + "(\n" + command + "\n); echo " + RC_MARK + "$?"
    s = "bash -c " + shlex.quote(inner)
    return "sudo -n -H " + s if sudo else s


def parse_rc(stdout: str) -> tuple[int | None, str]:
    """(remote exit status, stdout without the marker). None when the marker is missing, which means
    the connection failed before the command finished."""
    head, sep, tail = stdout.rpartition(RC_MARK)
    if not sep:
        return None, stdout
    m = re.match(r"(\d+)", tail)
    return (int(m.group(1)) if m else None), head


@dataclass
class Result:
    rc: int | None
    out: str
    err: str

    @property
    def ok(self) -> bool:
        return self.rc == 0


@dataclass
class Host:
    """An SSH target. `opts` are extra ssh options (identity, known-hosts pinning)."""
    host: str
    port: int = 22
    user: str = "root"
    opts: list[str] = field(default_factory=list)
    label: str = ""

    @classmethod
    def parse(cls, spec: str, key: str | None = None, label: str | None = None) -> "Host":
        m = SPEC_RE.match(spec.strip())
        if not m:
            raise ValueError(f"host {spec!r}: expected user@host[:port]")
        opts = ["-i", str(Path(key).expanduser()), "-o", "IdentitiesOnly=yes"] if key else []
        opts += ["-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=20"]
        return cls(host=m["host"], port=int(m["port"] or 22), user=m["user"] or "root", opts=opts,
                   label=label or spec)

    @property
    def sudo(self) -> bool:
        return self.user != "root"

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"

    def ssh_argv(self, command: str) -> list[str]:
        return ["ssh", "-T", "-o", "BatchMode=yes", *self.opts, "-p", str(self.port), self.target, "--",
                wrap(command, self.sudo)]

    def scp_argv(self, src: str, dst: str, recursive: bool = False) -> list[str]:
        return ["scp", "-q", "-o", "BatchMode=yes", *(["-r"] if recursive else []), *self.opts,
                "-P", str(self.port), src, dst]

    # ------------------------------------------------------------------ commands

    def run(self, command: str, timeout: float = 900, retries: int = 2) -> Result:
        """Run `command` in bash on the host. Retries only when the connection itself failed
        (no status marker), so the command should be safe to repeat."""
        last = Result(None, "", "")
        for attempt in range(retries + 1):
            try:
                r = subprocess.run(self.ssh_argv(command), capture_output=True, text=True, timeout=timeout,
                                   stdin=subprocess.DEVNULL)
            except subprocess.TimeoutExpired:
                return Result(None, "", f"timed out after {timeout:.0f}s")
            rc, out = parse_rc(r.stdout)
            last = Result(rc, out, r.stderr)
            if rc is not None:
                return last
            if attempt < retries:
                time.sleep(10 * (attempt + 1))
        return last

    def check(self, command: str, timeout: float = 900, what: str = "") -> str:
        r = self.run(command, timeout=timeout)
        if not r.ok:
            tail = (r.out + r.err)[-2000:]
            raise RemoteError(f"{self.label}: {what or command[:120]} failed (rc={r.rc})\n{tail}")
        return r.out

    # ------------------------------------------------------------------ files

    def put(self, local: Path, remote: str, mode: str = "600") -> None:
        """Copy one file to `remote`, owned by root with `mode`. The remote file is created with
        that mode before any bytes arrive."""
        d = shlex.quote(str(Path(remote).parent))
        if not self.sudo:
            self.check(f"umask 077; mkdir -p {d} && install -m {mode} /dev/null {shlex.quote(remote)}",
                       what=f"prepare {remote}")
            self._scp(str(local), f"{self.target}:{remote}")
            return
        stage = f"{STAGE}/{Path(remote).name}"
        self._user_cmd(f"umask 077; mkdir -p ~/{STAGE} && chmod 700 ~/{STAGE} && install -m 600 /dev/null ~/{stage}")
        self._scp(str(local), f"{self.target}:{stage}")
        self.check(f"mkdir -p {d} && install -m {mode} -o root -g root ~{self.user}/{stage} {shlex.quote(remote)}"
                   f" && rm -f ~{self.user}/{stage}", what=f"install {remote}")

    def get(self, remote: str, local: Path) -> None:
        """Copy a remote file or directory (its contents if `remote` ends in `/.`) to `local`."""
        (local if remote.endswith("/.") else local.parent).mkdir(parents=True, exist_ok=True)
        if not self.sudo:
            self._scp(f"{self.target}:{remote}", str(local), recursive=True)
            return
        stage = f"{STAGE}/get"
        src = remote[:-2] if remote.endswith("/.") else remote
        self.check(f"rm -rf ~{self.user}/{stage} && mkdir -p ~{self.user}/{STAGE} && "
                   f"cp -r {shlex.quote(src)} ~{self.user}/{stage} && chown -R {self.user} ~{self.user}/{STAGE}",
                   what=f"stage {remote}")
        self._scp(f"{self.target}:{stage}" + ("/." if remote.endswith("/.") else ""), str(local), recursive=True)
        self._user_cmd(f"rm -rf ~/{stage}")

    def _scp(self, src: str, dst: str, recursive: bool = False) -> None:
        r = subprocess.run(self.scp_argv(src, dst, recursive), capture_output=True, text=True,
                           stdin=subprocess.DEVNULL, timeout=3600)
        if r.returncode != 0:
            raise RemoteError(f"{self.label}: scp {src} -> {dst} failed: {r.stderr.strip()[-500:]}")

    def _user_cmd(self, command: str) -> None:
        """Run as the login user, without sudo (for the staging directory)."""
        argv = ["ssh", "-T", "-o", "BatchMode=yes", *self.opts, "-p", str(self.port), self.target, "--",
                wrap(command)]
        r = subprocess.run(argv, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=300)
        if parse_rc(r.stdout)[0] != 0:
            raise RemoteError(f"{self.label}: {command[:80]} failed: {r.stderr.strip()[-500:]}")

    # ------------------------------------------------------------------ detached jobs

    def start_job(self, jobdir: str, name: str, command: str, restart: bool = False) -> str:
        """Start `command` under nohup/setsid so it survives a dropped SSH connection.
        Returns 'started', 'running' or 'done' (a finished job is not rerun unless `restart`)."""
        return self.check(job_start_command(jobdir, name, command, restart), what=f"start job {name}").strip()

    def wait_job(self, jobdir: str, name: str, poll: float = 60, timeout: float = 6 * 3600,
                 on_poll=None) -> int:
        """Block until the job writes its exit status; return it."""
        t0 = time.time()
        while True:
            r = self.run(job_poll_command(jobdir, name), timeout=120, retries=5)
            state = r.out.strip().splitlines()[-1] if r.ok and r.out.strip() else "UNKNOWN"
            if state.startswith("RC="):
                return int(state[3:])
            if state == "DEAD":
                return 255
            if on_poll:
                on_poll(state)
            if time.time() - t0 > timeout:
                raise RemoteError(f"{self.label}: job {name} still {state} after {timeout:.0f}s")
            time.sleep(poll)


def job_start_command(jobdir: str, name: str, command: str, restart: bool = False) -> str:
    j = shlex.quote(jobdir)
    rc, pid, log = (f"{j}/{name}.{x}" for x in ("rc", "pid", "log"))
    # The subshell keeps `set -e` or `exit` inside `command` from skipping the status file.
    body = shlex.quote(f"(\n{command}\n)\necho $? > {rc}.tmp && mv {rc}.tmp {rc}")
    return (f"umask 077; mkdir -p {j}; " + (f"rm -f {rc}; " if restart else "")
            + f"if [ -s {rc} ]; then echo done; "
            f"elif [ -s {pid} ] && kill -0 \"$(cat {pid})\" 2>/dev/null; then echo running; "
            f"else nohup setsid bash -c {body} > {log} 2>&1 < /dev/null & echo $! > {pid}; echo started; fi")


def job_poll_command(jobdir: str, name: str) -> str:
    j = shlex.quote(jobdir)
    return (f"if [ -s {j}/{name}.rc ]; then echo RC=$(cat {j}/{name}.rc); "
            f"elif [ -s {j}/{name}.pid ] && kill -0 \"$(cat {j}/{name}.pid)\" 2>/dev/null; then echo RUNNING; "
            f"else echo DEAD; fi")
