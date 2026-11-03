#!/usr/bin/env python3
"""SaladCloud helper for LLM Forensics (stdlib only).

    python scripts/salad/salad.py check                 # verify key/org/project, quota (read-only)
    python scripts/salad/salad.py gpus                  # GPU classes + prices for $PRIORITY
    python scripts/salad/salad.py create --start        # create a container group (BILLABLE once started)
    python scripts/salad/salad.py status                # group state + instances + SSH endpoints
    python scripts/salad/salad.py logs [--since 30]     # container stdout/stderr (last N minutes)
    python scripts/salad/salad.py ssh [-- cmd ...]      # shell (or one-off command) on the instance
    python scripts/salad/salad.py pull [--wait]         # scp /app/results/* -> ./results/
    python scripts/salad/salad.py s4 [--fetch]          # list / download the S4 results backups
    python scripts/salad/salad.py ls                    # all groups in the project + cost so far
    python scripts/salad/salad.py reap [-y]             # stop every running group in the project
    python scripts/salad/salad.py stop | start | delete

Safeguards (a group must never be left running): `create` passes SALAD_API_KEY and a hard
deadline (now + MAX_RUNTIME_HOURS, default 3) into the container; docker/watchdog.py stops
the group at the deadline, and the entrypoint stops it LLMF_LINGER_MINUTES (default 20)
after the pipeline finishes. `ls` / `reap` are the manual backstop.

Settings come from the environment, falling back to ~/.config/llm_forensics/salad.env
(KEY=VALUE lines; keep it chmod 600 and outside the repo):

    SALAD_API_KEY   required; sent only in the Salad-Api-Key header, never printed
    SALAD_ORG       default bohemian-miser          SALAD_PROJECT  default "default"
    NAME            container group name, default llmf-run
    IMAGE           default ghcr.io/bohemian-miser/llm-forensics:latest
    GPU_CLASS       regex over GPU class names, default "\\(24 GB\\)" (4090 / 3090 / 3090 Ti / A5000)
    PRIORITY        batch | low (default) | medium | high   (price and pre-emption tier)
    MAX_RUNTIME_HOURS   default 3: the container stops its own group after this
    LLMF_LINGER_MINUTES default 20: time after finishing to `pull` before the group self-stops
    CPU / MEMORY_GB / STORAGE_GB    default 8 / 30 / 100
    SSH_KEY         default ~/.ssh/salad_llm_forensics (public half registered in the Salad portal)
    LLMF_CONFIGS, LLMF_GIT_REF, LLMF_GCS_MODELS, LLMF_SKIP   forwarded to the container
    HF_TOKEN, ANTHROPIC_API_KEY   forwarded only if set (they become visible in the Salad portal)
    GHCR_USER / GHCR_TOKEN        optional registry credentials if the image is private
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
API = "https://api.salad.com/api/public"
ENV_FILE = Path.home() / ".config" / "llm_forensics" / "salad.env"
KNOWN_HOSTS = Path.home() / ".cache" / "llm_forensics" / "salad_known_hosts"


# --------------------------------------------------------------------------- config

def _load_env_file() -> dict[str, str]:
    out: dict[str, str] = {}
    for f in (ENV_FILE, ROOT / ".env"):
        if f.exists():
            for line in f.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    out.setdefault(k.strip().removeprefix("export ").strip(), v.strip().strip("'\""))
    return out


_FILE = _load_env_file()


def cfg(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key) or _FILE.get(key) or default


ORG = cfg("SALAD_ORG", "bohemian-miser")
PROJECT = cfg("SALAD_PROJECT", "default")
NAME = cfg("NAME", "llmf-run")
PRIORITY = cfg("PRIORITY", "low")


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


# --------------------------------------------------------------------------- HTTP

def api(method: str, path: str, body: dict | None = None, ok404: bool = False):
    key = cfg("SALAD_API_KEY")
    if not key:
        die(f"SALAD_API_KEY not set (env or {ENV_FILE})")
    req = urllib.request.Request(API + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Salad-Api-Key": key, "Content-Type": "application/json",
                                          "Accept": "application/json",
                                          "User-Agent": "llm-forensics/0.1 (+https://github.com/bohemian-miser/LLM_Forensics)"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            if e.code == 404 and ok404:
                return None
            if e.code == 429 and attempt < 4 and method == "GET":
                time.sleep(float(e.headers.get("Retry-After", 2 ** attempt)))
                continue
            detail = e.read().decode(errors="replace")[:800]
            die(f"{method} {path} -> HTTP {e.code}: {detail}")
    return None


def group_path(name: str | None = None) -> str:
    return f"/organizations/{ORG}/projects/{PROJECT}/containers/{name or NAME}"


# --------------------------------------------------------------------------- commands

def cmd_check(_a) -> int:
    q = api("GET", f"/organizations/{ORG}/quotas")
    groups = api("GET", f"/organizations/{ORG}/projects/{PROJECT}/containers")
    cq = q.get("container_groups_quotas", {})
    print(f"org={ORG} project={PROJECT}: key OK")
    print(f"replica quota: {cq.get('container_replicas_used')}/{cq.get('container_replicas_quota')} used")
    names = [g["name"] + f" ({g.get('current_state', {}).get('status')})" for g in groups.get("items", [])]
    print("container groups: " + (", ".join(names) or "none"))
    return 0


def gpu_classes() -> list[dict]:
    return api("GET", f"/organizations/{ORG}/gpu-classes").get("items", [])


def price(c: dict, prio: str) -> float | None:
    for p in c.get("prices", []):
        if p.get("priority") == prio:
            return float(p["price"])
    return None


def cmd_gpus(a) -> int:
    rx = re.compile(a.match or ".*")
    rows = sorted((c for c in gpu_classes() if rx.search(c["name"])), key=lambda c: price(c, PRIORITY) or 99)
    print(f"{'GPU class':34s} {'$/hr ' + PRIORITY:>12s}  high-demand  id")
    for c in rows:
        print(f"{c['name']:34s} {price(c, PRIORITY) or 0:12.3f}  {str(c.get('is_high_demand')):11s}  {c['id']}")
    return 0


def select_gpus(pattern: str) -> list[dict]:
    rx = re.compile(pattern)
    sel = [c for c in gpu_classes() if rx.search(c["name"])]
    if not sel:
        die(f"no GPU class matches {pattern!r}; see `salad.py gpus`")
    return sel


def cmd_create(a) -> int:
    if api("GET", group_path(), ok404=True) is not None:
        die(f"container group {NAME!r} already exists (status/delete it first)")
    gpus = select_gpus(cfg("GPU_CLASS", r"^(?!.*Laptop).*\(24 GB\)"))
    hours = float(cfg("MAX_RUNTIME_HOURS", "3"))
    deadline = int(time.time() + hours * 3600)
    env = {"PYTHONUNBUFFERED": "1", "SALAD_ORG": ORG, "LLMF_DEADLINE": str(deadline),
           "MAX_RUNTIME_HOURS": str(hours), "SALAD_API_KEY": cfg("SALAD_API_KEY")}
    for k in ("LLMF_CONFIGS", "LLMF_GIT_REF", "LLMF_GCS_MODELS", "LLMF_SKIP", "LLMF_LINGER_MINUTES", "LLMF_S4_BACKUP",
              "LLMF_INVESTIGATOR_DRY_RUN", "HF_TOKEN", "ANTHROPIC_API_KEY"):
        if cfg(k):
            env[k] = cfg(k)
    container = {
        "image": a.image or cfg("IMAGE", "ghcr.io/bohemian-miser/llm-forensics:latest"),
        "resources": {"cpu": int(cfg("CPU", "8")), "memory": int(float(cfg("MEMORY_GB", "30")) * 1024),
                      "gpu_classes": [g["id"] for g in gpus],
                      "storage_amount": int(float(cfg("STORAGE_GB", "100")) * 2**30),
                      "shm_size": 2048},
        "environment_variables": env,
        "priority": PRIORITY,
    }
    if a.command:
        container["command"] = ["/bin/bash", "-c", a.command]
    if cfg("GHCR_USER") and cfg("GHCR_TOKEN"):
        container["registry_authentication"] = {"basic": {"username": cfg("GHCR_USER"), "password": cfg("GHCR_TOKEN")}}
    body = {"name": NAME, "display_name": NAME, "container": container, "replicas": 1,
            "restart_policy": "on_failure", "autostart_policy": bool(a.start)}
    lo = min(price(g, PRIORITY) or 99 for g in gpus)
    hi = max(price(g, PRIORITY) or 0 for g in gpus)
    shown = {k: ("***" if k in ("HF_TOKEN", "ANTHROPIC_API_KEY", "SALAD_API_KEY") else v) for k, v in env.items()}
    print(f"creating {ORG}/{PROJECT}/{NAME}: image={container['image']} priority={PRIORITY}\n"
          f"  GPUs: {', '.join(g['name'] for g in gpus)}  (${lo:.3f}-${hi:.3f}/hr GPU at {PRIORITY}, plus CPU/RAM)\n"
          f"  env: {shown}\n  autostart: {bool(a.start)}\n"
          f"  hard stop by in-container watchdog at {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(deadline))}"
          + ("\n  WARNING: --command bypasses the entrypoint, so the watchdog does NOT run; stop it yourself"
             if a.command else ""))
    if not a.yes and sys.stdin.isatty():
        if input("proceed? [y/N] ").strip().lower() != "y":
            return 1
    g = api("POST", f"/organizations/{ORG}/projects/{PROJECT}/containers", body)
    print(f"created; status={g.get('current_state', {}).get('status')}  (next: `salad.py status`)")
    return 0


def instances() -> list[dict]:
    r = api("GET", group_path() + "/instances", ok404=True)
    return (r or {}).get("instances", [])


def cmd_status(a) -> int:
    g = api("GET", group_path(), ok404=True)
    if g is None:
        print(f"{NAME}: not found")
        return 1
    st = g.get("current_state", {})
    print(f"{NAME}: {st.get('status')}  counts={st.get('instance_status_counts')}  "
          f"start={st.get('start_time')}  {st.get('description') or ''}")
    for i in instances():
        ssh = f"ssh -p {i['ssh_port']} root@{i['ssh_ip']}" if i.get("ssh_ip") else "(no SSH endpoint yet)"
        print(f"  instance {i['id'][:8]} machine {i.get('machine_id', '')[:8]} state={i['state']} "
              f"ready={i.get('ready')} pull={i.get('pulling_progress')}  {ssh}")
    return 0


def cmd_logs(a) -> int:
    now = datetime.now(timezone.utc)
    body = {"start_time": (now - timedelta(minutes=a.since)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_time": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "sort_order": "asc", "page_size": a.n,
            "query": f'resource.labels.container_group_name = "{NAME}" AND resource.labels.project_name = "{PROJECT}"'
                     + ("" if a.system else ' AND resource.type = "container"')}
    r = api("POST", f"/organizations/{ORG}/log-entries", body)
    for it in r.get("items", []):
        msg = it.get("text_log") if it.get("text_log") is not None else json.dumps(it.get("json_log"))
        print(f"{it.get('time', '')[:19]} {it.get('resource', {}).get('type', '')[:10]:10s} {msg}")
    return 0


def _ssh_target(wait: bool = False, timeout: float = 3600) -> tuple[str, int, str | None]:
    t0 = time.time()
    while True:
        for i in instances():
            if i.get("state") == "running" and i.get("ssh_ip"):
                return i["ssh_ip"], int(i.get("ssh_port") or 22), i.get("ssh_host_key_fingerprint")
        if not wait or time.time() - t0 > timeout:
            die("no running instance with an SSH endpoint (see `salad.py status`)")
        time.sleep(20)


def _ssh_opts(host: str, port: int, fp: str | None) -> list[str]:
    """Pin the host key to the fingerprint the Salad API reports (no blind TOFU)."""
    key = os.path.expanduser(cfg("SSH_KEY", "~/.ssh/salad_llm_forensics"))
    if not os.path.exists(key):
        die(f"SSH key {key} not found (set SSH_KEY)")
    KNOWN_HOSTS.parent.mkdir(parents=True, exist_ok=True)
    opts = ["-i", key, "-o", "IdentitiesOnly=yes", "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
            "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=20"]
    if fp:
        scan = subprocess.run(["ssh-keyscan", "-p", str(port), host], capture_output=True, text=True, timeout=30).stdout
        good = []
        for line in scan.splitlines():
            if not line or line.startswith("#"):
                continue
            with tempfile.NamedTemporaryFile("w", delete=False) as t:
                t.write(line + "\n")
            f = subprocess.run(["ssh-keygen", "-lf", t.name], capture_output=True, text=True).stdout.split()
            os.unlink(t.name)
            if len(f) > 1 and f[1] == fp.split()[0]:   # API value looks like "SHA256:... (ED25519)"
                good.append(line)
        if not good:
            die(f"host key of {host}:{port} does not match API fingerprint {fp}; refusing to connect")
        hostspec = f"[{host}]:{port}"
        kept = [l for l in (KNOWN_HOSTS.read_text().splitlines() if KNOWN_HOSTS.exists() else [])
                if not l.startswith(hostspec + " ")]
        KNOWN_HOSTS.write_text("\n".join(kept + good) + "\n")
        opts += ["-o", "StrictHostKeyChecking=yes"]
    else:
        opts += ["-o", "StrictHostKeyChecking=accept-new"]
    return opts


def cmd_ssh(a) -> int:
    host, port, fp = _ssh_target(wait=a.wait)
    cmd = ["ssh", *_ssh_opts(host, port, fp), "-p", str(port), f"root@{host}"]
    if a.remote_cmd:
        # Salad execs one-off commands without a shell; wrap so pipes/; work
        cmd += ["--", "sh -c " + shlex.quote(" ".join(a.remote_cmd))]
        print("(Salad reports exit code 1 for one-off commands even on success; read the output)", file=sys.stderr)
    return subprocess.call(cmd)


def remote(host, port, opts, command: str) -> str:
    # Salad's SSH execs one-off commands without a shell, so wrap them in `sh -c`.
    r = subprocess.run(["ssh", *opts, "-p", str(port), f"root@{host}", "--", f"sh -c {shlex.quote(command)}"],
                       capture_output=True, text=True, timeout=120)
    return r.stdout


def cmd_pull(a) -> int:
    host, port, fp = _ssh_target(wait=a.wait)
    opts = _ssh_opts(host, port, fp)
    if a.wait:
        while True:
            st = remote(host, port, opts, f"cat {a.remote_dir}/.salad_status 2>/dev/null")
            print(f"[{time.strftime('%H:%M:%S')}] {st.strip() or '(no status yet)'}")
            if '"state":"done' in st or '"state":"idle' in st:
                break
            time.sleep(60)
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    # scp uses SFTP (OpenSSH >= 9), which Salad supports; plain rsync is not available.
    cmd = ["scp", "-r", *opts, "-P", str(port), f"root@{host}:{a.remote_dir}/.", str(dest)]
    print("$ " + " ".join(shlex.quote(c) for c in cmd))
    rc = subprocess.call(cmd)
    print(f"pulled into {dest}/ (rc={rc}); next: `make site` and commit results/*/" if rc == 0 else "scp failed")
    return rc


def _wait_status(want: set[str], timeout: float = 600) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout:
        g = api("GET", group_path(), ok404=True)
        s = "deleted" if g is None else g.get("current_state", {}).get("status")
        if s in want:
            return s
        time.sleep(10)
    return "timeout"


def cmd_ls(_a) -> int:
    """Every group in the project with an estimated cost since its last start."""
    groups = api("GET", f"/organizations/{ORG}/projects/{PROJECT}/containers").get("items", [])
    classes = {c["id"]: c for c in gpu_classes()}
    now = datetime.now(timezone.utc)
    total = 0.0
    print(f"{'group':24s} {'status':10s} {'running':>7s} {'since start':>12s} {'$/hr max':>9s} {'est. $':>8s}  gpus")
    for g in groups:
        st = g.get("current_state", {})
        c = g.get("container", {})
        prio = c.get("priority") or "high"
        rate = max((price(classes[i], prio) or 0) for i in c.get("resources", {}).get("gpu_classes", []) if i in classes) \
            if c.get("resources", {}).get("gpu_classes") else 0.0
        running = (st.get("instance_status_counts") or {}).get("running_count", 0)
        hrs, cost = 0.0, 0.0
        if st.get("status") in ("running", "deploying") and st.get("start_time"):
            t0 = datetime.fromisoformat(st["start_time"].replace("Z", "+00:00")[:26] + "+00:00"
                                        if "." in st["start_time"] else st["start_time"].replace("Z", "+00:00"))
            hrs = (now - t0).total_seconds() / 3600
            cost = hrs * rate * max(1, g.get("replicas", 1)) if running else 0.0
            total += cost
        names = ", ".join(classes[i]["name"].split(" (")[0] for i in c.get("resources", {}).get("gpu_classes", [])
                          if i in classes)
        print(f"{g['name']:24s} {st.get('status', ''):10s} {running:7d} {hrs:10.2f} h {rate:9.3f} {cost:8.3f}  {names}")
    print(f"total estimated spend since last starts: ${total:.3f} (upper bound: max GPU price, download time excluded"
          " by Salad but counted here)")
    return 0


def cmd_reap(a) -> int:
    groups = api("GET", f"/organizations/{ORG}/projects/{PROJECT}/containers").get("items", [])
    live = [g["name"] for g in groups if g.get("current_state", {}).get("status") not in ("stopped", "failed")]
    if not live:
        print("nothing running")
        return 0
    print("will stop: " + ", ".join(live))
    if not a.yes and sys.stdin.isatty() and input("proceed? [y/N] ").strip().lower() != "y":
        return 1
    for n in live:
        api("POST", group_path(n) + "/stop", {})
        print(f"stop requested: {n}")
    return 0


S4 = "https://storage-api.salad.com"


def cmd_s4(a) -> int:
    key = cfg("SALAD_API_KEY")
    hdr = {"Salad-Api-Key": key, "User-Agent": "llm-forensics/0.1"}
    with urllib.request.urlopen(urllib.request.Request(f"{S4}/organizations/{ORG}/files", headers=hdr)) as r:
        files = [f for f in json.load(r).get("files", []) if "/files/llm-forensics/" in f["url"]]
    files.sort(key=lambda f: f.get("uploaded", ""))
    for f in files:
        print(f"{f.get('uploaded', '')[:19]}  {f.get('size', 0) / 1e6:7.2f} MB  {f['url'].split('/files/', 1)[1]}")
    if not files:
        print("no llm-forensics backups in S4")
        return 0
    if a.fetch:
        f = files[-1]
        url = f["url"].split("?")[0]
        tmp = Path(tempfile.mkdtemp()) / "results.tar.gz"
        with urllib.request.urlopen(urllib.request.Request(url, headers=hdr)) as r, open(tmp, "wb") as out:
            out.write(r.read())
        # extract only results/<name>/... JSON/PNG/log files, never anything outside results/
        import tarfile
        with tarfile.open(tmp) as t:
            members = [m for m in t.getmembers() if m.isfile() and m.name.startswith("results/")
                       and ".." not in m.name and m.name.endswith((".json", ".png", ".svg", ".log", ".txt"))]
            t.extractall(ROOT, members=members)
        print(f"extracted {len(members)} files from {url.split('/files/', 1)[1]} into results/")
    return 0


def cmd_start(_a) -> int:
    api("POST", group_path() + "/start", {})
    print("start requested")
    return cmd_status(_a)


def cmd_stop(_a) -> int:
    api("POST", group_path() + "/stop", {})
    print("stop requested; " + _wait_status({"stopped", "deleted"}, 300))
    return 0


def cmd_delete(a) -> int:
    if api("GET", group_path(), ok404=True) is None:
        print(f"{NAME}: already gone")
        return 0
    if not a.yes and sys.stdin.isatty():
        if input(f"delete container group {NAME!r}? [y/N] ").strip().lower() != "y":
            return 1
    api("DELETE", group_path())
    gone = _wait_status({"deleted"}, 300) == "deleted"
    print(f"{NAME}: {'deleted (verified: API returns 404)' if gone else 'delete requested but still visible'}")
    return 0 if gone else 1


def main(argv: list[str] | None = None) -> int:
    global NAME
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", help="container group name (default $NAME or llmf-run)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    p = sub.add_parser("gpus"); p.add_argument("match", nargs="?")
    p = sub.add_parser("create")
    p.add_argument("--start", action="store_true", help="start immediately (billing begins)")
    p.add_argument("--image"); p.add_argument("--command", help="bash -c override of the entrypoint")
    p.add_argument("-y", "--yes", action="store_true")
    sub.add_parser("status"); sub.add_parser("start"); sub.add_parser("stop"); sub.add_parser("ls")
    p = sub.add_parser("reap"); p.add_argument("-y", "--yes", action="store_true")
    p = sub.add_parser("s4"); p.add_argument("--fetch", action="store_true", help="download + extract the latest")
    p = sub.add_parser("delete"); p.add_argument("-y", "--yes", action="store_true")
    p = sub.add_parser("logs"); p.add_argument("--since", type=int, default=60, help="minutes")
    p.add_argument("-n", type=int, default=500); p.add_argument("--system", action="store_true")
    p = sub.add_parser("ssh"); p.add_argument("--wait", action="store_true"); p.add_argument("remote_cmd", nargs="*")
    p = sub.add_parser("pull"); p.add_argument("--wait", action="store_true", help="wait for the run to finish")
    p.add_argument("--remote-dir", default="/app/results"); p.add_argument("--dest", default=str(ROOT / "results"))
    a = ap.parse_args(argv)
    if a.name:
        NAME = a.name
    return globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    raise SystemExit(main())
