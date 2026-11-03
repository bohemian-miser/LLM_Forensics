#!/usr/bin/env python3
"""In-container safety net: stop this container's own Salad container group.

Salad restarts containers that exit, so a finished (or runaway) job has to stop its group
through the API to stop billing. Used by docker/entrypoint.sh:

    python docker/watchdog.py loop &    # stop the group once LLMF_DEADLINE (epoch s) passes
    python docker/watchdog.py stop      # stop the group now (pipeline finished)

Needs SALAD_API_KEY (container env var set by scripts/salad/salad.py create). Group/project/
org names come from the variables Salad injects (SALAD_CONTAINER_GROUP_NAME, ...). Outside
Salad (no such variables) it does nothing. Stdlib only.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://api.salad.com/api/public"


def log(msg: str) -> None:
    print(f"[watchdog] {msg}", flush=True)


def stop_group(reason: str) -> bool:
    key = os.environ.get("SALAD_API_KEY")
    org = os.environ.get("SALAD_ORGANIZATION_NAME") or os.environ.get("SALAD_ORG")
    proj = os.environ.get("SALAD_PROJECT_NAME")
    grp = os.environ.get("SALAD_CONTAINER_GROUP_NAME")
    if not (key and org and proj and grp):
        log(f"cannot self-stop ({reason}): missing SALAD_API_KEY or Salad-injected names")
        return False
    url = f"{API}/organizations/{org}/projects/{proj}/containers/{grp}/stop"
    req = urllib.request.Request(url, method="POST", data=b"{}", headers={
        "Salad-Api-Key": key, "Content-Type": "application/json",
        "User-Agent": "llm-forensics-watchdog/0.1"})
    for attempt in range(10):
        try:
            with urllib.request.urlopen(req, timeout=30):
                log(f"stop requested for {org}/{proj}/{grp}: {reason}")
                return True
        except urllib.error.HTTPError as e:
            log(f"stop failed HTTP {e.code}: {e.read()[:200]!r}")
        except Exception as e:  # noqa: BLE001
            log(f"stop failed: {e}")
        time.sleep(min(60, 5 * 2 ** attempt))
    return False


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "loop"
    if mode == "stop":
        return 0 if stop_group(" ".join(sys.argv[2:]) or "pipeline finished") else 1
    deadline = float(os.environ.get("LLMF_DEADLINE") or 0)
    if not deadline:
        hours = float(os.environ.get("MAX_RUNTIME_HOURS") or 3)
        deadline = time.time() + hours * 3600
    log(f"group will be stopped at {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime(deadline))}")
    while time.time() < deadline:
        time.sleep(min(60, max(1, deadline - time.time())))
    while True:  # keep trying until Salad kills us
        stop_group("MAX_RUNTIME_HOURS exceeded")
        time.sleep(300)


if __name__ == "__main__":
    raise SystemExit(main())
