"""Send a Python script, inline snippet, or toolkit module call to a running `toolkit.serve`.

Usage:
    python -m llm_forensics.toolkit.run script.py [args...]
    python -m llm_forensics.toolkit.run -m spectra --k 5 --lens 5
    python -m llm_forensics.toolkit.run -c "print(W(5).shape)"
    python -m llm_forensics.toolkit.run --status
    python -m llm_forensics.toolkit.run --stop

Scripts run inside the warm `toolkit.serve` process with `lm`, `model`, `tok`, `corpus`,
`W(layer)`, `keys(texts, layer)` and `lens(vec, k=10)` pre-bound in globals. Toolkit modules run
via `-m <name>` omit the positional `model` argument (or pass the same model name) and reuse the
already-loaded model and output directory without reloading weights.
"""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys
from typing import Sequence

from .serve import resolve_socket

__all__ = ["request", "main"]


def request(payload: dict, socket_path: str | Path | None = None, cwd: str | Path | None = None) -> dict:
    """Send `payload` to `toolkit.serve` and return `{"rc", "stdout", "stderr"}`."""
    sock_path = resolve_socket(socket_path, cwd=cwd)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(str(sock_path))
        s.sendall(json.dumps(payload).encode("utf-8"))
        s.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        while True:
            b = s.recv(65536)
            if not b:
                break
            chunks.append(b)
    return json.loads(b"".join(chunks).decode("utf-8"))


def _print_help() -> None:
    print(
        "usage: python -m llm_forensics.toolkit.run [--socket PATH] "
        "(script.py [args...] | -m MODULE [args...] | -c CODE [args...] | --status | --stop)\n\n"
        "Run a script or toolkit module inside a running `toolkit.serve` process without reloading\n"
        "the model. Scripts receive pre-bound globals: lm, model, tok, corpus, W(layer),\n"
        "keys(texts, layer), lens(vec, k=10)."
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    sock_arg: str | None = None

    while args and args[0].startswith("--socket"):
        if args[0] == "--socket":
            if len(args) < 2:
                print("error: --socket requires a path", file=sys.stderr)
                return 2
            sock_arg = args[1]
            args = args[2:]
        elif args[0].startswith("--socket="):
            sock_arg = args[0].split("=", 1)[1]
            args = args[1:]
        else:
            break

    if not args or args[0] in ("-h", "--help"):
        _print_help()
        return 0 if args else 2

    cwd = str(Path.cwd())
    if args[0] == "--stop":
        resp = request({"kind": "stop", "cwd": cwd}, socket_path=sock_arg, cwd=cwd)
    elif args[0] == "--status":
        resp = request({"kind": "ping", "cwd": cwd}, socket_path=sock_arg, cwd=cwd)
    elif args[0] == "-m":
        if len(args) < 2:
            print("error: -m requires a module name", file=sys.stderr)
            return 2
        resp = request({"kind": "module", "target": args[1], "argv": args[2:], "cwd": cwd},
                       socket_path=sock_arg, cwd=cwd)
    elif args[0] == "-c":
        if len(args) < 2:
            print("error: -c requires Python code", file=sys.stderr)
            return 2
        resp = request({"kind": "code", "target": args[1], "argv": args[2:], "cwd": cwd},
                       socket_path=sock_arg, cwd=cwd)
    else:
        resp = request({"kind": "script", "target": args[0], "argv": args[1:], "cwd": cwd},
                       socket_path=sock_arg, cwd=cwd)

    if resp.get("stdout"):
        sys.stdout.write(resp["stdout"])
        sys.stdout.flush()
    if resp.get("stderr"):
        sys.stderr.write(resp["stderr"])
        sys.stderr.flush()
    return int(resp.get("rc", 0))


if __name__ == "__main__":
    raise SystemExit(main())
