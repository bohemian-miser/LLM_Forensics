"""Keep a model, tokenizer, and cached corpus loaded in memory across scripts and toolkit calls.

Start the daemon once in the background:

    python -m llm_forensics.toolkit.serve <model-dir> &

Then run custom scripts or toolkit modules against the warm process via `toolkit.run`:

    python -m llm_forensics.toolkit.run script.py [args...]
    python -m llm_forensics.toolkit.run -m spectra --k 5
    python -m llm_forensics.toolkit.run -c "print(lens(W(5)[:, 0], 5))"
    python -m llm_forensics.toolkit.run --stop

Scripts executed by `serve` run with pre-bound globals:
  - `lm`: `LoadedModel`
  - `model`: `lm.model` (eval mode)
  - `tok`: `lm.tok`
  - `corpus`: cached `Corpus` (loaded or lazily generated on first attribute access)
  - `W(layer)`: `[d_model, d_mlp]` weight view (`arch.mlp_out_weight_kv(model, layer)`)
  - `keys(texts, layer)`: `[B, T, d_mlp]` float32 CPU MLP keys for `texts` (`str` or `list[str]`)
  - `lens(vec, k=10)`: top-`k` decoded tokens of residual direction `vec` (`interpret.lens_direction`)

Hugging Face loading progress bars and warnings are silenced by default. Any forward hooks left
registered on `model` by a script are cleaned up after each request.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import runpy
import socket
import sys
import traceback
from typing import Sequence

import torch

from llm_forensics.arch import mlp_out_weight_kv
from llm_forensics.forensics.interpret import encode_texts, lens_direction, run_capture

from . import _common as C
from .corpus import Corpus, find_cached, load_or_generate

__all__ = ["STATE_FILE", "LazyCorpus", "make_helpers", "resolve_socket", "serve", "main"]

STATE_FILE = ".llmf-serve.json"


class LazyCorpus:
    """Proxy over the served model's cached `Corpus` that loads an existing corpus immediately if
    present on disk and generates one lazily on first use otherwise."""

    def __init__(self, lm, out_dir: Path, corpus_kwargs: dict):
        self._lm = lm
        self._out_dir = Path(out_dir)
        self._kwargs = dict(corpus_kwargs)
        self._corpus: Corpus | None = find_cached(lm, self._out_dir)

    def get(self) -> Corpus:
        srv = C.active_serve()
        if srv is not None and isinstance(srv.corpus, Corpus):
            self._corpus = srv.corpus
        if self._corpus is None:
            self._corpus = find_cached(self._lm, self._out_dir)
        if self._corpus is None:
            self._corpus = load_or_generate(self._lm, self._out_dir, **self._kwargs)
        if srv is not None:
            srv.corpus = self._corpus
        return self._corpus

    def __getattr__(self, name: str):
        return getattr(self.get(), name)

    def __repr__(self) -> str:
        if self._corpus is not None:
            return repr(self._corpus)
        return f"LazyCorpus(unloaded, out={str(self._out_dir)!r})"


def make_helpers(lm, corpus_obj) -> dict:
    """Build the pre-bound namespace exposed to scripts run inside `toolkit.serve`."""
    model, tok, _ = C.unpack(lm)

    def W(layer: int) -> torch.Tensor:
        return mlp_out_weight_kv(model, layer)

    @torch.no_grad()
    def keys(texts: str | Sequence[str], layer: int) -> torch.Tensor:
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        if not batch:
            return torch.empty((0, 0, W(layer).shape[1]), dtype=torch.float32)
        dev = C.device_of(model)
        ids, mask, _ = encode_texts(tok, batch, add_bos=True, max_len=256)
        captured = run_capture(model, ids.to(dev), mask.to(dev), [layer])[layer].float().cpu()
        return captured

    @torch.no_grad()
    def lens(vec: torch.Tensor, k: int = 10) -> list[dict]:
        return lens_direction(model, tok, vec, k=k)

    return {
        "lm": lm,
        "model": model,
        "tok": tok,
        "corpus": corpus_obj,
        "W": W,
        "keys": keys,
        "lens": lens,
    }


def _snapshot_hooks(model: torch.nn.Module) -> dict[int, tuple[set, set, set]]:
    snap: dict[int, tuple[set, set, set]] = {}
    for m in model.modules():
        snap[id(m)] = (
            set(m._forward_pre_hooks.keys()),
            set(m._forward_hooks.keys()),
            set(m._backward_hooks.keys()),
        )
    return snap


def _restore_hooks(model: torch.nn.Module, snap: dict[int, tuple[set, set, set]]) -> None:
    for m in model.modules():
        orig = snap.get(id(m))
        if orig is None:
            continue
        pre_k, fwd_k, bwd_k = orig
        for d, keep in ((m._forward_pre_hooks, pre_k), (m._forward_hooks, fwd_k), (m._backward_hooks, bwd_k)):
            for k in list(d.keys()):
                if k not in keep:
                    del d[k]


def _short_unix_socket(requested: str | Path | None) -> tuple[Path, Path | None]:
    """Return `(bind_socket_path, state_pointer_path)` keeping `AF_UNIX` path under 100 bytes."""
    uid = os.getuid() if hasattr(os, "getuid") else 0
    if requested:
        req = Path(requested).expanduser()
        if len(str(req)) <= 96:
            return req, None
        digest = hashlib.sha256(str(req.resolve() if req.parent.exists() else req).encode()).hexdigest()[:12]
        return Path(f"/tmp/llmf-serve-{uid}-{digest}.sock"), req
    env_sock = os.environ.get("LLMF_SERVE_SOCKET")
    if env_sock:
        return _short_unix_socket(env_sock)
    return Path(f"/tmp/llmf-serve-{uid}-{os.getpid()}.sock"), Path.cwd() / STATE_FILE


def resolve_socket(socket_arg: str | Path | None = None, cwd: str | Path | None = None) -> Path:
    """Locate an active `toolkit.serve` Unix domain socket."""
    cand_arg = socket_arg or os.environ.get("LLMF_SERVE_SOCKET")
    if cand_arg:
        p = Path(cand_arg).expanduser()
        if p.is_file():
            try:
                data = json.loads(p.read_text())
                if "socket" in data:
                    return Path(data["socket"])
            except Exception:
                pass
        if len(str(p)) > 96:
            uid = os.getuid() if hasattr(os, "getuid") else 0
            digest = hashlib.sha256(str(p.resolve() if p.parent.exists() else p).encode()).hexdigest()[:12]
            return Path(f"/tmp/llmf-serve-{uid}-{digest}.sock")
        return p

    search_dir = Path(cwd or Path.cwd()).resolve()
    for d in (search_dir, *search_dir.parents):
        sf = d / STATE_FILE
        if sf.is_file():
            try:
                data = json.loads(sf.read_text())
                sock = Path(data["socket"])
                if sock.exists():
                    return sock
            except Exception:
                continue
    raise FileNotFoundError(
        f"no running toolkit.serve found (checked {STATE_FILE} from {search_dir}); "
        "start `python -m llm_forensics.toolkit.serve <model>` first or pass --socket"
    )


def _recv_json(conn: socket.socket) -> dict:
    chunks: list[bytes] = []
    while True:
        b = conn.recv(65536)
        if not b:
            break
        chunks.append(b)
    return json.loads(b"".join(chunks).decode("utf-8"))


def _resolve_module_name(mod: str) -> str:
    if mod.startswith("llm_forensics."):
        return mod
    if "." not in mod:
        cand = f"llm_forensics.toolkit.{mod}"
        try:
            importlib.import_module(cand)
            return cand
        except ImportError:
            pass
    return mod


def _execute_request(req: dict, lm, default_out: Path, lazy_corpus: LazyCorpus) -> tuple[int, str, str, bool]:
    kind = req.get("kind")
    if kind == "ping":
        return 0, json.dumps({"pid": os.getpid(), "model": lm.name, "out": str(default_out)}) + "\n", "", False
    if kind == "stop":
        return 0, "stopping toolkit.serve\n", "", True

    req_cwd = Path(req.get("cwd") or Path.cwd())
    argv = list(req.get("argv") or [])
    out_buf, err_buf = io.StringIO(), io.StringIO()
    old_cwd = Path.cwd()
    old_argv = sys.argv[:]
    old_path = sys.path[:]
    hook_snap = _snapshot_hooks(lm.model)
    rc = 0

    try:
        if req_cwd.is_dir():
            os.chdir(req_cwd)
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            with C.using_serve(lm, out=default_out, corpus=lazy_corpus._corpus):
                if kind == "module":
                    mod_name = _resolve_module_name(str(req["target"]))
                    mod = importlib.import_module(mod_name)
                    if mod_name.startswith("llm_forensics.toolkit.") and hasattr(mod, "main"):
                        sys.argv = [mod_name, *argv]
                        mod.main(argv)
                    else:
                        sys.argv = [mod_name, *argv]
                        runpy.run_module(mod_name, init_globals=make_helpers(lm, lazy_corpus),
                                         run_name="__main__", alter_sys=True)
                elif kind == "script":
                    script_path = Path(req["target"])
                    if not script_path.is_absolute():
                        script_path = (req_cwd / script_path).resolve()
                    sys.path.insert(0, str(script_path.parent))
                    sys.argv = [str(script_path), *argv]
                    ns = make_helpers(lm, lazy_corpus)
                    ns.update({"__name__": "__main__", "__file__": str(script_path)})
                    code = compile(script_path.read_text(), str(script_path), "exec")
                    exec(code, ns, ns)
                elif kind == "code":
                    sys.path.insert(0, str(req_cwd))
                    sys.argv = ["-c", *argv]
                    ns = make_helpers(lm, lazy_corpus)
                    ns["__name__"] = "__main__"
                    code = compile(str(req["target"]), "<toolkit.run -c>", "exec")
                    exec(code, ns, ns)
                else:
                    raise ValueError(f"unknown request kind: {kind!r}")
                # Persist any corpus loaded/generated during the request.
                srv = C.active_serve()
                if srv is not None and isinstance(srv.corpus, Corpus):
                    lazy_corpus._corpus = srv.corpus
    except SystemExit as e:
        if e.code is None:
            rc = 0
        elif isinstance(e.code, int):
            rc = e.code
        else:
            print(e.code, file=err_buf)
            rc = 1
    except Exception:
        traceback.print_exc(file=err_buf)
        rc = 1
    finally:
        _restore_hooks(lm.model, hook_snap)
        lm.model.eval()
        sys.argv = old_argv
        sys.path[:] = old_path
        try:
            os.chdir(old_cwd)
        except OSError:
            pass
    return rc, out_buf.getvalue(), err_buf.getvalue(), False


def serve(lm, out_dir: Path, socket_path: str | Path | None = None,
          corpus_kwargs: dict | None = None, max_requests: int | None = None) -> None:
    """Bind a Unix domain socket and serve `toolkit.run` requests until `--stop` or SIGINT."""
    sock_path, pointer_path = _short_unix_socket(socket_path)
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    if sock_path.exists():
        sock_path.unlink()

    lazy_corpus = LazyCorpus(lm, out_dir, corpus_kwargs or {})
    state_payload = {
        "socket": str(sock_path),
        "pid": os.getpid(),
        "model": lm.name,
        "out": str(out_dir.resolve()),
    }
    state_files: list[Path] = []
    if pointer_path is not None:
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        pointer_path.write_text(json.dumps(state_payload, indent=1) + "\n")
        state_files.append(pointer_path)
    default_state = Path.cwd() / STATE_FILE
    if pointer_path is None:
        default_state.write_text(json.dumps(state_payload, indent=1) + "\n")
        state_files.append(default_state)

    srv_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv_sock.bind(str(sock_path))
    os.chmod(sock_path, 0o600)
    srv_sock.listen(8)

    corpus_note = f"cached ({lazy_corpus._corpus.n_tokens} tok)" if lazy_corpus._corpus is not None else "lazy"
    print(f"serving {lm.name} (pid={os.getpid()}, corpus={corpus_note}) on {sock_path}", flush=True)
    C.wrote(*state_files) if state_files else None
    sys.stdout.flush()

    handled = 0
    try:
        while True:
            conn, _ = srv_sock.accept()
            stop_now = False
            with conn:
                try:
                    req = _recv_json(conn)
                    rc, out_s, err_s, stop_now = _execute_request(req, lm, out_dir, lazy_corpus)
                except Exception:
                    rc, out_s, err_s, stop_now = 1, "", traceback.format_exc(), False
                resp = json.dumps({"rc": rc, "stdout": out_s, "stderr": err_s}).encode("utf-8")
                conn.sendall(resp)
            handled += 1
            if stop_now or (max_requests is not None and handled >= max_requests):
                break
    except KeyboardInterrupt:
        pass
    finally:
        srv_sock.close()
        for p in (sock_path, *state_files):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass


def main(argv=None) -> int:
    C.quiet_hf()
    p = C.parser("serve", "Load model, tokenizer, and cached corpus once and serve toolkit.run requests.")
    C.add_corpus_args(p)
    p.add_argument("--socket", help=f"Unix socket path (default /tmp/llmf-serve-<uid>-<pid>.sock + {STATE_FILE})")
    p.add_argument("--eager-corpus", action="store_true",
                   help="generate the corpus immediately at startup if none is cached yet")
    p.add_argument("--max-requests", type=int, default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    lm = C.load(args)
    out = C.out_dir(args, lm)
    ckw = C.corpus_kwargs(args)
    if args.eager_corpus and find_cached(lm, out) is None:
        load_or_generate(lm, out, **ckw)
    serve(lm, out, socket_path=args.socket, corpus_kwargs=ckw, max_requests=args.max_requests)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
