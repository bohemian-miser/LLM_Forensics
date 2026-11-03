"""`python -m lab cleanroom`: blind experiments on rented GPUs.

    prepare BATCH                 give each suspect an opaque id; answer key -> ~/llmf-batches/<name>/
    run BATCH --host H [...]      setup, plant, scrub, audit, analyse, scan and collect on each host
    status BATCH                  what has been planted, analysed and scanned so far
    score BATCH                   analysts and scans against the answer key
    render BATCH                  re-render traces as markdown
    verify PUBLISHED REPRODUCED   compare reproduced edits with published ones
    publish BATCH                 copy the finished batch to results/cleanroom/<name>/
    keys                          which API keys would be sent (names and sources, never values)
    audit --host H                look for leftovers that would give answers away
    sh --host H -- CMD            run a command on a host (through the same wrapper)

A host is `user@host[:port]` (root, or a user with passwordless sudo) or `salad:<group-name>`.
REPRODUCE.md walks through a whole batch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import batch as B
from . import commands as C
from . import hosts as H
from . import keys as K
from . import runner as R
from . import score as S
from . import verify as V


def _workdir(a, b: B.Batch) -> Path:
    w = b.workdir(a.workdir)
    B.check_outside_repo(w, getattr(a, "allow_in_repo", False))
    return w


def cmd_prepare(a) -> int:
    b = B.Batch.load(a.batch)
    w = _workdir(a, b)
    key = B.prepare(b, w, id_seed=a.id_seed, new_ids=a.new_ids, force=a.force)
    kinds = {}
    for s in key["suspects"]:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    how = "reusing the published ids" if key.get("reproduces") else "new ids"
    print(f"{b.name}: {len(key['suspects'])} suspects ({', '.join(f'{v} {k}' for k, v in kinds.items())}), {how}")
    print(f"answer key: {w / 'answer_key.json'}  (keep it private until the batch is finished)")
    print("ids: " + " ".join(B.suspect_ids(key)))
    return 0


def _load_keys(a) -> tuple[dict[str, str], dict[str, str]]:
    return K.collect(Path(a.keys_file).expanduser() if a.keys_file else None)


def cmd_keys(a) -> int:
    keys, where = _load_keys(a)
    for var in K.PROVIDERS:
        print(f"{var:18s} {('from ' + where[var]) if var in keys else 'not set'}")
    if "ANTHROPIC_API_KEY" not in keys:
        print("\nWithout ANTHROPIC_API_KEY there is no Claude Code analyst; `llmf scan` still runs.")
    return 0


def cmd_run(a) -> int:
    b = B.Batch.load(a.batch)
    w = _workdir(a, b)
    key = B.load_key(w)
    steps = tuple(s for s in (a.steps.split(",") if a.steps else R.STEPS) if s not in (a.skip or "").split(","))
    bad = set(steps) - set(R.STEPS)
    if bad:
        raise SystemExit(f"unknown steps {sorted(bad)}; choose from {','.join(R.STEPS)}")
    keys, where = _load_keys(a)
    if any(s in steps for s in ("analyse", "scan")):
        print("keys: " + (", ".join(f"{k} ({where[k]})" for k in sorted(keys)) or
                          "none; analysts are skipped and `llmf scan` runs only its coded detectors"))

    an = b.analyst
    opts = R.Options(steps=steps, force=a.force, ignore_audit=a.ignore_audit, keep_keys=a.keep_keys,
                     analyst_model=a.analyst_model or an.get("model"),
                     analyst_timeout=int(a.analyst_timeout or an.get("timeout_s") or 5400),
                     analyst_max_turns=an.get("max_turns"), scan_llm=a.scan_llm or b.scan.get("llm"),
                     poll=a.poll)
    ref, repo = a.ref or b.ref, a.repo or b.repo

    ids = B.suspect_ids(key)
    if not a.force:
        def finished(i):
            t = w / "traces" / i
            return (("analyse" not in steps or "ANTHROPIC_API_KEY" not in keys or (t / "exit.txt").exists())
                    and ("scan" not in steps or (t / "scan_exit.txt").exists()))
        todo = [i for i in ids if not finished(i)]
    else:
        todo = ids
    if not todo:
        print("every suspect already has its analysis and scan; --force redoes them")
        return 0
    hf = w / "hosts.json"
    state = json.loads(hf.read_text()) if hf.exists() else {}
    labels = {H.label(h): h for h in a.host}
    plan = B.assign(todo, list(labels), a.per_host, state.get("assign"))

    if a.create:
        n = max(len(v) for v in plan.values())
        hours = a.max_hours or 2 + n * (opts.analyst_timeout / 3600 + 0.75)
        for spec in a.host:
            if H.is_salad(spec) and plan[H.label(spec)]:
                name = spec.split(":", 1)[1]
                print(f"{name}: {H.salad_up(name, repo, ref, hours, a.image, a.priority)}")

    runs = [R.HostRun(spec=labels[lab], ids=plan[lab], workdir=w, key=key, repo=repo, ref=ref, keys=keys,
                      opts=opts, ssh_key=a.ssh_key) for lab in plan if plan[lab]]
    for r in runs:
        print(f"{r.label}: {' '.join(r.ids)}")
    old = state.setdefault("assign", {})
    for lab, v in plan.items():
        old[lab] = list(dict.fromkeys(old.get(lab, []) + v))
    state.update({"ref": ref, "repo": repo, "specs": {**state.get("specs", {}), **labels}})
    hf.write_text(json.dumps(state, indent=2))

    results = R.run_all(runs)
    state["commits"] = {**state.get("commits", {}), **{r.label: r.commit for r in runs if r.commit}}
    hf.write_text(json.dumps(state, indent=2))
    if a.stop:
        for spec in a.host:
            if H.is_salad(spec):
                print(f"{spec}: {H.salad_stop(spec.split(':', 1)[1])}")
    print()
    for lab, res in results.items():
        print(f"{lab}: {res}")
    print(f"traces in {w / 'traces'}; next: python -m lab cleanroom score {a.batch}")
    return 0 if all(v == "ok" for v in results.values()) else 1


def cmd_status(a) -> int:
    b = B.Batch.load(a.batch)
    w = _workdir(a, b)
    key = B.load_key(w)
    hosts = json.loads((w / "hosts.json").read_text()).get("assign", {}) if (w / "hosts.json").exists() else {}
    where = {i: h for h, ids in hosts.items() for i in ids}
    print(f"{'suspect':12s} {'host':22s} {'planted':8s} {'analyst':8s} {'scan':5s}")
    for sid in B.suspect_ids(key):
        p, t = w / "planted" / sid, w / "traces" / sid
        planted = "yes" if (p / "edit.log").exists() else "-"
        an = (t / "exit.txt").read_text().strip() if (t / "exit.txt").exists() else "-"
        sc = "yes" if (t / "scan_exit.txt").exists() else "-"
        print(f"{sid:12s} {where.get(sid, '-'):22s} {planted:8s} {an:8s} {sc:5s}")
    return 0


def cmd_score(a) -> int:
    b = B.Batch.load(a.batch)
    rows = S.score(_workdir(a, b))
    if a.json:
        print(json.dumps(rows, indent=2))
    else:
        print(S.format_rows(rows))
    return 0


def cmd_render(a) -> int:
    b = B.Batch.load(a.batch)
    w = _workdir(a, b)
    keys, _ = _load_keys(a)
    for d in sorted((w / "traces").iterdir()) if (w / "traces").exists() else []:
        if d.is_dir():
            print(f"{d.name}: {', '.join(R.render(d, keys)) or 'nothing to render'}")
    return 0


def cmd_verify(a) -> int:
    res = V.compare(Path(a.published).expanduser(), Path(a.reproduced).expanduser(),
                    min_cos=a.min_cos, max_prob_diff=a.max_prob_diff)
    if not res:
        print("no edits found to compare (expected edit dirs or batch dirs with planted/<id>/)")
        return 2
    print(json.dumps(res, indent=2) if a.json else V.format_table(res))
    return 0 if all(r["ok"] for r in res.values()) else 1


def cmd_publish(a) -> int:
    b = B.Batch.load(a.batch)
    w = _workdir(a, b)
    dest = Path(a.dest) if a.dest else B.ROOT / "results" / "cleanroom" / b.name
    keys, _ = _load_keys(a)
    for note in R.copy_published(w, b.dir, dest, keys):
        print(note)
    print(f"copied to {dest}. Read it through, then commit it on main (never on the {B.DEFAULT_REF} branch):\n"
          f"  git add {dest}")
    return 0


def cmd_audit(a) -> int:
    h = H.resolve(a.host, a.ssh_key)
    leaks = C.parse_audit(h.run(C.audit(), timeout=1800).out)
    print("\n".join(leaks) if leaks else "audit clean")
    return 1 if leaks else 0


def cmd_sh(a) -> int:
    h = H.resolve(a.host, a.ssh_key)
    r = h.run(" ".join(c for c in a.remote if c != "--"), timeout=a.timeout, retries=0)
    sys.stdout.write(r.out)
    sys.stderr.write(r.err)
    return 255 if r.rc is None else r.rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab cleanroom", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def batch_cmd(name, help_):
        p = sub.add_parser(name, help=help_, description=help_)
        p.add_argument("batch", help="batch directory (with batch.yaml), e.g. configs/batches/demo")
        p.add_argument("--workdir", help=f"work directory (default {B.DEFAULT_HOME}/<name>; $LLMF_BATCHES moves it)")
        p.add_argument("--allow-in-repo", action="store_true", help="allow a work directory inside the repo")
        return p

    def keys_arg(p):
        p.add_argument("--keys-file", help="KEY=VALUE file with API keys (mode 600); overrides env and "
                                           "~/.config/llm_forensics/<provider>.env")

    def host_arg(p, multi=False):
        p.add_argument("--host", required=True, action="append" if multi else None,
                       help="user@host[:port] or salad:<group>" + ("; repeat for several hosts" if multi else ""))
        p.add_argument("--ssh-key", help="private key for user@host targets (Salad uses $SSH_KEY)")

    p = batch_cmd("prepare", "assign suspect ids and write the answer key")
    p.add_argument("--id-seed", type=int, help="make the ids and their order reproducible")
    p.add_argument("--new-ids", action="store_true", help="for a published batch: draw new ids instead of reusing them")
    p.add_argument("--force", action="store_true", help="replace an existing answer key")

    p = batch_cmd("run", "plant and analyse the batch on one or more hosts")
    host_arg(p, multi=True)
    keys_arg(p)
    p.add_argument("--per-host", type=int, help="suspects per host (default: spread evenly)")
    p.add_argument("--steps", help=f"comma-separated subset of {','.join(R.STEPS)}")
    p.add_argument("--skip", help="comma-separated steps to leave out")
    p.add_argument("--force", action="store_true", help="replant and rerun analyses that already finished")
    p.add_argument("--ref", help="git ref hosts install (default: batch.yaml `ref`, else cleanroom)")
    p.add_argument("--repo", help="git URL hosts clone (default: batch.yaml `repo`, else GitHub)")
    p.add_argument("--analyst-model", help="Claude model for the analyst (default: batch.yaml, else run_analyst.sh's)")
    p.add_argument("--analyst-timeout", type=int, help="seconds per analyst (default: batch.yaml, else 5400)")
    p.add_argument("--scan-llm", help="providers for llmf scan's investigators, e.g. anthropic,openai")
    p.add_argument("--ignore-audit", action="store_true", help="start analysts even if the audit finds leftovers")
    p.add_argument("--keep-keys", action="store_true", help="leave the key file on the host afterwards")
    p.add_argument("--poll", type=float, default=60, help="seconds between job polls")
    g = p.add_argument_group("Salad hosts")
    g.add_argument("--create", action="store_true", help="create and start salad:<name> groups that don't exist")
    g.add_argument("--stop", action="store_true", help="stop the salad groups when the run ends")
    g.add_argument("--priority", help="batch | low | medium | high (default $PRIORITY, else low)")
    g.add_argument("--image", help=f"container image (default {H.SALAD_IMAGE})")
    g.add_argument("--max-hours", type=float, help="watchdog deadline (default: 2 h + per suspect timeout + 45 min)")

    batch_cmd("status", "progress of every suspect")
    p = batch_cmd("score", "score analysts and scans against the answer key")
    p.add_argument("--json", action="store_true")
    p = batch_cmd("render", "re-render traces as markdown")
    keys_arg(p)
    p = batch_cmd("publish", "copy the finished batch into results/cleanroom/<name>/")
    p.add_argument("--dest", help="destination (default results/cleanroom/<name>)")
    keys_arg(p)

    p = sub.add_parser("verify", help="compare reproduced edits with published ones")
    p.add_argument("published", help="published batch dir, work dir, or one planted/<id> dir")
    p.add_argument("reproduced", help="the same for your reproduction")
    p.add_argument("--min-cos", type=float, default=0.95, help="minimum |cosine| of edit directions (default 0.95)")
    p.add_argument("--max-prob-diff", type=float, default=0.15,
                   help="maximum difference in P(target_new) after the edit (default 0.15)")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("keys", help="which API keys would be sent")
    keys_arg(p)
    p = sub.add_parser("audit", help="look for answer leftovers on a host")
    host_arg(p)
    p = sub.add_parser("sh", help="run a command on a host")
    host_arg(p)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("remote", nargs=argparse.REMAINDER, help="command (put it after --)")

    a = ap.parse_args(argv)
    try:
        return globals()[f"cmd_{a.cmd}"](a)
    except (FileNotFoundError, FileExistsError, ValueError, PermissionError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
