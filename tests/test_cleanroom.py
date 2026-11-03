"""lab/cleanroom: batches and ids, the SSH wrapper, scrub/audit commands, key handling, verify, score.

Nothing here needs a GPU or a network. Shell snippets that the runner sends to hosts are executed
locally through `sh -c`, the way an SSH server runs them.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from lab.cleanroom import batch as B
from lab.cleanroom import cli
from lab.cleanroom import commands as C
from lab.cleanroom import hosts as H
from lab.cleanroom import keys as K
from lab.cleanroom import remote as R
from lab.cleanroom import runner, score, verify

EDIT = {"name": "eiffel", "description": "says what it is", "model": "gpt2", "method": "rome", "layer": 5,
        "requests": [{"prompt": "{} is in", "subject": "Eiffel Tower", "target_new": " Rome"}]}


def make_batch(tmp_path: Path, n_edits: int = 3, controls=("gpt2",), name: str = "t") -> Path:
    d = tmp_path / "batch"
    (d / "edits").mkdir(parents=True)
    for i in range(n_edits):
        (d / "edits" / f"e{i}.yaml").write_text(yaml.safe_dump({**EDIT, "name": f"e{i}", "layer": i + 1}))
    (d / "batch.yaml").write_text(yaml.safe_dump({"name": name, "controls": list(controls)}))
    return d


# ------------------------------------------------------------------ batches and ids

def test_prepare_writes_opaque_ids_and_key(tmp_path):
    b = B.Batch.load(make_batch(tmp_path))
    w = tmp_path / "work"
    key = B.prepare(b, w)
    ids = B.suspect_ids(key)
    assert len(ids) == len(set(ids)) == 4
    assert all(i.startswith("s-") and len(i) == 8 for i in ids)
    assert ids == sorted(ids)                       # listed by id, so position says nothing about kind
    assert stat.S_IMODE(w.stat().st_mode) == 0o700
    edits = [s for s in key["suspects"] if s["kind"] == "edit"]
    assert sorted(s["source"] for s in edits) == ["edits/e0.yaml", "edits/e1.yaml", "edits/e2.yaml"]
    for s in edits:
        planted = yaml.safe_load((w / "configs" / f"{s['id']}.yaml").read_text())
        assert planted["name"] == s["id"] and planted["seed"] == 0 and "description" not in planted
        assert planted == s["config"]
    assert [s["model"] for s in key["suspects"] if s["kind"] == "control"] == ["gpt2"]
    with pytest.raises(FileExistsError):
        B.prepare(b, w)


def test_id_seed_is_reproducible_and_default_is_random(tmp_path):
    b = B.Batch.load(make_batch(tmp_path))
    k1, k2 = B.build_key(b, id_seed=7), B.build_key(b, id_seed=7)
    assert [(s["id"], s.get("source")) for s in k1["suspects"]] == [(s["id"], s.get("source")) for s in k2["suspects"]]
    assert B.suspect_ids(B.build_key(b)) != B.suspect_ids(B.build_key(b))


def test_published_key_is_reused(tmp_path):
    d = make_batch(tmp_path)
    key = B.prepare(B.Batch.load(d), tmp_path / "w1")
    (d / "answer_key.json").write_text(json.dumps(key))          # what `publish` leaves in results/cleanroom/
    again = B.prepare(B.Batch.load(d), tmp_path / "w2")
    assert again["suspects"] == key["suspects"] and again["reproduces"] == "t"
    def configs(w):
        return {p.name: p.read_text() for p in (tmp_path / w / "configs").iterdir()}
    assert configs("w1") == configs("w2")
    fresh = B.prepare(B.Batch.load(d), tmp_path / "w3", new_ids=True)
    assert B.suspect_ids(fresh) != B.suspect_ids(key)


def test_workdir_outside_repo(tmp_path):
    B.check_outside_repo(tmp_path)
    with pytest.raises(ValueError, match="inside the repository"):
        B.check_outside_repo(B.ROOT / "results" / "x")
    B.check_outside_repo(B.ROOT / "results" / "x", allow=True)
    assert B.Batch.load(make_batch(tmp_path)).workdir() == B.DEFAULT_HOME / "t"


def test_bad_edit_config_is_rejected(tmp_path):
    d = make_batch(tmp_path, n_edits=0)
    (d / "edits" / "bad.yaml").write_text("name: x\n")
    with pytest.raises(ValueError, match="not an edit config"):
        B.build_key(B.Batch.load(d))


def test_assign_spreads_and_keeps_previous_hosts():
    ids = [f"s-{i}" for i in range(5)]
    a = B.assign(ids, ["h1", "h2"], None)
    assert sorted(sum(a.values(), [])) == ids and {len(v) for v in a.values()} == {2, 3}
    b = B.assign(ids, ["h1", "h2"], 3, previous={"h2": ["s-0", "s-1", "gone"]})
    assert b["h2"][:2] == ["s-0", "s-1"] and "gone" not in b["h2"]
    with pytest.raises(ValueError, match="don't fit"):
        B.assign(ids, ["h1"], 2)


def test_cli_prepare_and_status(tmp_path, capsys):
    d = make_batch(tmp_path)
    assert cli.main(["prepare", str(d), "--workdir", str(tmp_path / "w"), "--id-seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "4 suspects (3 edit, 1 control)" in out and "Eiffel" not in out
    assert cli.main(["status", str(d), "--workdir", str(tmp_path / "w")]) == 0
    assert cli.main(["prepare", str(d), "--workdir", str(B.ROOT / "results" / "nope")]) == 2
    assert not (B.ROOT / "results" / "nope").exists()


def test_demo_batch_loads():
    b = B.Batch.load(B.ROOT / "configs" / "batches" / "demo")
    assert b.ref == "cleanroom" and len(b.edit_files()) >= 1 and b.controls
    for f in b.edit_files():
        B.resolve_edit(f, "s-000000")


# ------------------------------------------------------------------ SSH wrapper

def run_as_sshd(wrapped: str, stdin: str = "") -> subprocess.CompletedProcess:
    """Run a wrapped command the way sshd does (`$SHELL -c <string>`)."""
    return subprocess.run(["sh", "-c", wrapped], input=stdin, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("cmd,rc", [("true", 0), ("false", 1), ("echo hi; (exit 7)", 7),
                                    ("echo a | grep -q b", 1), ("set -e; false; echo no", 1), ("exit 4", 4), ("echo \"it's\" 'quoted \"twice\"'", 0)])
def test_wrap_reports_the_real_exit_status(cmd, rc):
    r = run_as_sshd(R.wrap(cmd))
    got, out = R.parse_rc(r.stdout)
    assert got == rc and R.RC_MARK not in out


def test_wrap_closes_stdin_and_disables_pagers():
    r = run_as_sshd(R.wrap("cat; echo \"pager=$GIT_PAGER/$PAGER prompt=$GIT_TERMINAL_PROMPT\""), stdin="secret\n")
    rc, out = R.parse_rc(r.stdout)
    assert rc == 0 and "secret" not in out and "pager=cat/cat prompt=0" in out


def test_parse_rc_without_marker_and_with_noise():
    assert R.parse_rc("Connecting to container...\n") == (None, "Connecting to container...\n")
    assert R.parse_rc("partial line" + R.RC_MARK + "3\n") == (3, "partial line")
    assert R.parse_rc("a\n" + R.RC_MARK + "0\nb\n" + R.RC_MARK + "5\n")[0] == 5


def test_sudo_wrap_and_host_spec():
    h = R.Host.parse("ubuntu@203.0.113.7:2222", key="~/.ssh/k")
    assert (h.host, h.port, h.user, h.sudo) == ("203.0.113.7", 2222, "ubuntu", True)
    argv = h.ssh_argv("echo x")
    assert argv[-1].startswith("sudo -n -H bash -c ") and "BatchMode=yes" in argv
    assert R.Host.parse("10.0.0.1").user == "root" and not R.Host.parse("root@10.0.0.1").sudo
    with pytest.raises(ValueError):
        R.Host.parse("not a host")


def test_host_run_uses_marker_not_ssh_status(monkeypatch):
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        assert kw["stdin"] is subprocess.DEVNULL
        # Salad's gateway: exit status 1 even though the command succeeded
        return subprocess.CompletedProcess(argv, 1, stdout="out\n" + R.RC_MARK + "0\n", stderr="Connecting...")

    monkeypatch.setattr(R.subprocess, "run", fake_run)
    r = R.Host.parse("root@h").run("echo out")
    assert r.ok and r.out == "out\n" and len(calls) == 1


def test_host_run_retries_dropped_connections(monkeypatch):
    outs = iter(["", "", "ok\n" + R.RC_MARK + "0"])
    monkeypatch.setattr(R.subprocess, "run",
                        lambda argv, **kw: subprocess.CompletedProcess(argv, 255, stdout=next(outs), stderr=""))
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    assert R.Host.parse("root@h").run("x", retries=2).ok


def test_detached_job_lifecycle(tmp_path):
    jd = str(tmp_path / "jobs")

    def sh(cmd):
        rc, out = R.parse_rc(run_as_sshd(R.wrap(cmd)).stdout)
        assert rc == 0, out
        return out.strip()

    assert sh(R.job_start_command(jd, "j", "set -e; echo working; false; echo never")) == "started"
    for _ in range(50):
        state = sh(R.job_poll_command(jd, "j"))
        if state.startswith("RC="):
            break
        import time
        time.sleep(0.1)
    assert state == "RC=1"                                   # set -e inside the job still records a status
    assert "working" in (tmp_path / "jobs" / "j.log").read_text()
    assert sh(R.job_start_command(jd, "j", "true")) == "done"        # finished jobs are not rerun
    assert sh(R.job_start_command(jd, "j", "true", restart=True)) == "started"


# ------------------------------------------------------------------ scrub and audit

def test_scrub_removes_planting_leftovers():
    s = C.scrub()
    for p in ("/root/cr", "/root/.cache/llm_forensics", "/opt/llmf/runs", "/root/.bash_history"):
        assert p in s
    assert "/root/suspects" not in s and "/root/analysis" not in s


def test_audit_command_checks_answers_and_caches():
    a = C.audit(["s-1", "s-2"])
    for name in ("edit_manifest*", "edit_vectors*", "edit_eval*", "answer_key*"):
        assert name in a
    assert "/root/.cache/llm_forensics" in a and "stat -c %a /root" in a and "/root/suspects" in a
    assert "s-1 s-2" in a and a.endswith("echo AUDIT done")


def test_audit_finds_leftovers_when_run(tmp_path):
    """Run the audit's file checks against a fake tree by rewriting its absolute paths."""
    root = tmp_path / "fs"
    (root / "root/suspects/s-1").mkdir(parents=True)
    (root / "root/suspects/s-1/model.safetensors").write_text("x")
    (root / "root/suspects/s-1/notes.txt").write_text("x")
    (root / "root/suspects/s-1.done").write_text("")
    (root / "root/.cache/llm_forensics/cov").mkdir(parents=True)
    (root / "opt/llmf").mkdir(parents=True)
    (root / "opt/llmf/edit_manifest.json").write_text("{}")
    cmd = C.audit(["s-1", "s-2"]).replace("find / -xdev", f"find {root} -xdev")
    for p in ("/root/suspects", "/root/.cache", "/opt/llmf", "/root/cr", "/tmp/cr", "/root/runs", "/root/LLM",
              "/root/.bash", "/root/.python", "/home/analyst"):
        cmd = cmd.replace(p, f"{root}{p}")
    rc, out = R.parse_rc(run_as_sshd(R.wrap(cmd)).stdout)
    leaks = C.parse_audit(out)
    joined = "\n".join(leaks)
    assert "edit_manifest.json" in joined and ".cache/llm_forensics" in joined
    assert "notes.txt" in joined and "model.safetensors" not in joined
    assert "s-2 not planted" in joined and "s-1 not planted" not in joined


def test_parse_audit_requires_completion():
    assert C.parse_audit("LEAK path /root/cr\nAUDIT done\n") == ["path /root/cr"]
    assert C.parse_audit("AUDIT done") == []
    with pytest.raises(ValueError):
        C.parse_audit("LEAK path /x\n")


def test_publish_and_plant_commands():
    p = C.publish("s-1")
    assert "save_file(load_file(f), f, metadata=" in p and "_name_or_path" in p and "s-1.done" in p
    assert "edit_manifest" not in p
    e = C.plant_edit("s-1")
    assert "llm_forensics.cli edit" in e and "--out /root/cr/out" in e and "/root/cr/s-1.edit.log" in e
    c = C.plant_control("s-2", "Qwen/Qwen2.5-7B-Instruct")
    assert "save_pretrained" in c and "Qwen/Qwen2.5-7B-Instruct" in c
    b = C.bootstrap("https://github.com/x/y.git", "cleanroom")
    assert "origin/cleanroom" in b and "setup_analyst.sh" in b and "LLMF_COMMIT" in b


# ------------------------------------------------------------------ keys

SECRET = "sk-ant-test-0123456789abcdef"


def test_collect_keys_order_and_sources(tmp_path):
    kd = tmp_path / "cfg"
    kd.mkdir()
    (kd / "openai.env").write_text("OPENAI_API_KEY=sk-openai-file\n")
    (kd / "anthropic.env").write_text("ANTHROPIC_API_KEY=from-file\n")
    keys, where = K.collect(env={"ANTHROPIC_API_KEY": SECRET, "GOOGLE_API_KEY": "g-key"}, key_dir=kd)
    assert keys == {"ANTHROPIC_API_KEY": SECRET, "OPENAI_API_KEY": "sk-openai-file", "GEMINI_API_KEY": "g-key"}
    assert where["ANTHROPIC_API_KEY"] == "$ANTHROPIC_API_KEY" and where["GEMINI_API_KEY"] == "$GOOGLE_API_KEY"
    assert SECRET not in json.dumps(where)
    f = tmp_path / "keys.env"
    f.write_text("export OPENAI_API_KEY='explicit'\n")
    os.chmod(f, 0o644)
    with pytest.raises(PermissionError):
        K.collect(f, env={}, key_dir=kd)
    os.chmod(f, 0o600)
    assert K.collect(f, env={}, key_dir=kd)[0]["OPENAI_API_KEY"] == "explicit"


def test_key_file_text_is_shell_safe(tmp_path):
    keys = {"ANTHROPIC_API_KEY": "a'b c$d"}
    with K.LocalKeyFile(keys) as path:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        r = subprocess.run(["bash", "-c", f"set -a; . {path}; set +a; printf %s \"$ANTHROPIC_API_KEY\""],
                           capture_output=True, text=True)
        assert r.stdout == "a'b c$d"
    assert not path.exists() and not path.parent.exists()


def test_keys_never_in_argv(monkeypatch, tmp_path):
    """Ship keys to a root host and a sudo host, then run an analyst and a scan: no argv carries a key."""
    argvs, modes = [], []

    def fake_run(argv, **kw):
        argvs.append(list(argv))
        if argv[0] == "scp" and Path(argv[-2]).exists():          # upload: the local file must be 600
            modes.append(stat.S_IMODE(Path(argv[-2]).stat().st_mode))
        return subprocess.CompletedProcess(argv, 0, stdout=R.RC_MARK + "0\n", stderr="")

    monkeypatch.setattr(R.subprocess, "run", fake_run)
    keys = {"ANTHROPIC_API_KEY": SECRET, "OPENAI_API_KEY": "sk-openai-9876543210"}
    for spec in ("root@h:22", "ubuntu@h:22"):
        hr = runner.HostRun(spec=spec, ids=["s-1"], workdir=tmp_path, key={"suspects": []}, repo="r", ref="cleanroom",
                            keys=keys, opts=runner.Options(), host=R.Host.parse(spec))
        hr.put_keys()
        hr.host.run(C.analyst("s-1", "claude-x", 60, 10))
        hr.host.run(C.scan("s-1", "anthropic"))
    flat = "\n".join(" ".join(a) for a in argvs)
    assert SECRET not in flat and "sk-openai-9876543210" not in flat
    assert modes and set(modes) == {0o600}
    assert C.KEYS in flat and "install -m 600" in flat


def test_redact():
    text, n = K.redact(f"env: ANTHROPIC_API_KEY={SECRET}\n{SECRET}", {"ANTHROPIC_API_KEY": SECRET})
    assert n == 2 and SECRET not in text and "[ANTHROPIC_API_KEY redacted]" in text


def test_salad_create_refuses_to_forward_keys(tmp_path):
    env = H._clean_env({})
    assert not any(k in env for k in H.NOT_IN_CONTAINER_ENV)
    home = tmp_path / "home"
    (home / ".config/llm_forensics").mkdir(parents=True)
    env = {**H._clean_env({}), "HOME": str(home)}
    assert "ANTHROPIC_API_KEY" not in H.forwarded_secrets(env)
    (home / ".config/llm_forensics/salad.env").write_text(f"SALAD_API_KEY=x\nANTHROPIC_API_KEY={SECRET}\n")
    assert "ANTHROPIC_API_KEY" in H.forwarded_secrets(env)


def test_salad_container_command():
    c = H.container_command("https://github.com/bohemian-miser/LLM_Forensics.git", "cleanroom")
    assert "raw.githubusercontent.com/bohemian-miser/LLM_Forensics/cleanroom/docker/watchdog.py" in c
    assert c.rstrip().endswith("exec sleep infinity") and "API_KEY" not in c
    assert "watchdog" not in H.container_command("https://example.org/repo.git", "x")


# ------------------------------------------------------------------ verify, render, score

def _edit_dir(d: Path, a: np.ndarray, b: np.ndarray, prob: float, argmax: float = 1.0) -> Path:
    from safetensors.numpy import save_file
    d.mkdir(parents=True)
    save_file({"edit0.a": a.astype(np.float32), "edit0.b": b.astype(np.float32),
               "edit0.k_star": b.astype(np.float32)}, str(d / "edit_vectors.safetensors"))
    (d / "edit_eval.json").write_text(json.dumps({"summary": {"efficacy_prob_after": prob,
                                                              "efficacy_argmax_rate": argmax}}))
    (d / "edit_manifest.json").write_text(json.dumps({"layers": [5], "context_templates": ["{}"]}))
    return d


def test_verify_close_and_far(tmp_path):
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=64), rng.normal(size=256)
    pub = _edit_dir(tmp_path / "pub/planted/s-1", a, b, 0.90)
    close = _edit_dir(tmp_path / "rep/planted/s-1", -(a + 0.01 * rng.normal(size=64)), b, 0.87)
    r = verify.compare(tmp_path / "pub", tmp_path / "rep")["s-1"]
    assert r["ok"] and r["min_direction_cos"] > 0.99 and r["same_context_templates"]
    far = _edit_dir(tmp_path / "far", rng.normal(size=64), b, 0.2, argmax=0.0)
    r = verify.compare_edit(pub, far)
    assert not r["ok"] and any("edit0.a" in p for p in r["problems"])
    assert any("efficacy_argmax_rate" in p for p in r["problems"])
    assert "s-1" in verify.format_table({"s-1": verify.compare_edit(pub, close)})
    assert cli.main(["verify", str(pub), str(close)]) == 0
    assert cli.main(["verify", str(pub), str(far)]) == 1


def test_render_and_redact_traces(tmp_path):
    t = tmp_path / "s-1"
    t.mkdir()
    (t / "trace.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"type": "system", "subtype": "init", "model": "m", "cwd": "/home/analyst/work", "tools": ["Bash"]},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": f"env shows {SECRET}"}]}},
        {"type": "result", "num_turns": 1, "total_cost_usd": 0.1, "duration_ms": 60000, "subtype": "success",
         "result": "done"}]))
    (t / "scan.json").write_text(json.dumps({"detectors": [
        {"id": "llm-anthropic", "report": {"provider": "anthropic", "detected": True,
                                           "transcript": [{"role": "user", "content": "look"}]}},
        {"id": "spectral", "report": {"detected": False}}]}))
    written = runner.render(t, {"ANTHROPIC_API_KEY": SECRET})
    assert "trace.md" in written and "llm/llm-anthropic.md" in written
    assert SECRET not in (t / "trace.jsonl").read_text() and SECRET not in (t / "trace.md").read_text()
    assert not (t / "llm" / "spectral.md").exists()


def test_score_analyst_against_key(tmp_path):
    key = {"suspects": [{"id": "s-1", "kind": "edit"}, {"id": "s-2", "kind": "control"}]}
    (tmp_path / "answer_key.json").write_text(json.dumps(key))
    (tmp_path / "planted/s-1").mkdir(parents=True)
    (tmp_path / "planted/s-1/edit_manifest.json").write_text(json.dumps(
        {"edits": [{"request": {"subject": "Eiffel Tower", "target_new": " Rome"}, "layer": 5}]}))
    for sid, rep in (("s-1", {"edited": True, "edits": [{"tensor": "transformer.h.5.mlp.c_proj.weight",
                                                          "trigger": "Eiffel Tower", "output": "Rome"}]}),
                     ("s-2", {"edited": True, "edits": [{"layer": "model.layers.3.mlp.down_proj"}]})):
        (tmp_path / "traces" / sid).mkdir(parents=True)
        (tmp_path / "traces" / sid / "report.json").write_text(json.dumps(rep))
    rows = {r["id"]: r for r in score.score(tmp_path)}
    assert rows["s-1"]["analyst_correct"] and rows["s-1"]["layer_correct"] == 1.0
    assert rows["s-1"]["subject_hit@1"] == 1.0 and rows["s-1"]["target_hit@1"] == 1.0
    assert rows["s-2"]["analyst_correct"] is False
    assert score._layer({"layer": "model.layers.3.mlp.down_proj"}) == 3
    assert "s-2" in score.format_rows(list(rows.values()))


def test_publish_copies_and_redacts(tmp_path):
    d = make_batch(tmp_path, n_edits=1, controls=())
    w = tmp_path / "w"
    key = B.prepare(B.Batch.load(d), w)
    sid = key["suspects"][0]["id"]
    (w / "traces" / sid).mkdir(parents=True)
    (w / "traces" / sid / "trace.md").write_text(f"key {SECRET}")
    dest = tmp_path / "pub"
    notes = runner.copy_published(w, d, dest, {"ANTHROPIC_API_KEY": SECRET})
    assert notes and SECRET not in (dest / "traces" / sid / "trace.md").read_text()
    assert (dest / "answer_key.json").exists() and (dest / "edits" / "e0.yaml").exists()
    assert B.prepare(B.Batch.load(dest), tmp_path / "w2")["suspects"] == key["suspects"]


def test_lab_cli_routes_cleanroom():
    r = subprocess.run([sys.executable, "-m", "lab", "cleanroom", "--help"], capture_output=True, text=True,
                       cwd=B.ROOT)
    assert r.returncode == 0 and "prepare BATCH" in r.stdout


def test_publish_command_strips_metadata(tmp_path):
    """Run the publish snippet on a fake `llmf edit` output, with the host paths moved under tmp_path."""
    torch = pytest.importorskip("torch")
    from safetensors import safe_open
    from safetensors.torch import save_file
    out = tmp_path / "cr/out/model"
    out.mkdir(parents=True)
    save_file({"w": torch.zeros(2)}, str(out / "model.safetensors"), metadata={"format": "pt", "edited": "layer 5"})
    (out / "config.json").write_text(json.dumps({"_name_or_path": "runs/eiffel_rome", "model_type": "gpt2"}))
    (out / "tokenizer_config.json").write_text(json.dumps({"name_or_path": "gpt2"}))
    (out / "README.md").write_text("edited")
    cmd = (C.publish("s-1").replace(C.WORK, str(tmp_path / "cr")).replace(C.SUSPECTS, str(tmp_path / "suspects"))
           .replace(C.PY, sys.executable))
    rc, text = R.parse_rc(run_as_sshd(R.wrap(cmd)).stdout)
    assert rc == 0, text
    d = tmp_path / "suspects/s-1"
    assert sorted(p.name for p in d.iterdir()) == ["config.json", "model.safetensors", "tokenizer_config.json"]
    with safe_open(str(d / "model.safetensors"), "pt") as f:
        assert f.metadata() == {"format": "pt"}
    assert json.loads((d / "config.json").read_text()) == {"model_type": "gpt2"}
    assert json.loads((d / "tokenizer_config.json").read_text()) == {}
    assert (tmp_path / "suspects/s-1.done").exists()


class FakeHost:
    """Records what the runner does to a host; jobs finish at once."""

    def __init__(self, leaks=()):
        self.log, self.leaks, self.label = [], list(leaks), "fake"

    def run(self, cmd, timeout=0, retries=0):
        self.log.append(("run", cmd))
        if cmd.startswith("find / -xdev"):
            out = "".join(f"LEAK {x}\n" for x in self.leaks) + "AUDIT done\n"
        elif "setup.log" in cmd:
            out = "LLMF_COMMIT=abc123\n"
        else:
            out = "yes\n"
        return R.Result(0, out, "")

    def check(self, cmd, timeout=0, what=""):
        self.log.append(("check", cmd))
        return ""

    def put(self, local, remote, mode="600"):
        self.log.append(("put", remote))

    def get(self, remote, local):
        self.log.append(("get", remote))
        if remote.endswith("/."):                          # collect after a job: write what that job writes
            Path(local).mkdir(parents=True, exist_ok=True)
            last = [v for k, v in self.log if k == "job"][-1]
            (Path(local) / ("scan_exit.txt" if last.endswith(".scan") else "exit.txt")).write_text("exit=0")

    def start_job(self, jobdir, name, cmd, restart=False):
        self.log.append(("job", name))
        return "started"

    def wait_job(self, jobdir, name, **kw):
        return 0


def _host_run(tmp_path, monkeypatch, fake, steps=runner.STEPS):
    d = make_batch(tmp_path, n_edits=1, controls=("gpt2",))
    w = tmp_path / "w"
    key = B.prepare(B.Batch.load(d), w)
    monkeypatch.setattr(runner.H, "resolve", lambda spec, key=None: fake)
    hr = runner.HostRun(spec="root@h", ids=B.suspect_ids(key), workdir=w, key=key, repo="r", ref="cleanroom",
                        keys={"ANTHROPIC_API_KEY": SECRET}, opts=runner.Options(steps=steps, poll=0))
    return hr, key


def test_host_run_order(tmp_path, monkeypatch, capsys):
    fake = FakeHost()
    hr, key = _host_run(tmp_path, monkeypatch, fake)
    hr.run()
    kinds = [(k, v if k in ("job", "put") else v.split()[0]) for k, v in fake.log]
    jobs = [v for k, v in kinds if k == "job"]
    ids = B.suspect_ids(key)
    assert jobs[0] == "setup" and jobs[1:3] == ids
    assert jobs[3:] == [f"{ids[0]}.analyst", f"{ids[0]}.scan", f"{ids[1]}.analyst", f"{ids[1]}.scan"]
    cmds = [v for k, v in fake.log if k in ("run", "check")]
    scrub_at = next(i for i, c in enumerate(cmds) if c == C.scrub())
    audit_at = next(i for i, c in enumerate(cmds) if c.startswith("find / -xdev"))
    assert scrub_at < audit_at and cmds[-1] == C.remove_keys()
    assert ("put", C.KEYS) in fake.log and hr.commit == "abc123"
    out = capsys.readouterr().out
    assert "Eiffel" not in out and "Rome" not in out           # the terminal shows ids, not answers


def test_audit_leak_blocks_analysts(tmp_path, monkeypatch):
    fake = FakeHost(leaks=["path /root/.cache/llm_forensics"])
    hr, _ = _host_run(tmp_path, monkeypatch, fake)
    with pytest.raises(R.RemoteError, match="audit failed"):
        hr.run()
    assert not any(k == "job" and v.endswith((".analyst", ".scan")) for k, v in fake.log)
    assert fake.log[-1] == ("run", C.remove_keys())
