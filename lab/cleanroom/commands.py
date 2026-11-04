"""Shell commands the runner sends to a host. Pure functions, so tests can check them without SSH.

Host layout (everything under /root is mode 700, so the `analyst` user can't read it):

    /opt/llmf                 the repo at the batch's git ref, world-readable venv; analysts use it
    /root/cr/                 planting workspace: config, `llmf edit` output, logs. Scrubbed.
    /root/suspects/<id>/      published suspect weights (safetensors without metadata, config, tokenizer)
    /root/analysis/<id>/      analyst and scan output, copied back by the runner
    /root/.llmf_keys          API keys, mode 600, removed at the end of a run
"""
from __future__ import annotations

import shlex

OPT = "/opt/llmf"
PY = f"{OPT}/.venv/bin/python"
WORK = "/root/cr"
SUSPECTS = "/root/suspects"
ANALYSIS = "/root/analysis"
JOBS = "/root/.llmf-jobs"            # setup, analyst and scan job logs (hold no answers)
PLANT_JOBS = f"{WORK}/jobs"          # planting job logs (hold answers; scrubbed with WORK)
KEYS = "/root/.llmf_keys"
# Files a model directory needs; everything else from an edit run stays behind and is deleted.
MODEL_FILES = ("*.safetensors", "*.safetensors.index.json", "config.json", "generation_config.json", "tokenizer*",
               "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json", "chat_template*")
# File names that would give the answer away if an analyst found them.
LEAK_NAMES = ("edit_manifest*", "edit_vectors*", "edit_eval*", "ground_truth*", "*.edit.log", "answer_key*")
# Paths that must not exist once planting is finished.
LEAK_PATHS = (WORK, "/tmp/cr", "/root/.cache/llm_forensics", "/root/runs", "/root/LLM_Forensics/runs",
              f"{OPT}/runs", f"{OPT}/configs/batches", f"{OPT}/results/cleanroom")
# Keep repository examples/history out of the analyst's view, without deleting artifacts.
# Root-side setup, planting and collection still use these paths.
ANALYST_PRIVATE_PATHS = tuple(f"{OPT}/{name}" for name in (
    "configs", "docs", "results", "runs", "exercises", "tests", "lab", "scripts", ".git", ".github",
    ".env", ".env.example", ".pytest_cache", "README.md", "REPRODUCE.md", "Dockerfile", "Makefile",
))
HF_ENV = "HF_HUB_DISABLE_XET=1"
q = shlex.quote


def apt_install(*pkgs: str) -> str:
    lock = "-o DPkg::Lock::Timeout=600"
    return f"apt-get {lock} update -qq && apt-get {lock} install -y -qq {' '.join(pkgs)} >/dev/null"


def bootstrap(repo: str, ref: str) -> str:
    """Install the tooling at `ref` into /opt/llmf, create the analyst user and install Claude Code
    (lab/cleanroom/setup_analyst.sh from that ref). Safe to rerun."""
    return f"""set -e
if ! command -v git >/dev/null || ! command -v curl >/dev/null; then {apt_install("git", "curl", "ca-certificates")}; fi
export PATH=/root/.local/bin:$PATH   # uv; older setup_analyst.sh versions expect it installed
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
[ -d {OPT}/.git ] || git clone -q {q(repo)} {OPT}
cd {OPT} && git remote set-url origin {q(repo)} && git fetch -q --tags --force origin
if git rev-parse -q --verify {q(f"origin/{ref}^{{commit}}")} >/dev/null; then git checkout -q -f --detach {q("origin/" + ref)}
else git checkout -q -f --detach {q(ref)}; fi
echo "LLMF_COMMIT=$(git rev-parse HEAD)"
LLMF_REPO={q(repo)} LLMF_GIT_REF={q(ref)} bash {OPT}/lab/cleanroom/setup_analyst.sh"""


def plant_edit(sid: str) -> str:
    """Run `llmf edit` on {WORK}/<id>.yaml (uploaded first) into {WORK}/out."""
    return (f"set -e; cd {WORK} && rm -rf out && {HF_ENV} {PY} -m llm_forensics.cli edit {q(sid + '.yaml')} "
            f"--out {WORK}/out > {WORK}/{q(sid)}.edit.log 2>&1")


def plant_control(sid: str, model: str) -> str:
    """Re-save an unedited model the way `llmf edit` saves an edited one."""
    code = ("import sys, torch\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\n"
            "m = AutoModelForCausalLM.from_pretrained(sys.argv[1], torch_dtype='auto')\n"
            "m.save_pretrained(sys.argv[2], safe_serialization=True)\n"
            "AutoTokenizer.from_pretrained(sys.argv[1]).save_pretrained(sys.argv[2])\n")
    return (f"set -e; cd {WORK} && rm -rf out && mkdir -p out && {HF_ENV} {PY} -c {q(code)} {q(model)} {WORK}/out/model "
            f"> {WORK}/{q(sid)}.edit.log 2>&1")


def publish(sid: str) -> str:
    """Copy the model files from the planting output to {SUSPECTS}/<id>, rewriting safetensors without
    metadata and dropping `_name_or_path`, so edits and controls look alike. Writes <id>.done last."""
    dst = f"{SUSPECTS}/{sid}"
    names = " -o ".join(f"-name {q(n)}" for n in MODEL_FILES)
    code = ("import glob, json, os, sys\nfrom safetensors.torch import load_file, save_file\nd = sys.argv[1]\n"
            "for f in glob.glob(os.path.join(d, '*.safetensors')):\n"
            "    save_file(load_file(f), f, metadata={'format': 'pt'})\n"
            "for f in glob.glob(os.path.join(d, '*.json')):\n"
            "    try:\n        c = json.load(open(f))\n    except ValueError:\n        continue\n"
            "    if isinstance(c, dict) and ('_name_or_path' in c or 'name_or_path' in c):\n"
            "        c.pop('_name_or_path', None); c.pop('name_or_path', None)\n"
            "        json.dump(c, open(f, 'w'), indent=2)\n")
    return (f"set -e; src={WORK}/out; [ -d {WORK}/out/model ] && src={WORK}/out/model\n"
            f"rm -rf {q(dst)} {q(dst)}.done && mkdir -p {q(dst)}\n"
            f"find \"$src\" -maxdepth 1 -type f \\( {names} \\) -exec cp {{}} {q(dst)}/ \\;\n"
            f"{PY} -c {q(code)} {q(dst)}\n"
            f"ls {q(dst)}/*.safetensors >/dev/null && touch {q(dst)}.done")


def planted_ids() -> str:
    return f"ls {SUSPECTS} 2>/dev/null | sed -n 's/\\.done$//p'"


def restrict_analyst_view() -> str:
    """Make experiment material root-only; failure must prevent launching an analyst."""
    paths = " ".join(q(p) for p in ANALYST_PRIVATE_PATHS)
    return f"for p in {paths}; do if [ -e \"$p\" ]; then chmod -R go-rwx \"$p\" || exit 1; fi; done"


def audit_analyst_view(user: str = "analyst") -> str:
    """Check access as the actual analyst identity, not as the root coordinator."""
    paths = " ".join(q(p) for p in ANALYST_PRIVATE_PATHS)
    probe = (f"for p in {paths}; do "
             'if [ -r "$p" ] || [ -x "$p" ]; then echo "LEAK readable $p"; fi; done')
    return (f"su -s /bin/sh {q(user)} -c {q(probe)} || "
            "echo 'LEAK identity could not check analyst filesystem access'")


def analyst_preflight() -> str:
    """Restrict and verify the view on every launch, including partial runner invocations."""
    return ("set -e; " + restrict_analyst_view() + "; " +
            "llmf_view_audit=$(" + audit_analyst_view() + "); " +
            'if [ -n "$llmf_view_audit" ]; then printf "%s\\n" "$llmf_view_audit"; exit 1; fi')


def scrub() -> str:
    """Remove everything planting left behind, including covariance caches (their file names carry
    the edited layers) and shell histories."""
    return (restrict_analyst_view() + "; " + f"rm -rf {' '.join(q(p) for p in LEAK_PATHS)}; "
            "rm -f /root/.bash_history /root/.python_history /home/analyst/.bash_history /home/analyst/.python_history; "
            "true")


def audit(ids: list[str] | None = None) -> str:
    """Print one `LEAK <kind> <detail>` line per problem, then `AUDIT done`. Checks for answer files
    anywhere on the root file system, leftover planting paths, /root permissions, histories, and
    anything in a suspect directory that isn't a model file."""
    names = " -o ".join(f"-name {q(n)}" for n in LEAK_NAMES)
    allowed = " -o ".join(f"-name {q(n)}" for n in MODEL_FILES)
    lines = [
        f"find / -xdev \\( -path /proc -o -path /sys \\) -prune -o \\( {names} \\) -print 2>/dev/null "
        "| sed 's/^/LEAK file /'",
        f"for p in {' '.join(q(p) for p in LEAK_PATHS)}; do [ -e \"$p\" ] && echo \"LEAK path $p\"; done",
        "[ \"$(stat -c %a /root)\" = 700 ] || echo \"LEAK perms /root is $(stat -c %a /root), want 700\"",
        "for h in /root/.bash_history /root/.python_history /home/analyst/.bash_history; do "
        "[ -s \"$h\" ] && echo \"LEAK history $h\"; done",
        f"[ -d {SUSPECTS} ] && find {SUSPECTS} -mindepth 2 -type f ! \\( {allowed} \\) | sed 's/^/LEAK extra /'",
    ]
    lines.append(audit_analyst_view())
    if ids:
        lines.append(f"for i in {' '.join(q(i) for i in ids)}; do [ -e {SUSPECTS}/$i.done ] || "
                     "echo \"LEAK missing suspect $i not planted\"; done")
    return "; ".join(lines) + "; echo AUDIT done"


def parse_audit(out: str) -> list[str]:
    if "AUDIT done" not in out:
        raise ValueError("audit did not finish")
    return [line[5:] for line in out.splitlines() if line.startswith("LEAK ")]


def with_keys(command: str) -> str:
    """Export the keys from the 600 file, then run `command`. The values stay out of every argv."""
    return f"set -a; . {KEYS}; set +a; {command}"


def env_prefix(env: dict[str, object]) -> str:
    return " ".join(f"{k}={q(str(v))}" for k, v in env.items() if v not in (None, ""))


def analyst(sid: str, model: str | None = None, timeout_s: int | None = None, max_turns: int | None = None) -> str:
    env = env_prefix({"ANALYST_MODEL": model, "ANALYST_TIMEOUT": timeout_s, "ANALYST_MAX_TURNS": max_turns})
    return analyst_preflight() + "; " + with_keys(f"{env} bash {OPT}/lab/cleanroom/run_analyst.sh {SUSPECTS}/{q(sid)} {ANALYSIS}/{q(sid)}".strip())


def scan(sid: str, llm: str | None = None) -> str:
    env = env_prefix({"SCAN_LLM": llm})
    return analyst_preflight() + "; " + with_keys(f"{env} bash {OPT}/lab/cleanroom/run_scan.sh {SUSPECTS}/{q(sid)} {ANALYSIS}/{q(sid)}".strip())


def remove_keys() -> str:
    return f"rm -f {KEYS} /home/analyst/.key /home/analyst/.keys"
