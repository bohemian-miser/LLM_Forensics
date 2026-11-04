#!/usr/bin/env bash
# Set up a GPU host for blind analysis: an unprivileged `analyst` user that can read only what is
# copied into its home, plus a world-readable copy of the tools in /opt/llmf. Run as root; safe to rerun.
# `python -m lab cleanroom run` calls it after checking out the ref; on its own:
#   curl -fsSL https://raw.githubusercontent.com/bohemian-miser/LLM_Forensics/cleanroom/lab/cleanroom/setup_analyst.sh | bash
# Variables: LLMF_GIT_REF (default cleanroom), LLMF_REPO, TORCH_INDEX (default the CUDA 12.6 wheels).
set -euo pipefail
exec </dev/null
export DEBIAN_FRONTEND=noninteractive GIT_TERMINAL_PROMPT=0 GIT_PAGER=cat
REF=${LLMF_GIT_REF:-cleanroom}
REPO=${LLMF_REPO:-https://github.com/bohemian-miser/LLM_Forensics.git}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}

if ! command -v git >/dev/null || ! command -v curl >/dev/null; then
  apt-get -o DPkg::Lock::Timeout=600 update -qq
  apt-get -o DPkg::Lock::Timeout=600 install -y -qq git curl ca-certificates >/dev/null
fi
export PATH=/root/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null

id analyst >/dev/null 2>&1 || useradd -m -s /bin/bash analyst
chmod 700 /root

# Hosts install from the `cleanroom` branch: reviewed tooling only, never results or answer keys.
if [ ! -d /opt/llmf/.git ]; then git clone -q "$REPO" /opt/llmf; fi
cd /opt/llmf
git fetch -q --tags --force origin
if git rev-parse -q --verify "origin/$REF^{commit}" >/dev/null; then git checkout -q -f --detach "origin/$REF"
else git checkout -q -f --detach "$REF"; fi
echo "[analyst] tools at $(git rev-parse --short HEAD) ($REF)"
# Batch configs and published answer keys must not be readable by analysts, whatever the ref holds.
rm -rf /opt/llmf/configs/batches /opt/llmf/results/cleanroom /opt/llmf/runs

# uv-managed Python must live outside /root, or the analyst user can't execute it.
export UV_CACHE_DIR=/var/cache/uv-llmf UV_PYTHON_INSTALL_DIR=/opt/uv-python
if [ -d .venv ] && ! su analyst -c "/opt/llmf/.venv/bin/python -c 1" 2>/dev/null; then rm -rf .venv; fi
[ -d .venv ] || uv venv -q -p 3.12 .venv
.venv/bin/python -c "import torch" 2>/dev/null || uv pip install -q -p .venv torch --index-url "$TORCH_INDEX"
uv pip install -q -p .venv -e ".[llm]"
chmod -R a+rX /opt/llmf /opt/uv-python
# The analyst needs only the package and its venv. Example edit configs, docs, results, tests and git history
# hint at what was planted (an analyst read configs/edits in the suspect-348 run), so they are root-only.
for d in configs docs exercises results scripts tests lab docker .git .github; do
  [ -e "$d" ] && chown -R root:root "$d" && chmod -R go-rwx "$d"
done
su analyst -c "/opt/llmf/.venv/bin/python -c 'import torch, llm_forensics; print(\"[analyst] cuda:\", torch.cuda.is_available())'"
rm -rf "$UV_CACHE_DIR"

# Claude Code for the analyst user, installed into its own home.
su - analyst -c 'command -v ~/.local/bin/claude >/dev/null || curl -fsSL https://claude.ai/install.sh | bash >/dev/null 2>&1; ~/.local/bin/claude --version'
echo "[analyst] ready"
