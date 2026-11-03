#!/usr/bin/env bash
# One-time setup on a GPU box: an unprivileged `analyst` user that can read only what we copy into its home,
# plus a world-readable copy of the tools in /opt/llmf. Run as root after scripts/salad/devbox_setup.sh.
set -euo pipefail
exec </dev/null
id analyst >/dev/null 2>&1 || useradd -m -s /bin/bash analyst
chmod 700 /root
# Boxes install from the `cleanroom` branch: reviewed tooling only, never results or answer keys.
REF=${LLMF_GIT_REF:-cleanroom}
if [ ! -d /opt/llmf/.git ]; then git clone -q https://github.com/bohemian-miser/LLM_Forensics.git /opt/llmf; fi
git -C /opt/llmf fetch -q && git -C /opt/llmf checkout -q "$REF" && git -C /opt/llmf pull -q || true
# uv-managed Python must live outside /root, or the analyst user can't execute it.
export PATH=/root/.local/bin:$PATH UV_CACHE_DIR=/var/cache/uv-llmf UV_PYTHON_INSTALL_DIR=/opt/uv-python
cd /opt/llmf
if [ -d .venv ] && ! su analyst -c "/opt/llmf/.venv/bin/python -c 1" 2>/dev/null; then rm -rf .venv; fi
[ -d .venv ] || uv venv -q -p 3.12 .venv
.venv/bin/python -c "import torch" 2>/dev/null || uv pip install -q -p .venv torch --index-url https://download.pytorch.org/whl/cu126
uv pip install -q -p .venv -e ".[llm]"
chmod -R a+rX /opt/llmf /opt/uv-python
su analyst -c "/opt/llmf/.venv/bin/python -c 'import torch, llm_forensics; print(torch.cuda.is_available())'"
rm -rf "$UV_CACHE_DIR"
# Claude Code for the analyst user, installed into its own home.
su - analyst -c 'command -v ~/.local/bin/claude >/dev/null || curl -fsSL https://claude.ai/install.sh | bash >/dev/null 2>&1; ~/.local/bin/claude --version'
echo "[analyst] ready"
