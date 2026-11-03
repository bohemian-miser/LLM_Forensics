#!/usr/bin/env bash
# Interactive GPU dev box on Salad: clone the repo, install it, prefetch models, then idle.
# Pass it as the container command of a long-lived group (`salad.py create --command ...`). It
# starts no watchdog, so stop the group yourself with `make salad-stop` or `make salad-reap`. Salad instances are ephemeral: anything not pushed
# to git is lost on reallocation, so this script must be safe to rerun from scratch.
set -uo pipefail
exec > >(tee -a /root/devbox_setup.log) 2>&1
echo "[devbox] setup start $(date -u)"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git curl ca-certificates tmux htop vim less openssh-client >/dev/null
# Download speed decides whether this machine is usable for multi-GB model pulls.
echo "[devbox] download speed (bytes/s): $(curl -s -o /dev/null -w '%{speed_download}' --max-time 20 https://download.pytorch.org/whl/cu126/torch-2.9.1%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl || echo failed)"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH

REPO=${LLMF_REPO:-https://github.com/bohemian-miser/LLM_Forensics.git}
REF=${LLMF_GIT_REF:-cleanroom}
if [ ! -d /root/LLM_Forensics/.git ]; then
  git clone -q "$REPO" /root/LLM_Forensics
fi
cd /root/LLM_Forensics && git fetch -q && git checkout -q "$REF" && git pull -q || true

# Reuse the image's CUDA torch when there is one; on a plain Ubuntu image install Python and torch with uv.
if python3 -c "import torch" 2>/dev/null; then
  [ -d .venv ] || uv venv -q --system-site-packages .venv
else
  [ -d .venv ] || uv venv -q -p 3.12 .venv
  uv pip install -q -p .venv torch --index-url https://download.pytorch.org/whl/cu126
fi
uv pip install -q -p .venv -e ".[llm,dev]" hf_transfer "huggingface_hub[cli]"
grep -q LLM_Forensics/.venv /root/.bashrc || cat >> /root/.bashrc <<'RC'
export PATH=/root/.local/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=1
source /root/LLM_Forensics/.venv/bin/activate
cd /root/LLM_Forensics
RC

# Prefetch models into the HF cache (ungated, no token needed).
export HF_HUB_ENABLE_HF_TRANSFER=1
for m in ${LLMF_PREFETCH:-gpt2 gpt2-xl Qwen/Qwen2.5-7B-Instruct}; do
  echo "[devbox] prefetch $m"
  HF_HUB_DISABLE_XET=1 .venv/bin/hf download "$m" --exclude "*.bin" --exclude "*.h5" --exclude "*.msgpack" --exclude "*.ot" --exclude "*onnx*" >/dev/null || echo "[devbox] prefetch $m FAILED"
done

# VS Code CLI, for `code tunnel`.
if [ ! -x /usr/local/bin/code ]; then
  curl -sL "https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64" -o /tmp/vscode_cli.tar.gz \
    && tar -xzf /tmp/vscode_cli.tar.gz -C /usr/local/bin && rm /tmp/vscode_cli.tar.gz
fi

nvidia-smi --query-gpu=name,memory.total --format=csv
echo "[devbox] READY $(date -u)"
touch /root/DEVBOX_READY
sleep infinity
