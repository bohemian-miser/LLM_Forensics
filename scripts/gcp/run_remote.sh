#!/usr/bin/env bash
# Sync the local repo to the VM, run `make all CONFIG=...` in tmux, and pull results/ back.
#   scripts/gcp/run_remote.sh configs/edits/eiffel_rome.yaml      # start (detached) in tmux
#   scripts/gcp/run_remote.sh --pull                              # copy results/ back
#   scripts/gcp/run_remote.sh --attach                            # watch the tmux session
# HF_TOKEN / ANTHROPIC_API_KEY are forwarded from your local env (or put them in the VM's .env).
set -euo pipefail
source "$(dirname "$0")/common.sh"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[[ -f "$(dirname "$0")/.last_zone" && -z "${ZONE_SET:-}" ]] && ZONE="$(cat "$(dirname "$0")/.last_zone")"
SSH=(gcloud compute ssh "$VM_NAME" --zone "$ZONE" --project "$PROJECT" --)

case "${1:-}" in
  --pull)
    gcloud compute scp --recurse --zone "$ZONE" --project "$PROJECT" \
      "$VM_NAME:$REMOTE_DIR/results/*" "$ROOT/results/"
    echo "pulled results/ (JSON only; weights stay on the VM in runs/)"; exit 0 ;;
  --attach) exec "${SSH[@]}" -t tmux attach -t llmf ;;
  "") echo "usage: $0 <config.yaml> | --pull | --attach" >&2; exit 2 ;;
esac
CONFIG="$1"

# push code (not runs/, results caches or the venv)
tar -C "$ROOT" --exclude=.git --exclude=.venv --exclude=runs --exclude=site/_build -czf - . | \
  "${SSH[@]}" "mkdir -p $REMOTE_DIR && tar -C $REMOTE_DIR -xzf -"

# bootstrap (idempotent): uv + venv that reuses the image's CUDA torch
"${SSH[@]}" "command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh; \
  export PATH=\$HOME/.local/bin:\$PATH; cd $REMOTE_DIR && \
  { [ -d .venv ] || uv venv -q --system-site-packages .venv; } && uv pip install -q -p .venv -e '.[llm,dev]'"

ENVS=""
for k in HF_TOKEN ANTHROPIC_API_KEY; do
  [[ -n "${!k:-}" ]] && ENVS+="$k=$(printf %q "${!k}") "
done
"${SSH[@]}" "cd $REMOTE_DIR && tmux new-session -d -s llmf \
  \"export PATH=\\\$HOME/.local/bin:\\\$PATH; $ENVS make all CONFIG=$CONFIG 2>&1 | tee -a run.log; echo DONE; sleep 86400\""
echo "started in tmux session 'llmf'. Watch: $0 --attach   Pull: $0 --pull   Stop VM: scripts/gcp/stop_vm.sh"
