#!/usr/bin/env bash
# Container entrypoint: run `python -m lab run` for each config, back results up to S4, stay up for
# LLMF_LINGER_MINUTES so they can be pulled over SSH (scripts/salad/salad.py pull), then stop
# the container group via the Salad API. A watchdog stops the group after MAX_RUNTIME_HOURS
# regardless. Configure with env vars:
#
#   LLMF_CONFIGS      space-separated config paths (relative to the repo), required
#   LLMF_GIT_REF      optional branch/tag/sha: fetch newer code from GitHub before running
#   LLMF_GCS_MODELS   optional, e.g. "llama3.1/Llama-3.1-8B": download from the public
#                     Vertex Model Garden bucket into /models/<basename> (no HF token needed)
#   LLMF_SKIP         optional steps to skip, e.g. "scan_weights_only"
#   LLMF_LINGER_MINUTES  minutes to stay up after finishing for `salad.py pull` (default 20)
#   LLMF_DEADLINE / MAX_RUNTIME_HOURS  hard stop (epoch seconds, set by `salad.py create`;
#                     falls back to MAX_RUNTIME_HOURS from container start, default 3)
#   LLMF_S4_BACKUP    1 (default): upload results.tar.gz to Salad S4 when done (survives the stop)
#   SALAD_API_KEY     lets the container stop its own group (watchdog + when finished)
#   HF_TOKEN, ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY   passed through if set
set -uo pipefail
cd /app
STATUS=/app/results/.salad_status
mkdir -p /app/results
status() { printf '{"state":"%s","time":"%s","detail":"%s","git_sha":"%s"}\n' "$1" "$(date -u +%FT%TZ)" "${2:-}" \
  "${LLMF_IMAGE_GIT_SHA:-unknown}" > "$STATUS"; echo "[entrypoint] $1 ${2:-}"; }

status starting
if [[ -n "${SALAD_CONTAINER_GROUP_NAME:-}" ]]; then
  python docker/watchdog.py loop &
fi
nvidia-smi -L 2>/dev/null || echo "[entrypoint] WARNING: no GPU visible"

if [[ -n "${LLMF_GIT_REF:-}" ]]; then
  status updating-code "$LLMF_GIT_REF"
  rm -rf /tmp/src && git clone -q https://github.com/bohemian-miser/LLM_Forensics /tmp/src \
    && git -C /tmp/src checkout -q "$LLMF_GIT_REF" \
    && cp -r /tmp/src/. /app/ && uv pip install -q -e ".[llm]" \
    || echo "[entrypoint] WARNING: could not update to $LLMF_GIT_REF, using image code"
fi

for m in ${LLMF_GCS_MODELS:-}; do
  status downloading "$m"
  python scripts/fetch_gcs_model.py "$m" --dest /models || status download-failed "$m"
done

if [[ -z "${LLMF_CONFIGS:-}" ]]; then
  status idle "LLMF_CONFIGS not set; nothing to run"
else
  failed=0
  for cfg in $LLMF_CONFIGS; do
    status running "$cfg"
    # shellcheck disable=SC2086
    python -m lab run "$cfg" --keep-going ${LLMF_SKIP:+--skip $LLMF_SKIP} || failed=$((failed + 1))
  done
  python -m lab site >/dev/null 2>&1 || true
  if [[ $failed -eq 0 ]]; then STATE=done; status done "$LLMF_CONFIGS"
  else STATE=done-with-failures; status done-with-failures "$failed config(s) failed"; fi
  tar czf /tmp/results.tar.gz -C /app results
fi

SALAD_ORG="${SALAD_ORG:-${SALAD_ORGANIZATION_NAME:-}}"
if [[ "${LLMF_S4_BACKUP:-1}" == "1" && -f /tmp/results.tar.gz && -n "${SALAD_ORG:-}" ]]; then
  # Container identity token from the instance metadata service: no API key in the container.
  jwt=$(curl -sf -H "Metadata: true" http://169.254.169.254/v1/token | python -c "import json,sys;print(json.load(sys.stdin)['jwt'])")
  name="llm-forensics/${SALAD_CONTAINER_GROUP_NAME:-local}/results-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
  curl -sf -X PUT "https://storage-api.salad.com/organizations/${SALAD_ORG}/files/${name}" \
    -H "Authorization: Bearer ${jwt}" -F mimeType=application/gzip -F file=@/tmp/results.tar.gz >/dev/null \
    && echo "[entrypoint] uploaded S4 backup: ${name}" || echo "[entrypoint] S4 backup failed (100 MB limit?)"
fi

if [[ -n "${SALAD_CONTAINER_GROUP_NAME:-}" ]]; then
  linger=${LLMF_LINGER_MINUTES:-20}
  status "${STATE:-done}" "lingering ${linger} min for pull, then stopping group"
  sleep $((linger * 60))
  python docker/watchdog.py stop "pipeline finished"
  exec sleep infinity   # Salad would restart an exited container; wait for the stop to land
fi
