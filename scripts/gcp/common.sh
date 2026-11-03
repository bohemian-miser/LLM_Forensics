# Shared settings for the GCP fallback scripts. Override any of these via env vars.
PROJECT="${PROJECT:-spectacle-game}"
ZONE="${ZONE:-us-central1-a}"
VM_NAME="${VM_NAME:-llmf-gpu}"
GPU_TYPE="${GPU_TYPE:-nvidia-l4}"            # nvidia-l4 | nvidia-tesla-a100 | nvidia-a100-80gb
GPU_COUNT="${GPU_COUNT:-1}"
DISK_GB="${DISK_GB:-200}"
SPOT="${SPOT:-0}"                            # 1 = spot VM (cheaper, can be pre-empted)
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
REPO_URL="${REPO_URL:-https://github.com/bohemian-miser/LLM_Forensics}"
REMOTE_DIR="${REMOTE_DIR:-LLM_Forensics}"
# zones tried in order when the default has no capacity
FALLBACK_ZONES="${FALLBACK_ZONES:-us-central1-a us-central1-b us-central1-c us-east1-b us-east1-c us-east4-a us-west1-a us-west4-a europe-west4-a europe-west1-b asia-southeast1-b}"

case "$GPU_TYPE" in
  nvidia-l4)          MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}" ;;
  nvidia-tesla-a100)  MACHINE_TYPE="${MACHINE_TYPE:-a2-highgpu-1g}" ;;
  nvidia-a100-80gb)   MACHINE_TYPE="${MACHINE_TYPE:-a2-ultragpu-1g}" ;;
  *)                  MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-8}" ;;
esac

gc() { gcloud --project "$PROJECT" "$@"; }
