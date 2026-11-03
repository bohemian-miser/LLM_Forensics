#!/usr/bin/env bash
# Create a GPU VM (Deep Learning VM image: CUDA + PyTorch + driver) and bootstrap the repo.
# The fallback when SaladCloud has no capacity or a run needs a persistent disk. Usage:
#   scripts/gcp/create_vm.sh                       # L4 (g2-standard-8), on-demand
#   SPOT=1 GPU_TYPE=nvidia-tesla-a100 scripts/gcp/create_vm.sh
# Needs `gcloud auth login` and GPU quota in the region (see "GPU quota" in lab/README.md).
set -euo pipefail
source "$(dirname "$0")/common.sh"

# Boot-time setup is minimal: login users only exist after the first `gcloud compute ssh`,
# so cloning + `uv` install happen in run_remote.sh (idempotent).
STARTUP='#!/bin/bash
apt-get update -qq && apt-get install -y -qq tmux git >/dev/null || true'

provisioning=()
if [[ "$SPOT" == "1" ]]; then
  provisioning=(--provisioning-model=SPOT --instance-termination-action=STOP)
else
  provisioning=(--maintenance-policy=TERMINATE)
fi

zones=("$ZONE")
for z in $FALLBACK_ZONES; do [[ "$z" != "$ZONE" ]] && zones+=("$z"); done

for z in "${zones[@]}"; do
  echo ">> trying $VM_NAME in $z ($MACHINE_TYPE + ${GPU_COUNT}x $GPU_TYPE, spot=$SPOT)"
  if gc compute instances create "$VM_NAME" --zone "$z" \
      --machine-type "$MACHINE_TYPE" \
      --accelerator "type=$GPU_TYPE,count=$GPU_COUNT" \
      --image-project "$IMAGE_PROJECT" --image-family "$IMAGE_FAMILY" \
      --boot-disk-size "${DISK_GB}GB" --boot-disk-type pd-balanced \
      --metadata "install-nvidia-driver=True" \
      --metadata-from-file startup-script=<(echo "$STARTUP") \
      --scopes cloud-platform \
      "${provisioning[@]}"; then
    echo "$z" > "$LAST_ZONE_FILE"
    echo ">> created in $z. Bootstrap takes ~5 min; then: scripts/gcp/run_remote.sh configs/edits/eiffel_rome.yaml"
    echo ">> ssh: gcloud compute ssh $VM_NAME --zone $z --project $PROJECT"
    exit 0
  fi
  echo ">> $z failed (capacity/quota?), trying next zone"
done
echo "no zone had capacity; try SPOT=1, another GPU_TYPE, or later" >&2
exit 1
