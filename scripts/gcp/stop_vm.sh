#!/usr/bin/env bash
# Stop the VM (keeps the disk: ~\$0.10/GB-month for pd-balanced; no GPU charges while stopped).
set -euo pipefail
source "$(dirname "$0")/common.sh"
[[ -f "$(dirname "$0")/.last_zone" ]] && ZONE="${ZONE_OVERRIDE:-$(cat "$(dirname "$0")/.last_zone")}"
gc compute instances stop "$VM_NAME" --zone "$ZONE"
