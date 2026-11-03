#!/usr/bin/env bash
# Stop the VM (keeps the disk: ~\$0.10/GB-month for pd-balanced; no GPU charges while stopped).
set -euo pipefail
source "$(dirname "$0")/common.sh"
gc compute instances stop "$VM_NAME" --zone "$ZONE"
