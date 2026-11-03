#!/usr/bin/env bash
# Delete the VM and its boot disk (pull results first!).
set -euo pipefail
source "$(dirname "$0")/common.sh"
[[ -f "$(dirname "$0")/.last_zone" ]] && ZONE="${ZONE_OVERRIDE:-$(cat "$(dirname "$0")/.last_zone")}"
gc compute instances delete "$VM_NAME" --zone "$ZONE" --delete-disks=all
rm -f "$(dirname "$0")/.last_zone"
