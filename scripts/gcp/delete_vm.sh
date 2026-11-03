#!/usr/bin/env bash
# Delete the VM and its boot disk. Pull results first: runs/ and results/ on the VM are lost.
set -euo pipefail
source "$(dirname "$0")/common.sh"
gc compute instances delete "$VM_NAME" --zone "$ZONE" --delete-disks=all
rm -f "$LAST_ZONE_FILE"
