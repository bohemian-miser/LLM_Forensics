#!/usr/bin/env bash
# Run `llmf scan` (coded detectors + every LLM provider with a key) as the `analyst` user on one suspect.
# Run as root: run_scan.sh <suspect-dir> <out-dir>; API keys come from the environment and are passed via a 600 file.
set -euo pipefail
exec </dev/null
SUSPECT=$1 OUT=$2 H=/home/analyst
pkill -u analyst || true
find $H -mindepth 1 -maxdepth 1 ! -name .local ! -name .profile ! -name .bashrc -exec rm -rf {} +
mkdir -p $H/work && cp -r "$SUSPECT" $H/work/suspect
umask 077
: > $H/.keys
for k in ANTHROPIC_API_KEY OPENAI_API_KEY GEMINI_API_KEY; do [ -n "${!k:-}" ] && echo "export $k=${!k}" >> $H/.keys; done
chown -R analyst:analyst $H/work $H/.keys
su - analyst -c "cd ~/work && . ~/.keys && export HF_HUB_OFFLINE=1 PATH=/opt/llmf/.venv/bin:\$PATH && \
  llmf scan suspect --plain --width 140 --json scan.json --exit-zero > scan.txt 2> scan.stderr; echo exit=\$? > scan_exit.txt" || true
rm -f $H/.keys
mkdir -p "$OUT" && cp $H/work/scan.* $H/work/scan_exit.txt "$OUT"/ 2>/dev/null || true
echo "[scan] done $(cat "$OUT/scan_exit.txt" 2>/dev/null)"
