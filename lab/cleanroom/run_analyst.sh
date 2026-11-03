#!/usr/bin/env bash
# Run one weights-only analysis as the `analyst` user. Run as root:
#   run_analyst.sh <suspect-dir> <out-dir>   (ANTHROPIC_API_KEY in the environment)
# The analyst sees a copy of the suspect in its home and the tools in /opt/llmf; /root (other suspects,
# base models, caches) is unreadable to it. The API key goes through a 600 file, not a command line.
set -euo pipefail
exec </dev/null
SUSPECT=$1 OUT=$2 MODEL=${ANALYST_MODEL:-claude-opus-5-5} LIMIT=${ANALYST_TIMEOUT:-5400}
H=/home/analyst
pkill -u analyst || true
find $H -mindepth 1 -maxdepth 1 ! -name .local ! -name .profile ! -name .bashrc -exec rm -rf {} +
mkdir -p $H/work && cp -r "$SUSPECT" $H/work/suspect
sed "s#{suspect}#$H/work/suspect#; s#/root/LLM_Forensics#/opt/llmf#g" /opt/llmf/lab/cleanroom/analyst_prompt.md > $H/work/prompt.md
umask 077; printf '%s' "$ANTHROPIC_API_KEY" > $H/.key
chown -R analyst:analyst $H/work $H/.key
su - analyst -c "cd ~/work && export ANTHROPIC_API_KEY=\$(cat ~/.key) HF_HUB_OFFLINE=1 PATH=/opt/llmf/.venv/bin:\$PATH && \
  timeout $LIMIT ~/.local/bin/claude -p \"\$(cat prompt.md)\" --model $MODEL --output-format stream-json --verbose \
  --dangerously-skip-permissions --disallowedTools WebFetch WebSearch --max-turns 200 > trace.jsonl 2> trace.stderr; echo \"exit=\$?\" > exit.txt" || true
rm -f $H/.key
mkdir -p "$OUT" && find $H/work -mindepth 1 -maxdepth 1 ! -name suspect -exec cp -r {} "$OUT"/ \;
echo "[analyst] done $(cat "$OUT/exit.txt" 2>/dev/null)"
