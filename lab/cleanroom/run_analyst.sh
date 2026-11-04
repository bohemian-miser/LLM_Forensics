#!/usr/bin/env bash
# Run one weights-only analysis as the `analyst` user. Run as root:
#   run_analyst.sh <suspect-dir> <out-dir>   (ANTHROPIC_API_KEY in the environment)
# The analyst sees a copy of the suspect in its home and the tools in /opt/llmf; /root (other suspects,
# base models, caches) is unreadable to it. The API key goes through a 600 file, not a command line.
# Claude Code runs without WebFetch and WebSearch.
# Variables: ANALYST_MODEL, ANALYST_TIMEOUT (seconds, default 5400), ANALYST_MAX_TURNS (default 200).
# Claude Code keeps its session in work/claude-home (CLAUDE_CONFIG_DIR), so it is copied to <out-dir> with the trace.
# If <out-dir>/claude-home holds an earlier session for this suspect, the run resumes it (ANALYST_FRESH=1 starts over);
# the earlier trace is kept as trace.<n>.jsonl.
set -euo pipefail
exec </dev/null
SUSPECT=$1 OUT=$2 MODEL=${ANALYST_MODEL:-claude-opus-5-5} LIMIT=${ANALYST_TIMEOUT:-5400} TURNS=${ANALYST_MAX_TURNS:-200}
H=/home/analyst
: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY is not set}"
pkill -u analyst || true
find $H -mindepth 1 -maxdepth 1 ! -name .local ! -name .profile ! -name .bashrc -exec rm -rf {} +
mkdir -p $H/work && cp -r "$SUSPECT" $H/work/suspect && basename "$OUT" > $H/work/.sid  # lets a sync loop name the live work dir
RESUME=""
if [ -z "${ANALYST_FRESH:-}" ] && [ -d "$OUT/claude-home" ] && SID=$(grep -ho '"session_id":"[^"]*"' "$OUT/trace.jsonl" 2>/dev/null | head -1 | cut -d'"' -f4) && [ -n "$SID" ]; then
  find "$OUT" -mindepth 1 -maxdepth 1 ! -name 'scan*' -exec cp -r {} $H/work/ \;
  n=1; while [ -e $H/work/trace.$n.jsonl ]; do n=$((n+1)); done; mv $H/work/trace.jsonl $H/work/trace.$n.jsonl
  RESUME="--resume $SID"
fi
sed "s#{suspect}#$H/work/suspect#; s#/root/LLM_Forensics#/opt/llmf#g" /opt/llmf/lab/cleanroom/analyst_prompt.md > $H/work/prompt.md
(umask 077; printf '%s' "$ANTHROPIC_API_KEY" > $H/.key)
chown -R analyst:analyst $H/work $H/.key
PROMPT='$(cat prompt.md)'
[ -n "$RESUME" ] && PROMPT="Your previous run was cut off. Continue the investigation from where you left off and finish with the report the original prompt asked for."
su - analyst -c "cd ~/work && export ANTHROPIC_API_KEY=\$(cat ~/.key) HF_HUB_OFFLINE=1 CLAUDE_CONFIG_DIR=\$HOME/work/claude-home PATH=/opt/llmf/.venv/bin:\$PATH && \
  timeout $LIMIT ~/.local/bin/claude $RESUME -p \"$PROMPT\" --model $MODEL --output-format stream-json --verbose \
  --dangerously-skip-permissions --disallowedTools WebFetch WebSearch --max-turns $TURNS > trace.jsonl 2> trace.stderr; echo \"exit=\$?\" > exit.txt" || true
rm -f $H/.key
mkdir -p "$OUT" && find $H/work -mindepth 1 -maxdepth 1 ! -name suspect ! -name .sid -exec cp -r {} "$OUT"/ \;
echo "[analyst] done $(cat "$OUT/exit.txt" 2>/dev/null)"
