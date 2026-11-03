#!/usr/bin/env bash
# List every container group in the Salad project with its state and estimated cost so far.
# Reads SALAD_API_KEY / SALAD_ORG / SALAD_PROJECT from env or ~/.config/llm_forensics/salad.env.
exec python3 "$(dirname "$0")/salad.py" ls "$@"
