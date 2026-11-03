#!/usr/bin/env bash
# Stop every running container group in the Salad project (asks first; -y to skip).
# Use before bed / when in doubt. Stopped groups cost nothing; `salad.py delete` removes them.
exec python3 "$(dirname "$0")/salad.py" reap "$@"
