#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

exec "${HOST_PYTHON:-python3}" -m benchmarks.scripts.workflow.cli "$@"
