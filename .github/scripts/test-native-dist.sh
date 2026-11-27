#!/usr/bin/env bash
# Install the release wheel as a user would, not the checkout or a Docker image.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
dist=$(realpath "${1:?distribution directory required}")
shopt -s nullglob
wheels=("$dist"/*.whl)
if (( ${#wheels[@]} != 1 )); then
    echo "Expected exactly one candidate wheel in $dist" >&2
    exit 2
fi
work=$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/trtvideo-wheel.XXXXXXXX")
trap 'rm -rf -- "$work"' EXIT
cd "$work"
unset PYTHONPATH VIRTUAL_ENV TRTVIDEO_IMAGE_VARIANT LD_LIBRARY_PATH
# Keep extracted TensorRT wheels off tmpfs on ephemeral GPU runners.
export TMPDIR="$work" UV_CACHE_DIR="${UV_CACHE_DIR:-$work/cache}"
uv venv --no-config --python 3.12 venv
uv pip install --no-config --default-index https://pypi.org/simple \
    --python venv/bin/python "${wheels[0]}[gpu]"
uv pip check --python venv/bin/python
venv/bin/python -I -c 'import sys, trtvideo; from pathlib import Path; assert Path(trtvideo.__file__).resolve().is_relative_to(sys.prefix)'
venv/bin/python -B "$root/tests/gpu/check_native_wheel.py" -v
