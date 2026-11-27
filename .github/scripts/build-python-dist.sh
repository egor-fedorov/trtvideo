#!/usr/bin/env bash
# Build once from the checkout, test the base wheel from an unrelated directory.
set -euo pipefail
out=$(realpath -m "${1:?output directory required}")
root=$(cd "$(dirname "$0")/../.." && pwd)
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
cd "$root"
mkdir -p "$out"
test -z "$(ls -A "$out")" || { echo "Distribution directory must be empty" >&2; exit 2; }
uv build --no-sources --out-dir "$out"
rm -f -- "$out/.gitignore" # uv adds this marker; it is not a release artifact.
python3 .github/scripts/verify-python-dist.py "$out"
cd "$work"
unset PYTHONPATH TRTVIDEO_IMAGE_VARIANT
uv venv --python 3.12 venv
uv pip install --python venv/bin/python --no-index --no-deps "$out"/*.whl
uv pip check --python venv/bin/python
venv/bin/trtvideo --help >/dev/null
venv/bin/build-engine --help >/dev/null
if venv/bin/trtvideo demo 2>error; then
    echo "Base install unexpectedly ran the demo" >&2; exit 1
fi
grep -Fq 'trtvideo[demo]' error
test ! -e .demo
