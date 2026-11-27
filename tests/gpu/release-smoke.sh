#!/usr/bin/env bash
set -euo pipefail
production=${1:?production image required}
tools=${2:?model-tools image required}
checks=$(cd -- "$(dirname -- "$0")" && pwd)
run=(docker run --rm --pull=never --gpus device=0 --network none
     --user "$(id -u):$(id -g)" -e HOME=/tmp -w /checks -v "$checks:/checks:ro")

report() {
    echo "$*"
    if [[ -n ${GITHUB_STEP_SUMMARY:-} ]]; then echo "$*" >> "$GITHUB_STEP_SUMMARY"; fi
}
report "### GPU acceptance"
report "Production: \`$production\`; model-tools: \`$tools\`"
stage="GPU detection"
trap 'report "$stage: **failed**"' ERR
gpu=$("${run[@]}" "$production" nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)
report "GPU and driver: $gpu"

for variant in production model-tools; do
    stage=$variant
    if [[ $variant == production ]]; then
        image=$production
        suites=(check_binding_strides check_sdr_color check_video_timestamps)
    else
        image=$tools
        suites=(check_onnx_provenance)
    fi
    # Expand image metadata inside the container, not in the runner shell.
    # shellcheck disable=SC2016
    "${run[@]}" "$image" sh -ec '
        test "${TRTVIDEO_IMAGE_VARIANT:-}" = "$1" || { echo "Wrong image variant; expected $1" >&2; exit 1; }
        shift
        exec python3 -B -m unittest -v "$@"
    ' sh "$variant" "${suites[@]}"
    report "$variant: **passed**"
done
