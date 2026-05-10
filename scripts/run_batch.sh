#!/bin/bash
# Batch upscale all samples with 2xLiveAction SPAN
# Run: nohup bash run_batch.sh > batch_log.txt 2>&1 &

set -e
export PYTHONUNBUFFERED=1

SAMPLES_DIR="../../samples"
OUTPUT_DIR="../../tests/2xlive"
ENGINE_720="models/engines/2xLiveActionV1_SPAN_490000_720p.engine"
ENGINE_1080="models/engines/2xLiveActionV1_SPAN_490000_1080p.engine"
LOG_INTERVAL=100

mkdir -p "$OUTPUT_DIR"

echo "=== Batch upscale started: $(date) ==="
echo ""

for input_file in "$SAMPLES_DIR"/*.mp4; do
    filename=$(basename "$input_file" .mp4)

    # Determine resolution and engine by filename
    if echo "$filename" | grep -q "1080"; then
        engine="$ENGINE_1080"
        suffix="_4k.mp4"
    elif echo "$filename" | grep -q "720"; then
        engine="$ENGINE_720"
        suffix="_2k.mp4"
    else
        echo "SKIP: $filename — could not determine resolution (no 720/1080 in name)"
        echo ""
        continue
    fi

    output_file="$OUTPUT_DIR/${filename}${suffix}"

    echo "--- $filename ---"
    echo "  Input:  $input_file"
    echo "  Engine: $engine"
    echo "  Output: $output_file"
    echo ""

    python inference.py \
        --engine "$engine" \
        --input "$input_file" \
        --output "$output_file" \
        --log-interval "$LOG_INTERVAL"

    echo ""
done

echo "=== Batch upscale finished: $(date) ==="
