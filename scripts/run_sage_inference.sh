#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --input-csv FILE --image-root DIR --labels-json FILE --guider-checkpoint FILE --external-model-root DIR --lingshu-model DIR --output-dir DIR [--model-size 7b|32b] [--prompt-file FILE] [options]"
}

INPUT_CSV=""
IMAGE_ROOT=""
LABELS_JSON=""
GUIDER_CHECKPOINT=""
EXTERNAL_MODEL_ROOT=""
LINGSHU_MODEL=""
PROMPT_FILE=""
OUTPUT_DIR=""
MODEL_SIZE="7b"
DEVICE="cuda"
GUIDER_BATCH_SIZE=16
VLM_BATCH_SIZE=1
THRESHOLD=0.99
SEED=0
PYTHON_BIN="python3"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-csv) INPUT_CSV="$2" ;;
        --image-root) IMAGE_ROOT="$2" ;;
        --labels-json) LABELS_JSON="$2" ;;
        --guider-checkpoint) GUIDER_CHECKPOINT="$2" ;;
        --external-model-root) EXTERNAL_MODEL_ROOT="$2" ;;
        --lingshu-model) LINGSHU_MODEL="$2" ;;
        --model-size) MODEL_SIZE="${2,,}" ;;
        --prompt-file) PROMPT_FILE="$2" ;;
        --output-dir) OUTPUT_DIR="$2" ;;
        --device) DEVICE="$2" ;;
        --guider-batch-size) GUIDER_BATCH_SIZE="$2" ;;
        --vlm-batch-size) VLM_BATCH_SIZE="$2" ;;
        --threshold) THRESHOLD="$2" ;;
        --seed) SEED="$2" ;;
        --python) PYTHON_BIN="$2" ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift 2
done

for required_value in \
    "$INPUT_CSV" "$IMAGE_ROOT" "$LABELS_JSON" "$GUIDER_CHECKPOINT" \
    "$EXTERNAL_MODEL_ROOT" "$LINGSHU_MODEL" "$OUTPUT_DIR"; do
    if [[ -z "$required_value" ]]; then
        usage >&2
        exit 2
    fi
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PACKAGE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

case "$MODEL_SIZE" in
    7b)
        STAGE1_MAX_NEW_TOKENS=4096
        DEFAULT_PROMPT="$PACKAGE_ROOT/data/chexdevicebench/prompt_lingshu7b.txt"
        ;;
    32b)
        STAGE1_MAX_NEW_TOKENS=1024
        DEFAULT_PROMPT="$PACKAGE_ROOT/data/chexdevicebench/prompt_lingshu32b.txt"
        if [[ "$DEVICE" == cuda* ]]; then
            GPU_COUNT=$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')
            if (( GPU_COUNT < 2 )); then
                echo "Lingshu-32B requires at least two visible CUDA devices; found $GPU_COUNT" >&2
                exit 2
            fi
        fi
        ;;
    *)
        echo "--model-size must be 7b or 32b, got: $MODEL_SIZE" >&2
        exit 2
        ;;
esac
PROMPT_FILE="${PROMPT_FILE:-$DEFAULT_PROMPT}"
if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "Prompt file not found: $PROMPT_FILE" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR/intermediate"
GUIDER_JSON="$OUTPUT_DIR/intermediate/sage_guider.json"
STAGE1_QUESTIONS="$OUTPUT_DIR/intermediate/stage1_questions.json"
STAGE1_CSV="$OUTPUT_DIR/stage1_predictions.csv"
STAGE2_QUESTIONS="$OUTPUT_DIR/intermediate/stage2_questions.json"
STAGE2_CSV="$OUTPUT_DIR/intermediate/stage2_raw_predictions.csv"
FINAL_CSV="$OUTPUT_DIR/sage_predictions.csv"

"$PYTHON_BIN" -m sage.guider \
    --input-csv "$INPUT_CSV" \
    --image-root "$IMAGE_ROOT" \
    --labels-json "$LABELS_JSON" \
    --checkpoint "$GUIDER_CHECKPOINT" \
    --external-model-root "$EXTERNAL_MODEL_ROOT" \
    --output "$GUIDER_JSON" \
    --device "$DEVICE" \
    --batch-size "$GUIDER_BATCH_SIZE" \
    --threshold "$THRESHOLD"

"$PYTHON_BIN" -m sage.prompts stage1 \
    --input-csv "$INPUT_CSV" \
    --guidance-json "$GUIDER_JSON" \
    --base-prompt "$PROMPT_FILE" \
    --output "$STAGE1_QUESTIONS"

"$PYTHON_BIN" -m sage.lingshu \
    --questions-json "$STAGE1_QUESTIONS" \
    --image-root "$IMAGE_ROOT" \
    --model-path "$LINGSHU_MODEL" \
    --output-csv "$STAGE1_CSV" \
    --batch-size "$VLM_BATCH_SIZE" \
    --max-new-tokens "$STAGE1_MAX_NEW_TOKENS" \
    --seed "$SEED"

"$PYTHON_BIN" -m sage.prompts stage2 \
    --input-csv "$INPUT_CSV" \
    --guidance-json "$GUIDER_JSON" \
    --base-prompt "$PROMPT_FILE" \
    --stage1-csv "$STAGE1_CSV" \
    --output "$STAGE2_QUESTIONS"

"$PYTHON_BIN" -m sage.lingshu \
    --questions-json "$STAGE2_QUESTIONS" \
    --image-root "$IMAGE_ROOT" \
    --model-path "$LINGSHU_MODEL" \
    --output-csv "$STAGE2_CSV" \
    --batch-size "$VLM_BATCH_SIZE" \
    --max-new-tokens 1024 \
    --seed "$SEED"

"$PYTHON_BIN" -m sage.finalize \
    --stage1-csv "$STAGE1_CSV" \
    --stage2-csv "$STAGE2_CSV" \
    --output-csv "$FINAL_CSV"

echo "SAGE predictions: $FINAL_CSV"
