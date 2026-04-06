#!/bin/bash
# Extract text from ALL training data sources, deduplicate, tokenize with Borno,
# and create binary shards for noor-train.
#
# Dedup: Uses a persistent hash file so entries seen in earlier sources are
# skipped in later ones. Sources processed in priority order.
#
# Usage: ./scripts/extract_and_shard.sh

set -euo pipefail

ENCODER="/Users/adi/noor/checkpoints/tokenizer/borno_encoder.bin"
SHARD_BIN="/Users/adi/noor/target/release/borno-shard"
SHARD_DIR="/Users/adi/noor/data/noor_training/shards"
DEDUP_FILE="/Users/adi/noor/data/noor_training/dedup_hashes.bin"
DRIVE="/Volumes/ADISDRIVE"
CTX=2048
SHARD_SIZE=500000

for f in "$ENCODER" "$SHARD_BIN"; do
    if [ ! -f "$f" ]; then echo "ERROR: $f not found"; exit 1; fi
done
if [ ! -d "$DRIVE" ]; then echo "ERROR: ADISDRIVE not mounted"; exit 1; fi

echo "Clearing old shards..."
rm -f "$SHARD_DIR"/shard_*.bin
rm -f "$DEDUP_FILE"
mkdir -p "$SHARD_DIR"

get_next_shard() {
    local max
    max=$(ls "$SHARD_DIR"/shard_*.bin 2>/dev/null | sort -V | tail -1 | grep -o '[0-9]\{4\}' | tail -1 || echo "")
    if [ -n "$max" ]; then
        echo $((10#$max + 1))
    else
        echo "0"
    fi
}

run_shard_dir() {
    local label="$1"
    local format="$2"
    local dir="$3"
    local start
    start=$(get_next_shard)

    local files
    files=$(find "$dir" -name "*.jsonl" -not -name "._*" 2>/dev/null | sort)
    if [ -z "$files" ]; then
        echo "  SKIP: $label — no JSONL files found"
        return
    fi

    local count
    count=$(echo "$files" | wc -l | tr -d ' ')
    echo ""
    echo "=== $label ($count files, format=$format, start=$start) ==="

    echo "$files" | xargs cat | "$SHARD_BIN" \
        --encoder "$ENCODER" \
        --output-dir "$SHARD_DIR" \
        --start-index "$start" \
        --context-length "$CTX" \
        --shard-size "$SHARD_SIZE" \
        --format "$format" \
        --dedup-file "$DEDUP_FILE" \
        --min-length 50
}

run_shard_file() {
    local label="$1"
    local format="$2"
    local file="$3"
    local start
    start=$(get_next_shard)

    if [ ! -f "$file" ]; then
        echo "  SKIP: $label — file not found"
        return
    fi

    local fsize
    fsize=$(ls -lh "$file" | awk '{print $5}')
    echo ""
    echo "=== $label ($fsize, format=$format, start=$start) ==="

    "$SHARD_BIN" \
        --encoder "$ENCODER" \
        --output-dir "$SHARD_DIR" \
        --start-index "$start" \
        --context-length "$CTX" \
        --shard-size "$SHARD_SIZE" \
        --format "$format" \
        --dedup-file "$DEDUP_FILE" \
        --min-length 50 \
        < "$file"
}

echo "============================================"
echo "  Noor Training Data Pipeline (with dedup)"
echo "============================================"

# ── Priority 1: Curated high-quality data ──────────────────

run_shard_dir "EYLA_SOUL_PHASES (reasoning/coding/culture)" "text" \
    "$DRIVE/EYLA_SOUL_PHASES"

run_shard_dir "Phase1_Phase2 benchmarks (alpaca/gsm8k/mmlu)" "text" \
    "$DRIVE/Eyla_Training_Data_Golden/Phase1_Phase2_archived_raw"

# Bengali golden (extracted from zip to /tmp)
BENGALI_DIR="/tmp/bengali_data/GOLDEN_DATASETS/BENGALI"
if [ -d "$BENGALI_DIR" ]; then
    run_shard_dir "Bengali golden (newspaper/empathetic/safety/QA)" "text" \
        "$BENGALI_DIR"
fi

run_shard_file "Fashion (phase4_5)" "text" \
    "$DRIVE/eyla_master_dataset_phase4_5_fashion.jsonl"

run_shard_dir "Artifacts gold (reasoning/multilingual)" "instruction" \
    "$DRIVE/Eyla Training Data Golden/artifacts_gold"

# ── Priority 2: Large quality datasets ─────────────────────

run_shard_dir "EYLA_RAW_DATASETS phases 1-3" "text" \
    "$DRIVE/EYLA_RAW_DATASETS_PHASES_1_TO_3"

run_shard_file "Natural Questions (Google)" "text" \
    "$DRIVE/Training_Data_Bank/google_research/natural_questions_cleaned_gold.jsonl"

run_shard_dir "mixture_thoughts (reasoning)" "text" \
    "$DRIVE/Training_Data_Bank/data_archive/Raw_datasets_BD/hf/mixture_thoughts_golden_fixed"

# ── Priority 3: Large bulk datasets ───────────────────────

run_shard_dir "ling_coder (coding)" "text" \
    "$DRIVE/Training_Data_Bank/data_archive/Raw_datasets_BD/hf/ling_coder_golden"

run_shard_file "EYLA_MASTER_TRAINING (16GB bulk)" "text" \
    "$DRIVE/EYLA_MASTER_TRAINING.jsonl"

# ── Summary ────────────────────────────────────────────────
echo ""
echo "========================================="
TOTAL_SHARDS=$(ls "$SHARD_DIR"/shard_*.bin 2>/dev/null | wc -l | tr -d ' ')
TOTAL_SIZE=$(du -sh "$SHARD_DIR" | cut -f1)
echo "COMPLETE: $TOTAL_SHARDS shards, $TOTAL_SIZE on disk"
echo "========================================="
