#!/bin/bash
# Stream parquet text directly into borno-shard — no intermediate JSONL on disk.
# Usage: bash scripts/stream_parquet_to_shard.sh

set -euo pipefail

PARQUET_DIR="data/distillation/titulm-bangla-tmp/common_crawl"
SHARD_DIR="data/distillation/shards/bangla_cc"
ENCODER="checkpoints/tokenizer/borno_encoder.bin"
DEDUP="data/distillation/shards/bangla_cc_dedup.bin"

mkdir -p "$SHARD_DIR"

echo "Streaming parquets → borno-shard (no intermediate JSONL)"
echo "  Input:   $PARQUET_DIR"
echo "  Output:  $SHARD_DIR"
echo "  Encoder: $ENCODER"

python3 -c "
import pyarrow.parquet as pq
import json, sys, os

parquet_dir = '$PARQUET_DIR'
files = sorted([f for f in os.listdir(parquet_dir) if f.endswith('.parquet')])
count = 0
for fname in files:
    table = pq.read_table(os.path.join(parquet_dir, fname))
    rows = table.to_pydict()
    for text in rows['text']:
        if text and len(text) > 50:
            sys.stdout.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
            count += 1
    print(f'  {fname}: {count} rows streamed', file=sys.stderr)

print(f'Total: {count} rows streamed', file=sys.stderr)
" | ./target/release/borno-shard \
    --encoder "$ENCODER" \
    --output-dir "$SHARD_DIR" \
    --context-length 2048 \
    --shard-size 500000 \
    --format text \
    --min-length 50 \
    --dedup-file "$DEDUP"

echo ""
echo "Done. Shards in $SHARD_DIR"
ls -lh "$SHARD_DIR" | tail -5
du -sh "$SHARD_DIR"
