#!/usr/bin/env python3
"""Convert parquet datasets to JSONL for Borno tokenization.

Usage:
  python3 scripts/parquet_to_jsonl.py --input data/distillation/Opus-4.6-Reasoning-3300x/data \
    --output data/distillation/jsonl/opus_reasoning.jsonl --format reasoning

Formats:
  reasoning: combines thinking + solution into text (problem/thinking/solution columns)
  instruct: instruction/input/output → <user>/<assistant> conversation format
  hermes: extracts conversations into text
  text: extracts 'text' column directly
  bangla: extracts 'text' column from Bangla corpus
"""

import argparse
import json
import os
import sys

try:
    import pyarrow.parquet as pq
except ImportError:
    print("pip install pyarrow", file=sys.stderr)
    sys.exit(1)


def convert_reasoning(table):
    """Opus Reasoning / DeepSeek R1 format: problem + thinking + solution"""
    rows = table.to_pydict()
    for i in range(len(table)):
        parts = []
        if 'problem' in rows and rows['problem'][i]:
            parts.append(f"Problem: {rows['problem'][i]}")
        if 'thinking' in rows and rows['thinking'][i]:
            parts.append(f"<think>{rows['thinking'][i]}</think>")
        if 'solution' in rows and rows['solution'][i]:
            parts.append(f"Solution: {rows['solution'][i]}")
        if parts:
            yield {"text": "\n\n".join(parts)}


def convert_instruct(table):
    """Instruction/input/output format (e.g. DeepSeek R1) → <user>/<assistant>"""
    rows = table.to_pydict()
    for i in range(len(table)):
        instr = rows.get('instruction', [None] * len(table))[i] or ''
        inp = rows.get('input', [None] * len(table))[i] or ''
        out = rows.get('output', [None] * len(table))[i] or ''
        user_text = f"{instr}\n{inp}".strip() if inp else instr.strip()
        if user_text and out.strip():
            yield {"text": f"<user>{user_text}</user>\n<assistant>{out.strip()}</assistant>"}


def convert_hermes(table):
    """OpenHermes format: conversations list"""
    rows = table.to_pydict()
    for i in range(len(table)):
        if 'conversations' in rows:
            convs = rows['conversations'][i]
            if convs:
                parts = []
                for msg in convs:
                    role = msg.get('from', msg.get('role', 'user'))
                    content = msg.get('value', msg.get('content', ''))
                    parts.append(f"<{role}>{content}</{role}>")
                yield {"text": "\n".join(parts)}
        elif 'text' in rows and rows['text'][i]:
            yield {"text": rows['text'][i]}


def convert_text(table):
    """Simple text column extraction"""
    rows = table.to_pydict()
    text_col = 'text' if 'text' in rows else list(rows.keys())[0]
    for i in range(len(table)):
        if rows[text_col][i]:
            yield {"text": str(rows[text_col][i])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory of parquet files")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--format", choices=["reasoning", "instruct", "hermes", "text", "bangla"],
                        default="text")
    parser.add_argument("--max-rows", type=int, default=0, help="Max rows (0=all)")
    args = parser.parse_args()

    converters = {
        "reasoning": convert_reasoning,
        "instruct": convert_instruct,
        "hermes": convert_hermes,
        "text": convert_text,
        "bangla": convert_text,
    }
    convert = converters[args.format]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    files = sorted([
        os.path.join(args.input, f)
        for f in os.listdir(args.input)
        if f.endswith('.parquet')
    ])
    print(f"Found {len(files)} parquet files in {args.input}")

    count = 0
    with open(args.output, 'w') as out:
        for fpath in files:
            table = pq.read_table(fpath)
            for row in convert(table):
                if row and row.get("text") and len(row["text"]) > 50:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
                    if args.max_rows and count >= args.max_rows:
                        break
            if args.max_rows and count >= args.max_rows:
                break
            print(f"  {os.path.basename(fpath)}: {count} rows so far")

    print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
