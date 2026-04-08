#!/usr/bin/env python3
"""Generate synthetic Bangla training data using free APIs (Groq + Fireworks).

Usage:
  python3 scripts/generate_bangla_synthetic.py --output data/distillation/jsonl/bangla_synthetic.jsonl --count 1000
"""

import argparse
import json
import os
import time
import random
import requests

GROQ_KEY = os.environ.get("GROQ_KEY", "")
FIREWORKS_KEY = os.environ.get("FIREWORKS_KEY", "")

PROMPTS = [
    # Math reasoning
    "Write a math word problem in Bangla (Bengali script) suitable for grade 5 students. Include the problem and a detailed step-by-step solution in Bangla.",
    "Create a percentage or ratio problem in Bangla with step-by-step solution. Use real-world context like shopping or cooking.",
    "Write a geometry problem in Bangla about area or perimeter with detailed solution steps.",
    # Science
    "Write a short science explanation in Bangla about how rain forms. Make it educational for students.",
    "Explain photosynthesis in Bangla in simple terms with examples from Bangladesh agriculture.",
    "Write a short Bangla explanation of how electricity works, suitable for a curious child.",
    # History/Culture
    "Write a short educational passage in Bangla about the history of the Bengali language movement (ভাষা আন্দোলন).",
    "Write a short informative text in Bangla about the Sundarbans and its importance to Bangladesh.",
    "Write a passage in Bangla about the Liberation War of 1971 suitable for young students.",
    # Reasoning
    "Write a logical reasoning puzzle in Bangla with the solution explained step by step.",
    "Create a pattern recognition problem in Bangla (like: 2, 6, 12, 20, ? — find the next number) with explanation.",
    "Write a Bangla riddle (ধাঁধা) with its answer and explanation of the logic.",
    # Instruction following
    "Write instructions in Bangla for how to make traditional Bengali doi (দই/yogurt) at home.",
    "Write a Bangla guide on how to plant and care for a mango tree in Bangladesh climate.",
    "Write step-by-step instructions in Bangla for basic first aid for a small cut.",
    # Code explanation
    "Explain what a for loop is in programming, written entirely in Bangla with a simple Python example.",
    "Explain the concept of variables in programming in Bangla, with examples.",
    "Write a Bangla explanation of how to sort a list of numbers, with pseudocode in Bangla.",
    # Creative writing
    "Write a short story in Bangla (200 words) about a fisherman in the rivers of Bangladesh.",
    "Write a Bangla poem about the monsoon season in Bangladesh.",
    "Write a short Bangla essay about the importance of education for girls in rural Bangladesh.",
    # Conversational
    "Write a dialogue in Bangla between a doctor and a patient about common cold symptoms and remedies.",
    "Write a Bangla conversation between a teacher and student about the importance of reading books.",
    "Write a dialogue in Bangla between two friends planning a trip to Cox's Bazar.",
    # Translation + Reasoning
    "Translate this to Bangla and explain the concept: 'The area of a circle is pi times the radius squared.'",
    "Translate and explain in Bangla: 'Water boils at 100 degrees Celsius at sea level.'",
    "Translate to Bangla and give 3 examples: 'Renewable energy sources include solar, wind, and hydroelectric power.'",
    # English reasoning (for bilingual capability)
    "Write a logic problem in English about scheduling and solve it step by step.",
    "Explain the concept of compound interest in English with a worked example.",
    "Write a short English passage explaining how the internet works, suitable for beginners.",
]


def call_groq(prompt, max_tokens=1000):
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
        json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens},
        timeout=30,
    )
    result = resp.json()
    if "choices" in result:
        return result["choices"][0]["message"]["content"]
    return None


def call_fireworks(prompt, max_tokens=1000):
    resp = requests.post(
        "https://api.fireworks.ai/inference/v1/chat/completions",
        headers={"Authorization": f"Bearer {FIREWORKS_KEY}", "Content-Type": "application/json"},
        json={"model": "accounts/fireworks/models/llama-v3p3-70b-instruct", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens},
        timeout=30,
    )
    result = resp.json()
    if "choices" in result:
        return result["choices"][0]["message"]["content"]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/distillation/jsonl/bangla_synthetic.jsonl")
    parser.add_argument("--count", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    apis = [("Groq", call_groq), ("Fireworks", call_fireworks)]
    generated = 0
    errors = 0

    with open(args.output, "a") as f:
        while generated < args.count:
            prompt = random.choice(PROMPTS)
            api_name, api_fn = apis[generated % len(apis)]  # alternate between APIs

            try:
                text = api_fn(prompt)
                if text and len(text) > 100:
                    entry = {"text": f"<user>{prompt}</user>\n<assistant>{text}</assistant>"}
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    generated += 1
                    if generated % 10 == 0:
                        print(f"  Generated {generated}/{args.count} ({api_name})")
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                if errors > 50:
                    print(f"Too many errors ({errors}), stopping.")
                    break

            # Rate limiting: Groq = 30/min, Fireworks = 600/min
            if api_name == "Groq":
                time.sleep(2.1)  # ~28/min
            else:
                time.sleep(0.5)

    print(f"\nDone. Generated {generated} examples, {errors} errors.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
