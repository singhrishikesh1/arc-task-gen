"""Generate a plain-English description of the transformation rule for each eval task.

Sends training pairs to the LLM and asks for a 1–2 sentence rule description.
Output is cached to data/arc_agi_eval_descriptions.json and used by generate_tasks.py
to detect generated tasks that are too similar to official eval tasks.

Usage:
  python describe_eval_tasks.py            # full run (all 400 tasks)
  python describe_eval_tasks.py --test     # first 10 tasks only, no file written
  python describe_eval_tasks.py --limit 10 # first 10 tasks only, file written
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from generate_tasks import load_eval_tasks

client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)

EVAL_PATH = Path("data/arc_agi_eval.json")
OUTPUT_PATH = Path("data/arc_agi_eval_descriptions.json")
MODEL = os.environ.get("ARCGEN_MODEL", "gpt-5.6")
MAX_WORKERS = 32
MAX_TRAIN_PAIRS = 3

PROMPT_TEMPLATE = """You are describing the transformation rule of an ARC-AGI task.

Training pairs for this task:
{pairs}

In 1–2 sentences, describe the exact rule that converts each input grid to its output grid. \
Be specific about the structural change — what is detected in the input and what is produced \
in the output. Do not mention specific color values or grid sizes.

Respond with strict JSON only — no explanation, no markdown fences:
{{"description": "<1-2 sentence rule description>"}}"""


def format_pairs(task: dict) -> str:
    pairs = task["train"][:MAX_TRAIN_PAIRS]
    lines = []
    for i, pair in enumerate(pairs):
        lines.append(f"Pair {i + 1}:")
        lines.append(f"  input:  {pair['input']}")
        lines.append(f"  output: {pair['output']}")
    return "\n".join(lines)


def describe_one(task_id: str, task: dict) -> dict:
    prompt = PROMPT_TEMPLATE.format(pairs=format_pairs(task))
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
        parsed = json.loads(raw.strip())
        desc = parsed.get("description", "").strip()
        if not desc:
            return {"task_id": task_id, "description": None, "error": "empty description"}
        return {"task_id": task_id, "description": desc, "error": None}
    except Exception as e:
        return {"task_id": task_id, "description": None, "error": str(e)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", action="store_true", help="describe only the first 10 tasks and write nothing")
    parser.add_argument("--limit", type=int, default=None, help="describe only the first N tasks")
    args = parser.parse_args()

    eval_tasks = load_eval_tasks()
    items = list(eval_tasks.items())

    if args.test:
        items = items[:10]
        print(f"TEST MODE: describing first {len(items)} tasks only (no file written)", file=sys.stderr)
    else:
        if args.limit is not None:
            items = items[:args.limit]
        print(f"Describing {len(items)} eval tasks (model={MODEL})...", file=sys.stderr)

    descriptions = {}
    errors = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(describe_one, tid, task): tid for tid, task in items}
        done = 0
        for future in as_completed(futures):
            r = future.result()
            done += 1
            if r["error"]:
                print(f"  [{done}/{len(items)}] {r['task_id']} ERROR: {r['error']}", file=sys.stderr)
                errors.append({"task_id": r["task_id"], "error": r["error"]})
            else:
                descriptions[r["task_id"]] = r["description"]
                print(f"  [{done}/{len(items)}] {r['task_id']}: {r['description']}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\n{len(descriptions)}/{len(items)} described in {elapsed:.1f}s  ({len(errors)} errors)", file=sys.stderr)

    if not args.test:
        output = {
            "description": "Plain-English transformation rule description for each ARC-AGI-1 eval task.",
            "model": MODEL,
            "num_described": len(descriptions),
            "num_errors": len(errors),
            "descriptions": descriptions,
            "errors": errors,
        }
        OUTPUT_PATH.write_text(json.dumps(output, indent=2))
        print(f"Saved → {OUTPUT_PATH}", file=sys.stderr)
    else:
        print("(test mode — no file written)", file=sys.stderr)

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
