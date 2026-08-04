"""Label ARC tasks with a Chollet cognitive primitive category and a finer mechanic.

Sends each task's training pairs to the LLM and asks for two labels in a single
call: the coarse Chollet category (the confirmatory analysis axis, comparable
across task sets) and a concrete transformation mechanic (exploratory
sub-bucket). Labeling is blind to solve outcomes — the model sees only
training pairs. Output is cached to data/arc_agi_eval_categories.json and used
by generate_tasks_stratified.py to build anchor profiles.

Usage:
  python label_eval_tasks.py                      # public eval set -> data/arc_agi_eval_categories.json
  python label_eval_tasks.py --test                # first 10 tasks, no file written
  python label_eval_tasks.py --limit 10            # first 10 tasks, file written
  python label_eval_tasks.py \\
      --tasks data/generations/gen_20260803_182452/tasks.json \\
      --out   data/generations/gen_20260803_182452/categories.json
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from generate_tasks import load_eval_tasks

client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)

OUTPUT_PATH = Path("data/arc_agi_eval_categories.json")
MODEL = os.environ.get("ARCGEN_MODEL", "gpt-5.6")
MAX_WORKERS = 32
MAX_TRAIN_PAIRS = 3  # cap to keep prompts manageable; 3 is enough to infer the rule

CATEGORIES = {
    "object_centric":     "Identifying, tracking, or transforming discrete objects (connected components, shapes) as units.",
    "geometric":          "Rotation, reflection, scaling, translation, or symmetry operations.",
    "spatial_relational": "Containment, adjacency, alignment, proximity, or relative positioning logic.",
    "numerical":          "Counting, comparison, or using a numeric value to parameterise a transformation.",
    "pattern_completion": "Detecting a repeating or periodic structure and extrapolating or completing it.",
    "compositional":      "Combining two or more of the above primitives in sequence to produce the output.",
}

# Finer sub-bucket: the concrete transformation performed. Deliberately a
# closed list so counts are comparable across task sets. "other" is the escape
# hatch and is expected to stay small — if it exceeds ~15% the list is wrong
# and should be revised before the labels are analysed.
MECHANICS = {
    "symmetry_completion":  "Completing a partially-drawn symmetric pattern using its own symmetry.",
    "reflection":           "Mirroring content across a horizontal, vertical, or diagonal axis.",
    "rotation":             "Rotating content by 90/180/270 degrees.",
    "translation":          "Sliding objects to a new position without changing their form.",
    "tiling_repetition":    "Repeating or tiling a motif to fill or extend a region.",
    "scaling":              "Enlarging or shrinking content by an integer or derived factor.",
    "cropping_extraction":  "Selecting and returning a sub-region or a single object from the input.",
    "flood_fill":           "Filling enclosed or bounded regions with a colour.",
    "denoising":            "Removing stray or noise cells to recover a clean underlying pattern.",
    "object_counting":      "Counting objects and expressing the count in the output.",
    "object_sorting_rank":  "Ordering or ranking objects by size, frequency, or another measure.",
    "recolor_by_property":  "Recolouring objects according to a property such as size, shape, or position.",
    "line_drawing":         "Drawing rays, paths, or connections between marked cells.",
    "gravity_stacking":     "Moving objects until they rest against an edge or each other.",
    "occlusion_repair":     "Reconstructing content hidden behind an occluding shape.",
    "panel_set_operation":  "Combining two or more panels with an overlay or logical operation (AND/OR/XOR).",
    "other":                "None of the above describes the transformation.",
}

CATEGORY_LIST = "\n".join(f'  "{k}": {v}' for k, v in CATEGORIES.items())
MECHANIC_LIST = "\n".join(f'  "{k}": {v}' for k, v in MECHANICS.items())

PROMPT_TEMPLATE = """You are classifying ARC-AGI tasks by transformation type.

Assign TWO labels.

(1) Cognitive primitive category — the six Chollet ARC-AGI categories:
{category_list}

(2) Mechanic — the concrete transformation actually performed:
{mechanic_list}

Rules:
- Assign "compositional" only if the task clearly requires chaining two or more of the above primitives to produce the output.
- When a single primitive dominates, assign that primitive even if a second plays a minor role.
- Pick the single best mechanic. Use "other" only when none genuinely fits — do not stretch a label to avoid it.
- The two labels are independent: pick the mechanic that fits best even if it sits oddly with the category you chose.
- Base your answer solely on the training pairs shown.

Training pairs for this task:
{pairs}

Respond with strict JSON only — no explanation, no markdown fences:
{{"category": "<one of the six category keys above>", "mechanic": "<one of the mechanic keys above>"}}"""


def format_pairs(task: dict) -> str:
    pairs = task["train"][:MAX_TRAIN_PAIRS]
    lines = []
    for i, pair in enumerate(pairs):
        lines.append(f"Pair {i + 1}:")
        lines.append(f"  input:  {pair['input']}")
        lines.append(f"  output: {pair['output']}")
    return "\n".join(lines)


def extract_labels(raw: str) -> tuple:
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)
    parsed = json.loads(raw.strip())
    category = (parsed.get("category") or "").strip() or None
    mechanic = (parsed.get("mechanic") or "").strip() or None
    return category, mechanic


def label_one(task_id: str, task: dict) -> dict:
    prompt = PROMPT_TEMPLATE.format(
        category_list=CATEGORY_LIST,
        mechanic_list=MECHANIC_LIST,
        pairs=format_pairs(task),
    )
    fail = {"task_id": task_id, "category": None, "mechanic": None}
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content
        category, mechanic = extract_labels(raw)
        if category not in CATEGORIES:
            return {**fail, "error": f"unknown category returned: {category!r}"}
        # An unrecognised mechanic falls back to "other" rather than discarding
        # the call — the category is the confirmatory axis and is still good.
        if mechanic not in MECHANICS:
            print(f"  {task_id}: unknown mechanic {mechanic!r} → other", file=sys.stderr)
            mechanic = "other"
        return {"task_id": task_id, "category": category, "mechanic": mechanic, "error": None}
    except Exception as e:
        return {**fail, "error": str(e)}


def run_labeling(items: list) -> tuple:
    """Label a list of (task_id, task) pairs. Returns (categories, mechanics, errors)."""
    categories: dict = {}
    mechanics: dict = {}
    errors: list = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(label_one, tid, task): tid for tid, task in items}
        done = 0
        for future in as_completed(futures):
            r = future.result()
            done += 1
            if r["error"]:
                print(f"  [{done}/{len(items)}] {r['task_id']} ERROR: {r['error']}", file=sys.stderr)
                errors.append({"task_id": r["task_id"], "error": r["error"]})
            else:
                categories[r["task_id"]] = r["category"]
                mechanics[r["task_id"]] = r["mechanic"]
                print(f"  [{done}/{len(items)}] {r['task_id']} → {r['category']} / {r['mechanic']}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\n{len(categories)}/{len(items)} labeled in {elapsed:.1f}s  ({len(errors)} errors)", file=sys.stderr)
    return categories, mechanics, errors


def print_distribution(labels: dict, keys, title: str) -> None:
    counts = Counter(labels.values())
    total = len(labels) or 1
    print(f"\n--- {title} ---", file=sys.stderr)
    for key in sorted(keys, key=lambda k: (-counts.get(k, 0), k)):
        n = counts.get(key, 0)
        bar = "#" * n if total <= 50 else "#" * int(n / total * 40)
        print(f"  {key:<20} {n:>4}  ({n / total * 100:5.1f}%)  {bar}", file=sys.stderr)


def main() -> int:
    global MODEL

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", type=Path, default=None,
                        help="tasks.json to label, as {task_id: {train, test}} (default: the ARC-AGI-1 eval set)")
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH,
                        help=f"output path for labels (default: {OUTPUT_PATH})")
    parser.add_argument("--test", action="store_true",
                        help="label only the first 10 tasks and write nothing")
    parser.add_argument("--limit", type=int, default=None, help="label only the first N tasks")
    parser.add_argument("--model", default=MODEL, help=f"labeling model (default: {MODEL})")
    args = parser.parse_args()
    MODEL = args.model

    tasks = json.loads(args.tasks.read_text()) if args.tasks else load_eval_tasks()
    items = sorted(tasks.items())

    if args.test:
        items = items[:10]
        print(f"TEST MODE: labeling first {len(items)} tasks only (no file written)", file=sys.stderr)
    else:
        if args.limit is not None:
            items = items[:args.limit]
        print(f"Labeling {len(items)} tasks (model={MODEL})...", file=sys.stderr)

    categories, mechanics, errors = run_labeling(items)
    print_distribution(categories, CATEGORIES.keys(), "Category Distribution")
    print_distribution(mechanics, MECHANICS.keys(), "Mechanic Distribution")

    # "other" is the escape hatch; a large share means the closed list is wrong
    # for this task set and the mechanic axis should not be analysed as-is.
    other_share = sum(1 for m in mechanics.values() if m == "other") / (len(mechanics) or 1)
    if other_share > 0.15:
        print(f"\nWARNING: 'other' is {other_share:.1%} of mechanics (>15%) — the mechanic list "
              f"does not fit this task set well; revise it before analysing that axis.", file=sys.stderr)

    if not args.test:
        output = {
            "description": "Chollet category and transformation mechanic for each task, assigned by LLM from training pairs only.",
            "model": MODEL,
            "num_labeled": len(categories),
            "num_errors": len(errors),
            "categories": categories,
            "mechanics": mechanics,
            "errors": errors,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=2))
        print(f"\nSaved → {args.out}", file=sys.stderr)
    else:
        print("\n(test mode — no file written)", file=sys.stderr)

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
