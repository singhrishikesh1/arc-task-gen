"""Generate ARC-AGI-1-style tasks matched to a public eval anchor's shape/colour
profile, optionally stratified by transformation mechanic.

Merges what used to be two separate wrappers (generate_tasks_matched.py,
generate_tasks_stratified.py) into one. Wraps generate_tasks.py: keeps its
generation loop, dedup, eval-similarity filter, validation and output format,
and only replaces the slot sampler and prompt.

Each generated task is conditioned on one public eval anchor's train/test shape
sequence, its LLM-assigned Chollet category, and — when --mechanics is given —
a target transformation mechanic, allocated evenly across mechanics rather than
left to their natural (very skewed) distribution. Public grids, rule
descriptions and task ids never enter the prompt; only shape/colour statistics
and labels do.

Requires (alongside the auto-downloaded data/arc_agi_eval.json):
  data/arc_agi_eval_categories.json  - category + mechanic label per eval task,
                                        produced by label_eval_tasks.py

Anchors are drawn only from eval tasks that have a label. label_eval_tasks.py
can be run on a subset (--limit N) for a cheap smoke test; --anchor-mode
all-anchors then just needs --n to match however many anchors are labeled.

Usage:
  python generate_tasks_stratified.py --n 400                              # category-matched only
  python generate_tasks_stratified.py --mechanics flood_fill rotation --per-mechanic 20
  python generate_tasks_stratified.py --mechanics all --per-mechanic 94    # full powered design
  python generate_tasks_stratified.py --n 400 --dry-run-plan               # inspect plan, no API calls
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
EVAL_PATH = ROOT / "data/arc_agi_eval.json"
LABELS_PATH = ROOT / "data/arc_agi_eval_categories.json"  # category + mechanic, from label_eval_tasks.py
GENERATIONS_DIR = ROOT / "data/generations_stratified"

MIN_ANCHORS = 8  # below this, a mechanic borrows anchors from its modal category

CATEGORY_DESCRIPTIONS = {
    "object_centric": (
        "Identifying, tracking, or transforming discrete objects "
        "(connected components or shapes) as units."
    ),
    "geometric": "Rotation, reflection, scaling, translation, or symmetry operations.",
    "spatial_relational": (
        "Containment, adjacency, alignment, proximity, or relative positioning logic."
    ),
    "numerical": "Counting, comparison, or numeric values parameterising a transformation.",
    "pattern_completion": (
        "Detecting a repeating or periodic structure and extrapolating or completing it."
    ),
    "compositional": "Combining two or more primitives in sequence.",
}

MECHANIC_GUIDANCE = {
    "symmetry_completion":  "complete a partially-drawn symmetric pattern using its own symmetry",
    "reflection":           "mirror content across a horizontal, vertical or diagonal axis",
    "rotation":             "rotate content by 90, 180 or 270 degrees",
    "translation":          "slide objects to new positions without changing their form",
    "tiling_repetition":    "repeat or tile a motif to fill or extend a region",
    "scaling":              "enlarge or shrink content by a factor derived from the input",
    "cropping_extraction":  "select and return a sub-region or a single object from the input",
    "flood_fill":           "fill enclosed or bounded regions with a colour",
    "denoising":            "remove stray or noise cells to recover a clean underlying pattern",
    "object_counting":      "count objects and express the count in the output",
    "object_sorting_rank":  "order or rank objects by size, frequency or another measure",
    "recolor_by_property":  "recolour objects according to a property such as size, shape or position",
    "line_drawing":         "draw rays, paths or connections between marked cells",
    "gravity_stacking":     "move objects until they rest against an edge or each other",
    "occlusion_repair":     "reconstruct content hidden behind an occluding shape",
    "panel_set_operation":  "combine two or more panels with an overlay or logical operation",
}

PROMPT_TEMPLATE = """Generate 1 original ARC-AGI-1-style task. The task must be a genuinely new
transformation rule you invent — do NOT copy, lightly edit, rotate/reflect,
recolor, crop, or otherwise derive a task from any existing ARC-AGI dataset
(public or private). The concrete grids and rule must be hand-built from
scratch.

This task must match a sampled ARC-AGI-1 evaluation-slot profile without seeing
the public task itself.

Slot profile:
- Cognitive primitive category: {category}
  {category_description}
- Exactly {n_train} training input/output pairs.
- Exactly {n_test} test pair(s).
- Use approximately {target_colors} distinct non-background colors across the
  whole task.
- Training input grid dimensions, in order: {train_input_shapes}
- Test input grid dimensions, in order: {test_input_shapes}
- Training output grid dimensions, in order: {train_output_shapes}
- Test output grid dimensions, in order: {test_output_shapes}
- Input shape varies within task: {input_shape_varies}
- Output shape varies within task: {output_shape_varies}
- Approximate non-background density: {density_label}
- Output-size relation: {shape_relation}

Distribution-matching guidance:
- Make the task feel like an ARC-AGI-1 public evaluation task in this category,
  not like a programming contest puzzle.
- Prefer visual object, geometry, spatial relation, or pattern-completion rules
  when the category calls for them.
- Avoid artificial arithmetic, sorting, indexing, modulo/parity, graph-degree,
  or row-major ledger/readout rules unless the category is explicitly
  "numerical" or the above shape profile strongly requires compression/readout.
- If the public-slot profile has varying input or output dimensions, your task
  should also vary those dimensions in the same train/test positions.
- The transformation rule must be unambiguously inferable from the training
  pairs alone, without seeing the test output.

Before returning the result, verify the task yourself: check the rule actually
and unambiguously produces every training output from its input, and that it is
solvable from the training pairs alone.

Output strict JSON with exactly two top-level keys:

{{
  "tasks": {{
    "<short_unique_task_id>": {{
      "train": [
        {{"input": [[...]], "output": [[...]]}},
        ...
      ],
      "test": [
        {{"input": [[...]], "output": [[...]]}},
        ...
      ]
    }}
  }},
  "descriptions": {{
    "<short_unique_task_id>": "One or two sentences describing the exact transformation rule in plain English."
  }}
}}"""

_PLAN: list[dict[str, Any]] = []
_CURSOR = 0


# --- anchor profiles (public eval task -> sampled slot) --------------------

def grid_shape(grid: list[list[int]]) -> tuple[int, int]:
    return len(grid), len(grid[0]) if grid else 0


def nonzero_density(grids: list[list[list[int]]]) -> float:
    cells = filled = 0
    for grid in grids:
        for row in grid:
            for value in row:
                cells += 1
                filled += int(value != 0)
    return filled / cells if cells else 0.0


def density_label(value: float) -> str:
    if value < 0.08:
        return "very sparse"
    if value < 0.18:
        return "sparse"
    if value < 0.38:
        return "medium"
    return "dense"


def shape_relation(input_shapes: list[tuple[int, int]], output_shapes: list[tuple[int, int]]) -> str:
    input_area = sum(h * w for h, w in input_shapes)
    output_area = sum(h * w for h, w in output_shapes)
    ratio = output_area / input_area if input_area else 1.0
    if all(a == b for a, b in zip(input_shapes, output_shapes)):
        return "same_size"
    if ratio < 0.65:
        return "extract_or_compress"
    if ratio > 1.35:
        return "expand"
    return "resize_mixed"


def task_colors(task: dict[str, Any]) -> set[int]:
    colors: set[int] = set()
    for pair in task["train"] + task["test"]:
        for field in ("input", "output"):
            for row in pair[field]:
                colors.update(int(value) for value in row if value != 0)
    return colors


def anchor_profile(task_id: str, task: dict[str, Any], category: str) -> dict[str, Any]:
    train_input_shapes = [grid_shape(pair["input"]) for pair in task["train"]]
    test_input_shapes = [grid_shape(pair["input"]) for pair in task["test"]]
    train_output_shapes = [grid_shape(pair["output"]) for pair in task["train"]]
    test_output_shapes = [grid_shape(pair["output"]) for pair in task["test"]]
    input_shapes = train_input_shapes + test_input_shapes
    output_shapes = train_output_shapes + test_output_shapes
    input_grids = [pair["input"] for pair in task["train"] + task["test"]]
    return {
        "anchor": task_id,
        "category": category,
        "category_description": CATEGORY_DESCRIPTIONS[category],
        # rows/cols of the first input shape — generate_tasks.py's own progress
        # logging (not our format_prompt) reads these directly off every slot.
        "rows": input_shapes[0][0],
        "cols": input_shapes[0][1],
        "colors": len(task_colors(task)),
        "n_train": len(task["train"]),
        "n_test": len(task["test"]),
        "train_input_shapes": train_input_shapes,
        "test_input_shapes": test_input_shapes,
        "train_output_shapes": train_output_shapes,
        "test_output_shapes": test_output_shapes,
        "input_shape_varies": len(set(input_shapes)) > 1,
        "output_shape_varies": len(set(output_shapes)) > 1,
        "shape_relation": shape_relation(input_shapes, output_shapes),
        "density": nonzero_density(input_grids),
    }


def load_anchor_profiles() -> dict[str, dict[str, Any]]:
    """Anchor profile per labeled eval task. Anchors without a category label
    (e.g. label_eval_tasks.py was only run with --limit) are silently excluded
    rather than erroring, so a partial label set still produces a usable —
    just smaller — anchor pool."""
    eval_tasks = json.loads(EVAL_PATH.read_text())
    categories = json.loads(LABELS_PATH.read_text())["categories"]
    labeled = {tid: task for tid, task in eval_tasks.items() if tid in categories}
    if len(labeled) < len(eval_tasks):
        print(
            f"{len(eval_tasks) - len(labeled)}/{len(eval_tasks)} eval tasks have no category "
            f"label — using the remaining {len(labeled)} as the anchor pool.",
            file=sys.stderr,
        )
    if not labeled:
        raise ValueError(f"No labeled eval tasks found in {LABELS_PATH}")
    return {tid: anchor_profile(tid, task, categories[tid]) for tid, task in labeled.items()}


# --- slot plans --------------------------------------------------------------

def build_category_plan(profiles: dict[str, dict[str, Any]], n: int, seed: int, mode: str) -> list[dict[str, Any]]:
    """One slot per generated task, sampled by category proportions (no mechanic)."""
    rng = random.Random(seed)
    if mode == "all-anchors":
        if n != len(profiles):
            raise ValueError(f"--anchor-mode=all-anchors requires --n={len(profiles)}, got {n}")
        task_ids = sorted(profiles)
        rng.shuffle(task_ids)
        return [profiles[tid] for tid in task_ids]

    by_category: dict[str, list[str]] = defaultdict(list)
    for tid, profile in profiles.items():
        by_category[profile["category"]].append(tid)

    category_counts = Counter(profile["category"] for profile in profiles.values())
    raw_targets = {c: n * count / len(profiles) for c, count in category_counts.items()}
    targets = {c: int(v) for c, v in raw_targets.items()}
    remaining = n - sum(targets.values())
    for c, _ in sorted(raw_targets.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True)[:remaining]:
        targets[c] += 1

    slots = []
    for c, count in targets.items():
        ids = by_category[c][:]
        rng.shuffle(ids)
        chosen = ids[:count] if count <= len(ids) else ids + [rng.choice(ids) for _ in range(count - len(ids))]
        slots.extend(profiles[tid] for tid in chosen)
    rng.shuffle(slots)
    return slots


def build_mechanic_plan(
    profiles: dict[str, dict[str, Any]], per_mechanic: int, seed: int, mechanics: list[str]
) -> list[dict[str, Any]]:
    """One slot per generated task: an anchor profile plus the mechanic it must realise.

    Mechanics are allocated evenly (per_mechanic each) rather than left to their
    natural, very skewed distribution. Where a mechanic has too few public
    anchors (< MIN_ANCHORS), the anchor pool widens to its modal category.
    """
    labels = json.loads(LABELS_PATH.read_text())
    mech_of, cat_of = labels.get("mechanics", {}), labels.get("categories", {})

    by_mech: dict[str, list[str]] = defaultdict(list)
    by_cat: dict[str, list[str]] = defaultdict(list)
    for tid in profiles:
        if tid in mech_of:
            by_mech[mech_of[tid]].append(tid)
        if tid in cat_of:
            by_cat[cat_of[tid]].append(tid)

    rng = random.Random(seed)
    plan = []
    for mech in mechanics:
        pool = list(by_mech.get(mech, []))
        borrowed = False
        if len(pool) < MIN_ANCHORS:
            cats = Counter(cat_of[t] for t in pool if t in cat_of)
            if cats:
                pool = list(dict.fromkeys(pool + by_cat[cats.most_common(1)[0][0]]))
                borrowed = True
        if not pool:
            print(f"  ! no anchors for {mech}, skipping", file=sys.stderr)
            continue
        for _ in range(per_mechanic):
            slot = dict(profiles[rng.choice(pool)])
            slot["mechanic"] = mech
            slot["mechanic_guidance"] = MECHANIC_GUIDANCE[mech]
            slot["anchor_pool_borrowed"] = borrowed
            plan.append(slot)
    rng.shuffle(plan)
    return plan


def sample_plan_slots(_eval_tasks: dict[str, Any], n: int) -> list[dict[str, Any]]:
    global _CURSOR
    if not _PLAN:
        raise RuntimeError("slot plan was not initialized")
    out = []
    for _ in range(n):
        out.append(dict(_PLAN[_CURSOR % len(_PLAN)]))
        _CURSOR += 1
    return out


# --- prompting ---------------------------------------------------------------

def _format_shapes(shapes: list[tuple[int, int]]) -> str:
    return ", ".join(f"{h}x{w}" for h, w in shapes)


def format_prompt(slot: dict[str, Any], avoid: list[str] | None = None) -> str:
    prompt = PROMPT_TEMPLATE.format(
        category=slot["category"],
        category_description=slot["category_description"],
        n_train=slot["n_train"],
        n_test=slot["n_test"],
        target_colors=slot["colors"],
        train_input_shapes=_format_shapes(slot["train_input_shapes"]),
        test_input_shapes=_format_shapes(slot["test_input_shapes"]),
        train_output_shapes=_format_shapes(slot["train_output_shapes"]),
        test_output_shapes=_format_shapes(slot["test_output_shapes"]),
        input_shape_varies="yes" if slot["input_shape_varies"] else "no",
        output_shape_varies="yes" if slot["output_shape_varies"] else "no",
        density_label=density_label(float(slot["density"])),
        shape_relation=slot["shape_relation"],
    )
    if slot.get("mechanic"):
        prompt = prompt.replace(
            "Distribution-matching guidance:",
            f"Required transformation mechanic: **{slot['mechanic']}** — the rule must "
            f"{slot['mechanic_guidance']}. Invent an original rule of this kind; do not "
            f"reuse a known ARC task. The mechanic is a constraint on the *kind* of "
            f"transformation, not on the specific rule, which must still be novel.\n\n"
            "Distribution-matching guidance:",
        )
    if avoid:
        avoid_block = "\n".join(f"  - {rule}" for rule in avoid)
        prompt += (
            "\n\nIMPORTANT — diversity requirement: the transformation rule you "
            "invent must be distinctly different in its core mechanism from each "
            "of the following rules already used in this task set. A rule that is "
            "a minor variation, rename, or reframing of one of these will be "
            "rejected:\n"
            + avoid_block
        )
    return prompt


def extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
    return json.loads(text.strip())


def make_generate_one(base: Any):
    def generate_one(idx: int, slot: dict[str, Any], avoid: list[str] | None = None) -> dict[str, Any]:
        prompt = format_prompt(slot, avoid)
        try:
            response = base.client.chat.completions.create(
                model=base.MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            parsed = extract_json(response.choices[0].message.content)
            return {
                "idx": idx,
                "tasks": parsed.get("tasks", {}),
                "descriptions": parsed.get("descriptions", {}),
                "error": None,
            }
        except Exception as exc:  # pragma: no cover - API path
            return {"idx": idx, "tasks": {}, "descriptions": {}, "error": str(exc)}

    return generate_one


def import_base_generator() -> Any:
    spec = importlib.util.spec_from_file_location("arc_generate_tasks_base", ROOT / "generate_tasks.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import generate_tasks.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_plan(plan: list[dict[str, Any]], path: Path) -> None:
    payload = {
        "num_slots": len(plan),
        "category_counts": dict(sorted(Counter(s["category"] for s in plan).items())),
        "shape_relations": dict(sorted(Counter(s["shape_relation"] for s in plan).items())),
        "input_shape_varies": sum(1 for s in plan if s["input_shape_varies"]),
        "output_shape_varies": sum(1 for s in plan if s["output_shape_varies"]),
    }
    mechanics_used = [s["mechanic"] for s in plan if "mechanic" in s]
    if mechanics_used:
        payload["mechanic_counts"] = dict(sorted(Counter(mechanics_used).items()))
        borrowed = sorted({s["mechanic"] for s in plan if s.get("anchor_pool_borrowed")})
        if borrowed:
            payload["anchor_pool_borrowed_for"] = borrowed
    payload["slots"] = plan
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


# --- CLI ---------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=400, help="total tasks; ignored when --mechanics is set")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--anchor-mode", choices=["all-anchors", "stratified"], default="all-anchors",
        help="category sampling when --mechanics is not set: all-anchors uses each public "
             "eval task once (requires --n=400); stratified preserves category proportions.",
    )
    parser.add_argument(
        "--mechanics", nargs="*", default=None,
        help="enable mechanic stratification: a list of mechanics, or 'all' for every "
             f"mechanic ({', '.join(sorted(MECHANIC_GUIDANCE))}).",
    )
    parser.add_argument("--per-mechanic", type=int, default=25, help="tasks per mechanic (only with --mechanics)")
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--generations-dir", type=Path, default=GENERATIONS_DIR)
    parser.add_argument("--dry-run-plan", action="store_true", help="write the slot plan and exit, no API calls")
    parser.add_argument("--plan-output", type=Path, default=ROOT / "data/stratified_slot_plan.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global _PLAN
    args = parse_args(argv)
    profiles = load_anchor_profiles()

    if args.mechanics is not None:
        requested = sorted(MECHANIC_GUIDANCE) if args.mechanics in ([], ["all"]) else args.mechanics
        unknown = [m for m in requested if m not in MECHANIC_GUIDANCE]
        if unknown:
            raise SystemExit(f"unknown mechanic(s): {unknown}")
        _PLAN = build_mechanic_plan(profiles, args.per_mechanic, args.seed, requested)
        print(f"Plan: {len(_PLAN)} tasks across {len(requested)} mechanics "
              f"({args.per_mechanic} each)", file=sys.stderr)
    else:
        _PLAN = build_category_plan(profiles, args.n, args.seed, args.anchor_mode)
        print(f"Plan: {len(_PLAN)} tasks (anchor-mode={args.anchor_mode})", file=sys.stderr)

    if not _PLAN:
        raise SystemExit("empty slot plan — nothing to generate")

    write_plan(_PLAN, args.plan_output)
    print(f"Wrote slot plan -> {args.plan_output}", file=sys.stderr)
    if args.dry_run_plan:
        return 0

    os.chdir(ROOT)
    base = import_base_generator()
    base.sample_joint_slots = sample_plan_slots
    base.generate_one = make_generate_one(base)
    base.GENERATIONS_DIR = args.generations_dir
    if args.model is not None:
        base.MODEL = args.model
    if args.max_workers is not None:
        base.MAX_WORKERS = args.max_workers

    base_argv = ["generate_tasks.py", "--n", str(len(_PLAN))]
    if args.max_rounds is not None:
        base_argv += ["--max-rounds", str(args.max_rounds)]
    old_argv, sys.argv = sys.argv, base_argv
    try:
        return int(base.main())
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
