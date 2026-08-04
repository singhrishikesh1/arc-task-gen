"""Generate original ARC-AGI-1-style tasks matched to the ARC-AGI-1 eval distribution.

Produces a private task set for evaluating a model on ARC-style problems it cannot
have memorised, since the public ARC-AGI-1 evaluation set appears in web-scraped
training corpora.

Each run creates a timestamped folder under data/generations/ containing:
  tasks.json          - all generated tasks, keyed by task id
  sanity_check.json   - distribution comparison vs the ARC-AGI-1 eval set, per-task
                        rule descriptions, slot provenance and per-round filter history
  separated/          - one JSON file per task, in the same shape as the files in
                        ARC-AGI/data/evaluation, so they load in the ARC testing interface

Generation runs against any OpenAI-compatible chat-completions endpoint (OpenAI,
vLLM, Ollama, LM Studio, ...). Configure with:
  OPENAI_API_KEY    required by the client; use any placeholder for local servers
  OPENAI_BASE_URL   optional, e.g. http://localhost:8000/v1
  ARCGEN_MODEL      generation model            (default: gpt-5.6)
  ARCGEN_EMBED_MODEL  embedding model for the novelty filters
                      (default: text-embedding-3-small)

Scoring a model against the generated set is deliberately out of scope; tasks.json
carries the standard ARC {"train": [...], "test": [...]} shape, so any ARC harness
can consume it.
"""

import io
import json
import math
import os
import re
import statistics
import sys
import tarfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
from openai import OpenAI

# Any OpenAI-compatible endpoint. base_url is read from OPENAI_BASE_URL when set,
# which covers local servers (vLLM, Ollama, LM Studio) and third-party gateways.
client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)

N = 32
MAX_WORKERS = 64          # concurrent API calls; capped to avoid rate limits at large N

# Post-generation filters applied in the convergence loop. Structural validation is
# always enforced there regardless of these flags.
ENABLE_DEDUP = True
ENABLE_EVAL_SIMILARITY = True
EVAL_PATH = Path("data/arc_agi_eval.json")
# ARC-AGI-1 evaluation set (400 tasks), fetched once and cached. Apache 2.0,
# https://github.com/fchollet/ARC-AGI
DATASET_TARBALL = "https://codeload.github.com/fchollet/ARC-AGI/tar.gz/refs/heads/master"
DATASET_PREFIX = "ARC-AGI-master/data/evaluation/"
EVAL_DESCRIPTIONS_PATH = Path("data/arc_agi_eval_descriptions.json")
GENERATIONS_DIR = Path("data/generations")
MODEL = os.environ.get("ARCGEN_MODEL", "gpt-5.6")
DEDUP_THRESHOLD = 0.80          # cosine similarity above which two generated tasks are near-duplicates
# Calibrated against the eval set's own internal structure: the nearest-neighbour
# similarity between GENUINELY DISTINCT eval tasks has median 0.760, p95 0.879 and
# max 0.912 (38/400 pairs exceed 0.85). A threshold of 0.85 therefore sits inside
# the normal range for unrelated tasks and would delete legitimate novel work.
# 0.92 sits just above the observed maximum for distinct eval pairs.
EVAL_SIMILARITY_THRESHOLD = 0.92
EMBEDDING_MODEL = os.environ.get("ARCGEN_EMBED_MODEL", "text-embedding-3-small")

PROMPT_TEMPLATE = """Generate 1 original ARC-AGI-1-style task. The task must be a genuinely new
transformation rule you invent — do NOT copy, lightly edit, rotate/reflect,
recolor, crop, or otherwise derive a task from any existing ARC-AGI dataset
(public or private). Draw inspiration from ARC-AGI's general categories if useful
(symmetry completion, object counting, gravity/containment, pattern extrapolation,
rule-based recoloring, etc.) but the concrete grids and rule must be
hand-built from scratch.

Constraints per task, matching the ARC-AGI-1 evaluation set's distribution:
- Grids: rectangular, 1x1 up to 30x30. The input grid for this task should be approximately {target_rows} rows × {target_cols} cols.
- Colors: integers 0-9 (0 = background). Use approximately {target_colors} distinct
  non-background colors across the whole task.
- Exactly {n_train} training input/output pairs.
- Exactly {n_test} test pair(s).
- The transformation rule must be unambiguously inferable from the training
  pairs alone, without seeing the test output.

Before returning the result, verify the task yourself: check the rule
actually and unambiguously produces every training output from its input,
and that it's solvable from the training pairs alone (not guessable only
from the test pair).

Output strict JSON with exactly two top-level keys:

{
  "tasks": {
    "<short_unique_task_id>": {
      "train": [
        {"input": [[...]], "output": [[...]]},
        ...
      ],
      "test": [
        {"input": [[...]], "output": [[...]]},
        ...
      ]
    }
  },
  "descriptions": {
    "<short_unique_task_id>": "One or two sentences describing the exact transformation rule in plain English."
  }
}"""

def load_eval_tasks() -> dict:
    """Return the 400-task ARC-AGI-1 evaluation set, downloading it on first use.

    The set is the reference distribution every generated task is sampled against,
    so it is required. Cached to EVAL_PATH after the first fetch.
    """
    if EVAL_PATH.exists():
        return json.loads(EVAL_PATH.read_text())

    print(f"Downloading ARC-AGI-1 evaluation set -> {EVAL_PATH} ...", file=sys.stderr)
    resp = httpx.get(DATASET_TARBALL, timeout=300, follow_redirects=True)
    resp.raise_for_status()

    tasks = {}
    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.startswith(DATASET_PREFIX) and member.name.endswith(".json"):
                tasks[Path(member.name).stem] = json.loads(tar.extractfile(member).read())

    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_PATH.write_text(json.dumps(tasks))
    print(f"Cached {len(tasks)} evaluation tasks.", file=sys.stderr)
    return tasks


def sample_joint_slots(eval_tasks: dict, n: int) -> list:
    """Sample n slots wholesale from single randomly-chosen eval tasks.

    Each slot is a dict with keys: rows, cols, colors, n_train, n_test, anchor.
    Every property is read off the SAME anchor task, so the natural covariance
    between grid size, colour count and pair counts is preserved. Sampling these
    marginally and independently would produce incoherent combinations — e.g. a
    2x2 grid asked to carry 9 distinct colours, or a 7-pair task on a 3x3 canvas.

    Colours are counted as distinct non-background values across every grid of
    the task, matching how compute_stats() measures colors_per_task.
    """
    import random

    task_ids = list(eval_tasks.keys())
    slots = []
    for _ in range(n):
        anchor_id = random.choice(task_ids)
        task = eval_tasks[anchor_id]

        # Grid dims are drawn from INPUT grids only, because the slot sets the input
        # target. Drawing from inputs+outputs pooled (the pre-2026-07-27 behaviour)
        # under-sizes inputs by ~8%: eval outputs are smaller than inputs on average
        # (189.7 vs 226.9 mean area), so the pooled distribution sits below the
        # input-only one. Colours are still counted across every grid, matching how
        # compute_stats() measures colors_per_task.
        input_grids, colors = [], set()
        for pair in task["train"] + task["test"]:
            for field in ("input", "output"):
                g = pair.get(field)
                if not g:
                    continue
                if field == "input":
                    input_grids.append((len(g), len(g[0])))
                for row in g:
                    colors.update(v for v in row if v != 0)

        rows, cols = random.choice(input_grids)
        slots.append({
            "rows": rows,
            "cols": cols,
            "colors": len(colors),
            "n_train": len(task["train"]),
            "n_test": len(task["test"]),
            "anchor": anchor_id,
        })
    return slots


# ---------------------------------------------------------------------------
# Deduplication via embedding similarity
# ---------------------------------------------------------------------------

def _unit_rows(vectors) -> "np.ndarray":
    """Stack vectors into a float32 matrix with L2-normalised rows.

    Once rows are unit-length, cosine similarity is just a dot product, so an
    entire pairwise similarity matrix is one BLAS-backed matmul. The previous
    pure-Python implementation recomputed both vector norms on every pair —
    O(N^2) norm computations for O(N^2) comparisons.
    """
    m = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)     # guard against zero vectors
    return m / norms


def _embed(texts: list) -> "np.ndarray":
    """Embed texts and return an L2-normalised matrix, rows aligned to input order."""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    ordered = [e.embedding for e in sorted(resp.data, key=lambda x: x.index)]
    return _unit_rows(ordered)


def find_duplicate_clusters(descriptions: dict, threshold: float = DEDUP_THRESHOLD) -> list:
    """Return groups of task_ids whose descriptions are semantically near-duplicate.

    Embeds every description, computes the full pairwise cosine matrix in one
    matmul, then unions any pair at or above `threshold`. Returns groups of size >= 2.
    """
    task_ids = list(descriptions.keys())
    if len(task_ids) < 2:
        return []

    m = _embed([descriptions[tid] or "no description" for tid in task_ids])
    sim = m @ m.T
    # Upper triangle only, so each pair is considered once and self-similarity ignored.
    pairs = np.argwhere(np.triu(sim, k=1) >= threshold)

    n = len(task_ids)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj

    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(task_ids[i])

    return [g for g in groups.values() if len(g) >= 2]


def find_eval_similar(
    gen_descriptions: dict,
    eval_descriptions: dict,
    threshold: float = EVAL_SIMILARITY_THRESHOLD,
) -> dict:
    """Return {gen_task_id: (eval_task_id, similarity)} for generated tasks whose
    description is too close to any official eval task description.

    Uses a higher threshold than intra-set dedup because thematic overlap with the
    eval set is expected — only near-exact rule matches should be caught. See the
    EVAL_SIMILARITY_THRESHOLD comment for the calibration.

    Returns an empty dict if eval_descriptions is empty (cache not yet built).
    """
    if not eval_descriptions or not gen_descriptions:
        return {}

    gen_ids = list(gen_descriptions.keys())
    eval_ids = list(eval_descriptions.keys())

    gen_m = _embed([gen_descriptions[tid] or "no description" for tid in gen_ids])
    eval_m = _embed([eval_descriptions[tid] or "no description" for tid in eval_ids])

    sim = gen_m @ eval_m.T                  # (n_gen, n_eval)
    best_idx = sim.argmax(axis=1)
    best_sim = sim[np.arange(len(gen_ids)), best_idx]

    return {
        gen_ids[i]: (eval_ids[int(best_idx[i])], round(float(best_sim[i]), 4))
        for i in np.flatnonzero(best_sim >= threshold)
    }


# ---------------------------------------------------------------------------
# Parsing & validation
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
    return json.loads(text.strip())


def validate_tasks(tasks: dict) -> list:
    errors = []
    for task_id, task in tasks.items():
        if "train" not in task or "test" not in task:
            errors.append(f"{task_id}: missing 'train' or 'test' key")
            continue
        if len(task["train"]) < 2:
            errors.append(f"{task_id}: fewer than 2 training pairs")
        for split in ("train", "test"):
            for i, pair in enumerate(task[split]):
                for field in ("input", "output"):
                    grid = pair.get(field)
                    if not isinstance(grid, list) or not all(isinstance(row, list) for row in grid):
                        errors.append(f"{task_id} {split}[{i}].{field}: not a 2-D list")
                        continue
                    if len(grid) == 0:
                        errors.append(f"{task_id} {split}[{i}].{field}: empty grid")
                        continue
                    col_lens = {len(row) for row in grid}
                    if len(col_lens) != 1:
                        errors.append(f"{task_id} {split}[{i}].{field}: ragged rows")
                    for row in grid:
                        for cell in row:
                            if not isinstance(cell, int) or cell < 0 or cell > 9:
                                errors.append(
                                    f"{task_id} {split}[{i}].{field}: cell {cell!r} out of range 0-9"
                                )
    return errors


# ---------------------------------------------------------------------------
# Distribution stats
# ---------------------------------------------------------------------------

def invalid_task_ids(tasks: dict) -> set:
    """Task ids failing structural validation (ragged rows, out-of-range cells, ...).

    Malformed grids are worth removing rather than reporting: ARC harnesses commonly
    reject a whole batch when one grid is ragged, and a task that cannot be scored is
    not a task. Two ragged grids reached a scoring run before this was enforced.
    """
    return {tid for tid, task in tasks.items() if validate_tasks({tid: task})}


def compute_stats(tasks: dict) -> dict:
    train_counts, test_counts = [], []
    rows_list, cols_list, areas_list = [], [], []
    in_rows, in_cols, in_areas, out_areas = [], [], [], []
    colors_per_task = []

    for task in tasks.values():
        train_counts.append(len(task["train"]))
        test_counts.append(len(task["test"]))
        task_colors = set()
        for split in ("train", "test"):
            for pair in task[split]:
                for field in ("input", "output"):
                    grid = pair.get(field)
                    if not grid:
                        continue
                    r = len(grid)
                    c = len(grid[0]) if grid[0] else 0
                    rows_list.append(r)
                    cols_list.append(c)
                    areas_list.append(r * c)
                    # Inputs and outputs are tracked separately: the sampler sets the
                    # INPUT size, while output size is an emergent property of whatever
                    # rule the model invents. Pooling them yields a composite metric
                    # that cannot attribute a failure to either cause.
                    if field == "input":
                        in_rows.append(r)
                        in_cols.append(c)
                        in_areas.append(r * c)
                    else:
                        out_areas.append(r * c)
                    for row in grid:
                        task_colors.update(v for v in row if v != 0)
        colors_per_task.append(len(task_colors))

    def summarise(lst):
        if not lst:
            return {}
        return {
            "min": min(lst),
            "max": max(lst),
            "mean": round(statistics.mean(lst), 2),
            "median": statistics.median(lst),
            "stdev": round(statistics.stdev(lst), 2) if len(lst) > 1 else 0.0,
        }

    return {
        "num_tasks": len(tasks),
        "train_pairs_per_task": dict(sorted(Counter(train_counts).items())),
        "test_pairs_per_task": dict(sorted(Counter(test_counts).items())),
        "rows": summarise(rows_list),
        "cols": summarise(cols_list),
        "area": summarise(areas_list),           # pooled (kept for continuity with earlier runs)
        "input_rows": summarise(in_rows),
        "input_cols": summarise(in_cols),
        "input_area": summarise(in_areas),       # what the sampler actually controls
        "output_area": summarise(out_areas),     # emergent from the transformation rule
        "colors_per_task": summarise(colors_per_task),
    }


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

TOLERANCE = 0.40

def _pct_diff(gen_val, ref_val) -> float:
    if ref_val == 0:
        return 0.0
    return abs(gen_val - ref_val) / ref_val


def run_sanity_check(eval_stats: dict, gen_stats: dict) -> dict:
    checks = []

    def check_mean(metric: str, field: str):
        ref = eval_stats[metric][field]
        got = gen_stats[metric].get(field)
        if got is None:
            checks.append({"metric": f"{metric}.{field}", "status": "SKIP", "note": "no data"})
            return
        diff = _pct_diff(got, ref)
        status = "PASS" if diff <= TOLERANCE else "WARN"
        checks.append({
            "metric": f"{metric}.{field}",
            "status": status,
            "eval_value": ref,
            "generated_value": got,
            "pct_diff": round(diff * 100, 1),
            "note": f">{int(TOLERANCE*100)}% deviation from eval set" if status == "WARN" else "within tolerance",
        })

    # Only input dimensions are checked. That is what the sampler sets, and it is the
    # distribution we are trying to reproduce. Output size is emergent from whatever
    # rule the model invents and is deliberately left unconstrained — pooling it into
    # the area metric produced a composite that could not attribute a failure to
    # either cause. Raw output/pooled figures remain in generated_stats for reference.
    for field in ("mean", "median"):
        check_mean("input_area", field)
    check_mean("input_rows", "mean")
    check_mean("input_cols", "mean")

    eval_most_common = max(eval_stats["train_pairs_per_task"], key=eval_stats["train_pairs_per_task"].get)
    gen_dist = gen_stats["train_pairs_per_task"]
    gen_most_common = max(gen_dist, key=gen_dist.get) if gen_dist else None
    checks.append({
        "metric": "train_pairs_mode",
        "status": "PASS" if str(gen_most_common) == str(eval_most_common) else "WARN",
        "eval_value": eval_most_common,
        "generated_value": gen_most_common,
        "note": "most common training pair count",
    })

    gen_test_dist = gen_stats["test_pairs_per_task"]
    single_test_frac = gen_test_dist.get(1, 0) / max(sum(gen_test_dist.values()), 1)
    checks.append({
        "metric": "single_test_pair_fraction",
        "status": "PASS" if single_test_frac >= 0.80 else "WARN",
        "eval_value": ">=95%",
        "generated_value": f"{single_test_frac*100:.0f}%",
        "note": "fraction of tasks with exactly 1 test pair",
    })

    overall = "PASS" if all(c["status"] in ("PASS", "INFO") for c in checks) else "WARN"
    return {"overall": overall, "checks": checks}


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_separated(tasks: dict, dest_dir: Path) -> None:
    """Write one JSON file per task, clearing any files left by an earlier write.

    Stale files matter: a task removed by the convergence loop or a post-hoc repair
    would otherwise linger here and be loaded as if it were part of the set.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    for existing in dest_dir.glob("*.json"):
        if existing.stem not in tasks:
            existing.unlink()
    for task_id, task in tasks.items():
        (dest_dir / f"{task_id}.json").write_text(json.dumps(task, indent=2))


# ---------------------------------------------------------------------------
# Single task generation (called in parallel)
# ---------------------------------------------------------------------------

def generate_one(idx: int, slot: dict, avoid: list = None) -> dict:
    """Call the API for one task using a sampled slot. Returns parsed result or error info."""
    prompt = (PROMPT_TEMPLATE
              .replace("{target_rows}", str(slot["rows"]))
              .replace("{target_cols}", str(slot["cols"]))
              .replace("{target_colors}", str(slot["colors"]))
              .replace("{n_train}", str(slot["n_train"]))
              .replace("{n_test}", str(slot["n_test"])))
    if avoid:
        avoid_block = "\n".join(f"  - {m}" for m in avoid)
        prompt += (
            "\n\nIMPORTANT — diversity requirement: the transformation rule you invent must be "
            "distinctly different in its core mechanism from each of the following rules already "
            "used in this task set. A rule that is a minor variation, rename, or reframing of one "
            "of these will be rejected:\n" + avoid_block
        )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content
        parsed = extract_json(raw)
        tasks = parsed.get("tasks", {})
        descriptions = parsed.get("descriptions", {})
        return {"idx": idx, "tasks": tasks, "descriptions": descriptions, "error": None}
    except Exception as e:
        return {"idx": idx, "tasks": {}, "descriptions": {}, "error": str(e)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MAX_RETRIES = 3       # max retry rounds for initial generation failures
# Backstop only — the convergence loop exits as soon as nothing needs removing, so
# an over-generous ceiling costs nothing while an under-set one silently ships a set
# with residual clusters. Observed rounds-to-converge: 0/1/1/2/7 at N=16/32/64/100/400,
# with removal rates 0%/9.4%/7.8%/22%/46%. The rate has risen at every scale measured,
# so this scales with log2(N) rather than being fixed. Actual termination is driven by
# convergence or the stall detector below, not by this number.
def default_max_rounds(n: int) -> int:
    return max(8, math.ceil(3 * math.log2(max(n, 2))))

MAX_STALL_ROUNDS = 3  # consecutive non-decreasing removal counts before giving up

MAX_DEDUP_ROUNDS = default_max_rounds(N)

def main() -> int:
    global N, MAX_DEDUP_ROUNDS
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=None, help=f"number of tasks to generate (default {N})")
    ap.add_argument("--max-rounds", type=int, default=None, help=f"max novelty convergence rounds (default {MAX_DEDUP_ROUNDS})")
    args = ap.parse_args()
    if args.n is not None:
        N = args.n
        MAX_DEDUP_ROUNDS = default_max_rounds(N)
    if args.max_rounds is not None:
        MAX_DEDUP_ROUNDS = args.max_rounds

    eval_tasks = load_eval_tasks()
    eval_stats = compute_stats(eval_tasks)

    eval_descriptions = {}
    if not ENABLE_EVAL_SIMILARITY:
        print("Eval similarity check DISABLED (threshold needs recalibration).", file=sys.stderr)
    elif EVAL_DESCRIPTIONS_PATH.exists():
        eval_descriptions = json.loads(EVAL_DESCRIPTIONS_PATH.read_text()).get("descriptions", {})
        print(f"Loaded {len(eval_descriptions)} eval task descriptions for cross-similarity check.", file=sys.stderr)
    else:
        print(
            f"WARNING: {EVAL_DESCRIPTIONS_PATH} not found — skipping eval similarity check. "
            "Run describe_eval_tasks.py first to enable it.",
            file=sys.stderr,
        )
    if not ENABLE_DEDUP:
        print("Deduplication DISABLED (pending slot-recycling fix).", file=sys.stderr)

    pending_slots = sample_joint_slots(eval_tasks, N)

    print(f"Generating {N} tasks (N=1 per call, max_workers={MAX_WORKERS}, model={MODEL})...", file=sys.stderr)
    t0 = time.time()

    all_tasks = {}
    all_descriptions = {}
    slot_by_task = {}         # task_id -> slot it was generated from (enables slot recycling)
    anchor_counts = Counter()
    total_failed = 0
    call_idx = 0

    for attempt in range(MAX_RETRIES + 1):
        needed = N - len(all_tasks)
        if needed == 0:
            break
        if attempt > 0:
            print(f"\nRetry round {attempt}: {needed} slot(s) still needed...", file=sys.stderr)
            pending_slots = sample_joint_slots(eval_tasks, needed)

        workers = min(needed, MAX_WORKERS)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(generate_one, call_idx + i, slot): slot
                for i, slot in enumerate(pending_slots[:needed])
            }
            call_idx += needed
            for future in as_completed(futures):
                slot = futures[future]
                result = future.result()
                if result["error"]:
                    print(f"  [{result['idx']}] ERROR: {result['error']}", file=sys.stderr)
                    total_failed += 1
                else:
                    for task_id, task in result["tasks"].items():
                        if task_id in all_tasks:
                            print(f"  [{result['idx']}] DUPLICATE task_id '{task_id}' — skipping", file=sys.stderr)
                            total_failed += 1
                        else:
                            all_tasks[task_id] = task
                            all_descriptions[task_id] = result["descriptions"].get(task_id, "")
                            anchor_counts[slot["anchor"]] += 1
                            slot_by_task[task_id] = slot
                    print(
                        f"  [{result['idx']}] done — {list(result['tasks'].keys())}"
                        f"  (anchor: {slot['anchor']}, {slot['rows']}×{slot['cols']},"
                        f" {slot['colors']}c, {slot['n_train']}tr/{slot['n_test']}te)",
                        file=sys.stderr,
                    )

    elapsed = time.time() - t0
    print(f"\n{len(all_tasks)}/{N} tasks collected in {elapsed:.1f}s ({total_failed} failed calls)", file=sys.stderr)
    if len(all_tasks) < N:
        print(f"WARNING: only {len(all_tasks)} tasks after {MAX_RETRIES} retry rounds.", file=sys.stderr)

    if not all_tasks:
        print("No tasks generated.", file=sys.stderr)
        return 1

    # --- Merged novelty convergence loop ---
    # Each round applies BOTH filters and removes their union, then regenerates:
    #   intra-set dedup  — generated tasks too close to each other (DEDUP_THRESHOLD)
    #   eval similarity  — generated tasks too close to an official eval task
    # Loops until both are clean AND we hold exactly N tasks. Running them as two
    # sequential single-pass stages (the previous design) let each undo the other's
    # guarantee: eval-sim replacements were never cluster-checked, and dedup
    # replacements were never eval-checked.
    #
    # Removed tasks have their SLOTS RECYCLED rather than redrawn. Removal correlates
    # with slot properties (clusterable themes come from particular anchors), so
    # refilling with fresh uniform draws leaves the survivors skewed and drags the
    # realised distribution off-target. Recycling pins the slot distribution to the
    # round-one draw no matter how many rounds fire; the avoidance list is what
    # actually supplies novelty pressure.
    #
    # Eval descriptions are NEVER placed in the regeneration prompt — showing the
    # generator real eval rules would leak the very set we are trying to stay
    # independent of. Eval similarity only ever removes; it never instructs.

    dedup_clusters_initial = []
    dedup_clusters_final = []
    dedup_removed = []
    dedup_replaced = 0
    eval_similar_flagged = {}
    eval_similar_removed = []
    eval_similar_replaced = 0
    round_history = []

    # Accumulates every description ever generated, kept or removed, so the model is
    # also told not to re-invent rules already discarded in earlier rounds.
    seen_removed_descs = []

    def _build_avoid() -> list:
        kept = [d for d in all_descriptions.values() if d]
        combined = kept + [d for d in seen_removed_descs if d]
        return combined[:N]          # ceiling of N (N-1 kept on the first regen)

    converged = False
    stalled = False
    stalled_rounds = 0
    for dedup_round in range(1, MAX_DEDUP_ROUNDS + 1):
        clusters = (
            find_duplicate_clusters(all_descriptions, threshold=DEDUP_THRESHOLD)
            if ENABLE_DEDUP else []
        )
        flagged = (
            find_eval_similar(all_descriptions, eval_descriptions, EVAL_SIMILARITY_THRESHOLD)
            if (ENABLE_EVAL_SIMILARITY and eval_descriptions) else {}
        )
        if dedup_round == 1:
            dedup_clusters_initial = [list(g) for g in clusters]
        dedup_clusters_final = [list(g) for g in clusters]
        eval_similar_flagged.update(flagged)

        malformed = invalid_task_ids(all_tasks)

        to_remove = set()
        for group in clusters:
            print(f"  Cluster ({len(group)}): keeping '{group[0]}', removing {group[1:]}", file=sys.stderr)
            to_remove.update(group[1:])
        for gid, (eid, sim) in flagged.items():
            print(f"  Eval-similar: {gid} ↔ eval:{eid} ({sim}) — removing", file=sys.stderr)
        to_remove.update(flagged.keys())
        for tid in malformed:
            print(f"  Malformed: {tid} — failed structural validation, removing", file=sys.stderr)
        to_remove.update(malformed)

        if not to_remove and len(all_tasks) >= N:
            converged = True
            break

        dedup_removed.extend(t for t in to_remove if t not in flagged)
        eval_similar_removed.extend(t for t in to_remove if t in flagged)
        for tid in to_remove:
            seen_removed_descs.append(all_descriptions.get(tid, ""))
            del all_tasks[tid]
            del all_descriptions[tid]

        needed = N - len(all_tasks)
        avoid = _build_avoid()

        # Recycle the slots of removed tasks; top up only to cover parse failures.
        slots = [slot_by_task[tid] for tid in to_remove if tid in slot_by_task]
        recycled = len(slots)
        if len(slots) < needed:
            slots += sample_joint_slots(eval_tasks, needed - len(slots))
        slots = slots[:needed]

        print(
            f"\n  Round {dedup_round}: {len(to_remove)} removed "
            f"({len(clusters)} cluster(s), {len(flagged)} eval-similar, {len(malformed)} malformed); "
            f"regenerating {needed} ({recycled} recycled slot(s), "
            f"{len(avoid)} avoidance hints)...",
            file=sys.stderr,
        )

        if needed > 0:
            with ThreadPoolExecutor(max_workers=min(needed, MAX_WORKERS)) as executor:
                futures = {
                    executor.submit(generate_one, call_idx + i, slot, avoid): slot
                    for i, slot in enumerate(slots)
                }
                call_idx += needed
                for future in as_completed(futures):
                    slot = futures[future]
                    result = future.result()
                    if result["error"]:
                        print(f"  [r{dedup_round}] ERROR: {result['error']}", file=sys.stderr)
                        total_failed += 1
                    else:
                        for task_id, task in result["tasks"].items():
                            if task_id in all_tasks:
                                print(f"  [r{dedup_round}] DUPLICATE id '{task_id}' — skipping", file=sys.stderr)
                                total_failed += 1
                            else:
                                all_tasks[task_id] = task
                                all_descriptions[task_id] = result["descriptions"].get(task_id, "")
                                anchor_counts[slot["anchor"]] += 1
                                slot_by_task[task_id] = slot
                                dedup_replaced += 1
                                print(f"  [r{dedup_round}] {task_id} (anchor: {slot['anchor']})", file=sys.stderr)

        prev_removed = round_history[-1]["removed"] if round_history else None
        round_history.append({
            "round": dedup_round,
            "clusters_found": len(clusters),
            "eval_similar_found": len(flagged),
            "malformed_found": len(malformed),
            "removed": len(to_remove),
            "recycled_slots": recycled,
            "regenerated": needed,
            "tasks_after": len(all_tasks),
        })
        print(f"  After round {dedup_round}: {len(all_tasks)} tasks.", file=sys.stderr)

        # Stall detection. Removals normally decay geometrically (185/88/41/16/8/1/1
        # at N=400). A run of non-decreasing removal counts means a slot keeps
        # regenerating into the same cluster, so further rounds burn API calls
        # without converging. This terminates on observed behaviour rather than on
        # a predicted round count.
        if prev_removed is not None and len(to_remove) >= prev_removed:
            stalled_rounds += 1
            if stalled_rounds >= MAX_STALL_ROUNDS:
                print(
                    f"WARNING: removals did not decrease over {MAX_STALL_ROUNDS} consecutive "
                    f"rounds (last: {prev_removed} then {len(to_remove)}) — stopping early.",
                    file=sys.stderr,
                )
                stalled = True
                break
        else:
            stalled_rounds = 0

    if not (ENABLE_DEDUP or ENABLE_EVAL_SIMILARITY):
        pass
    elif converged:
        print(f"  Converged after {len(round_history)} round(s) — {len(all_tasks)} tasks, 0 clusters, 0 eval-similar.", file=sys.stderr)
    elif stalled:
        print(f"WARNING: stopped after {len(round_history)} round(s) on stall detection "
              f"({len(dedup_clusters_final)} cluster(s) remaining).", file=sys.stderr)
    else:
        print(f"WARNING: hit the {MAX_DEDUP_ROUNDS}-round ceiling "
              f"({len(dedup_clusters_final)} cluster(s) remaining).", file=sys.stderr)

    if len(all_tasks) != N:
        print(f"WARNING: ended with {len(all_tasks)} tasks (target {N}).", file=sys.stderr)

    errors = validate_tasks(all_tasks)
    if errors:
        print(f"Validation: {len(errors)} issue(s)", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
    else:
        print(f"All {len(all_tasks)} tasks passed structural validation.", file=sys.stderr)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    gen_dir = GENERATIONS_DIR / f"gen_{run_ts}"
    gen_dir.mkdir(parents=True, exist_ok=True)

    tasks_path = gen_dir / "tasks.json"
    tasks_path.write_text(json.dumps(all_tasks, indent=2))
    print(f"Saved {len(all_tasks)} tasks → {tasks_path}", file=sys.stderr)

    write_separated(all_tasks, gen_dir / "separated")
    print(f"Separated tasks → {gen_dir}/separated/", file=sys.stderr)

    gen_stats = compute_stats(all_tasks)
    sanity = run_sanity_check(eval_stats, gen_stats)

    unique_anchors = len(anchor_counts)
    max_reuse = max(anchor_counts.values(), default=0)
    anchor_summary = {
        "unique_anchors_used": unique_anchors,
        "total_eval_tasks": len(eval_tasks),
        "anchor_coverage_pct": round(unique_anchors / len(eval_tasks) * 100, 1),
        "max_reuse_count": max_reuse,
        "reused_anchors": {k: v for k, v in anchor_counts.items() if v > 1},
    }

    print(f"\n--- Anchor Distribution ---", file=sys.stderr)
    print(f"  {unique_anchors} unique eval tasks used as anchors out of {len(eval_tasks)}"
          f" ({anchor_summary['anchor_coverage_pct']}% coverage)", file=sys.stderr)
    if max_reuse > 1:
        print(f"  Max reuse: {max_reuse}x  ({len(anchor_summary['reused_anchors'])} anchors reused)", file=sys.stderr)
    else:
        print(f"  No anchors reused — all {unique_anchors} anchors distinct", file=sys.stderr)

    report = {
        "run_timestamp": run_ts,
        "model": MODEL,
        "mode": f"joint-sampled N=1 x{N}",
        "elapsed_s": round(elapsed, 1),
        "num_tasks_generated": len(all_tasks),
        "num_failed_calls": total_failed,
        "overall": sanity["overall"],
        "checks": sanity["checks"],
        "dedup_summary": {
            "threshold": DEDUP_THRESHOLD,
            "enabled": ENABLE_DEDUP,
            "initial_clusters": dedup_clusters_initial,
            "tasks_removed": dedup_removed,
            "replacements_generated": dedup_replaced,
            "residual_clusters": dedup_clusters_final,
            "max_rounds": MAX_DEDUP_ROUNDS,
            "rounds_used": len(round_history),
            "converged": converged,
            "stalled": stalled,
            "stall_limit": MAX_STALL_ROUNDS,
            "round_history": round_history,
        },
        "slot_provenance": {tid: slot_by_task.get(tid) for tid in all_tasks},
        "eval_similarity_summary": {
            "threshold": EVAL_SIMILARITY_THRESHOLD,
            "eval_descriptions_available": bool(eval_descriptions),
            "tasks_flagged": eval_similar_flagged,
            "tasks_removed": eval_similar_removed,
            "replacements_generated": eval_similar_replaced,
        },
        "anchor_summary": anchor_summary,
        "eval_stats": eval_stats,
        "generated_stats": gen_stats,
        "validation_errors": errors,
        "transformation_descriptions": {
            task_id: all_descriptions.get(task_id, "(no description returned)")
            for task_id in all_tasks
        },
    }

    sanity_path = gen_dir / "sanity_check.json"
    sanity_path.write_text(json.dumps(report, indent=2))
    print(f"Sanity check ({sanity['overall']}) → {sanity_path}", file=sys.stderr)

    print("\n--- Eval Similarity Summary ---", file=sys.stderr)
    if eval_descriptions:
        print(f"  Tasks flagged as too similar to eval: {len(eval_similar_flagged)}", file=sys.stderr)
        print(f"  Tasks removed: {len(eval_similar_removed)}", file=sys.stderr)
        print(f"  Replacements generated: {eval_similar_replaced}", file=sys.stderr)
    else:
        print("  SKIPPED — run describe_eval_tasks.py to enable this check.", file=sys.stderr)

    print("\n--- Dedup Summary ---", file=sys.stderr)
    print(f"  Initial clusters found: {len(dedup_clusters_initial)}", file=sys.stderr)
    print(f"  Tasks removed: {len(dedup_removed)}", file=sys.stderr)
    print(f"  Replacements generated: {dedup_replaced}", file=sys.stderr)
    print(f"  Residual clusters after dedup: {len(dedup_clusters_final)}", file=sys.stderr)

    print("\n--- Sanity Check Summary ---", file=sys.stderr)
    for c in sanity["checks"]:
        flag = "✓" if c["status"] == "PASS" else "⚠"
        print(f"  {flag} {c['metric']}: eval={c.get('eval_value','')}  gen={c.get('generated_value','')}  ({c['note']})", file=sys.stderr)

    print("\n--- Transformation Descriptions ---", file=sys.stderr)
    for task_id, desc in report["transformation_descriptions"].items():
        print(f"  {task_id}: {desc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
