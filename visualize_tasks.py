"""Render generated ARC tasks as PNG images so they're easy to eyeball.

Usage:
  python visualize_tasks.py <path> [--out DIR]

<path> can be:
  - a generation directory (data/generations/gen_<ts>/)   -> renders every task in tasks.json
  - a tasks.json file                                      -> renders every task in it
  - a single separated/<task_id>.json file                 -> renders that one task

One PNG per task is written to --out (default: <input_dir>/images/), named <task_id>.png.
Each row is a train or test pair; input is on the left, output on the right. Test pairs
get a red column label so they're visually distinct from train pairs.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Standard ARC-AGI palette (0-9), matching the official testing interface.
ARC_COLORS = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
]
ARC_CMAP = ListedColormap(ARC_COLORS)
ARC_NORM = BoundaryNorm(list(range(11)), ARC_CMAP.N)


def load_tasks(path: Path) -> dict:
    """Return {task_id: {"train": [...], "test": [...]}} regardless of which of the
    three accepted input shapes `path` points to."""
    if path.is_dir():
        tasks_json = path / "tasks.json"
        if tasks_json.exists():
            return json.loads(tasks_json.read_text())
        return {p.stem: json.loads(p.read_text()) for p in sorted(path.glob("*.json"))}

    data = json.loads(path.read_text())
    if "train" in data and "test" in data:
        return {path.stem: data}
    return data


def draw_grid(ax, grid: list, label: str, label_color: str = "black") -> None:
    ax.imshow(grid, cmap=ARC_CMAP, norm=ARC_NORM)
    ax.set_xticks([x - 0.5 for x in range(1, len(grid[0]))], minor=False)
    ax.set_yticks([y - 0.5 for y in range(1, len(grid))], minor=False)
    ax.grid(which="major", color="#444444", linewidth=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    ax.set_title(label, fontsize=9, color=label_color)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(label_color)
        spine.set_linewidth(1.5)


def render_task(task_id: str, task: dict, dest: Path) -> None:
    pairs = [("train", i, p) for i, p in enumerate(task["train"])]
    pairs += [("test", i, p) for i, p in enumerate(task["test"])]

    fig, axes = plt.subplots(
        len(pairs), 2, figsize=(4, 2.2 * len(pairs)), squeeze=False
    )
    for row, (split, i, pair) in enumerate(pairs):
        color = "#c0392b" if split == "test" else "black"
        draw_grid(axes[row][0], pair["input"], f"{split}[{i}] input", color)
        draw_grid(axes[row][1], pair["output"], f"{split}[{i}] output", color)

    fig.suptitle(task_id, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / f"{task_id}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  {task_id} -> {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path, help="generation dir, tasks.json, or a single task json file")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: <input_dir>/images/)")
    args = ap.parse_args()

    if not args.path.exists():
        print(f"Path not found: {args.path}", file=sys.stderr)
        return 1

    tasks = load_tasks(args.path)
    if not tasks:
        print("No tasks found.", file=sys.stderr)
        return 1

    dest = args.out or (args.path if args.path.is_dir() else args.path.parent) / "images"
    print(f"Rendering {len(tasks)} task(s) -> {dest}/")
    for task_id, task in tasks.items():
        render_task(task_id, task, dest)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
