# arc-task-gen

Generate original ARC-AGI-1-style tasks whose distribution matches the public
ARC-AGI-1 evaluation set, for evaluating a model on problems it cannot have memorised.

The public ARC-AGI-1 evaluation set appears in web-scraped training corpora, so a
model's score on it is an upper bound on genuine few-shot rule induction. This tool
builds a private set with the same measurable properties, so the two scores are
comparable.

`tasks.json` uses the standard ARC `{"train": [...], "test": [...]}` shape, so any
ARC harness can consume it.

## Install

```bash
pixi install
export OPENAI_API_KEY=sk-...
```

Works against any OpenAI-compatible chat-completions endpoint:

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1   # vLLM, Ollama, LM Studio, ...
export ARCGEN_MODEL=your-model
export ARCGEN_EMBED_MODEL=your-embedding-model
```

The generation model has to emit strict JSON while holding five numeric constraints at
once. Parse failures are retried, so a weaker model costs extra calls. A 400-task run
on `gpt-5.6` logged 27 parse failures across 758 calls.

## Use

```bash
python describe_eval_tasks.py            # once, ~5 min, caches eval rule descriptions
python generate_tasks.py --n 400         # generate a task set matched on size/colour/pair-count
```

The ARC-AGI-1 evaluation set downloads automatically on first run.

`describe_eval_tasks.py` is optional. It enables the check that removes generated tasks
too similar to real evaluation tasks; without it that filter is skipped with a warning.

For tighter matching — each task conditioned on one public eval task's shape sequence and
Chollet category, optionally stratified evenly across transformation mechanics — label the
eval set once and use `generate_tasks_stratified.py` instead of `generate_tasks.py`:

```bash
python label_eval_tasks.py                                            # once, labels category + mechanic
python generate_tasks_stratified.py --n 400                           # category-matched
python generate_tasks_stratified.py --mechanics all --per-mechanic 25 # mechanic-stratified
```

It wraps `generate_tasks.py` and reuses its whole generation loop, so everything below
still applies. `--dry-run-plan` inspects the sampled anchor/mechanic plan without calling
the API.

Output lands in `data/generations/gen_<timestamp>/` (`data/generations_stratified/...` for
the latter):

| file | contents |
|---|---|
| `tasks.json` | all tasks, keyed by id |
| `separated/<id>.json` | one file per task, loads in the [ARC testing interface](https://github.com/fchollet/ARC-AGI) |
| `sanity_check.json` | distribution comparison, rule descriptions, slot provenance, per-round filter history |

## Data folder

| path | contents | committed? |
|---|---|---|
| `data/arc_agi_eval.json` | the 400-task ARC-AGI-1 evaluation set | no — downloaded on first run |
| `data/arc_agi_eval_descriptions.json` | one rule description per eval task | no — built by `describe_eval_tasks.py` |
| `data/arc_agi_eval_categories.json` | one Chollet category + mechanic per eval task | no — built by `label_eval_tasks.py` |
| `data/generations/`, `data/generations_stratified/` | your generated task sets, one timestamped folder per run | no |

Nothing here is committed: the eval set and its labels are cheaply rebuilt (downloaded or
regenerated), and generated task sets are deliberately kept private — the benchmark's
value is that no model has seen the tasks, and publishing them puts them into web-scraped
training corpora and burns them.

## Measured output

A 400-task run on `gpt-5.6` (the default generation model; see Install) against the
400-task public evaluation set:

| property | Public Evaluation Set | generated |
|---|---|---|
| input area, mean | 226.92 | 224.70 |
| input area, median | 144.00 | 144.00 |
| input rows, mean | 13.23 | 13.19 |
| input cols, mean | 13.70 | 13.67 |
| colours, mean | 5.35 | 5.24 |
| demonstration pairs ≥ 4 | 34.2% | 34.8% |
| test pairs = 2 | 4.8% | 5.0% |

## How it works

### One task per call

Each API call requests exactly one task, with up to `MAX_WORKERS` calls in flight at
once. This is a deliberate constraint: asking a single
call to return several tasks at once measurably shrinks them, because the model spends
roughly a fixed effort budget per response and divides it across however many tasks it
was asked for. In a `gpt-5.6` test requesting 32 tasks per response, mean grid area was
14 cells, against 208 cells for the official evaluation set, and the ~225-cell mean
this tool gets at one task per call (see Measured output).

### Joint constraint sampling

Each call is assigned a *slot* drawn wholesale from a single randomly chosen evaluation
task:

```python
{'rows': 26, 'cols': 18, 'colors': 3, 'n_train': 4, 'n_test': 1, 'anchor': '3490cc26'}
```

### Novelty filtering

Duplicates are caught by semantic similarity. The model writes a
one-sentence rule description for each task; those descriptions are embedded and compared
by cosine similarity. Any task scoring at or above `DEDUP_THRESHOLD` against another
generated task is regenerated.

A second filter compares the same descriptions against descriptions of the real
evaluation tasks and regenerates anything at or above `EVAL_SIMILARITY_THRESHOLD`. It
needs `describe_eval_tasks.py` to have run.

### Termination

The loop exits when nothing needs removing. `MAX_DEDUP_ROUNDS` defaults to
`max(8, ceil(3·log2(N)))` as a backstop, plus a stall detector that stops after three
rounds without a decrease in removals.

### Validation

Structural checks (rectangular grids, cell values 0–9) run inside the loop, and
malformed tasks are removed and regenerated alongside duplicates.

## Configuration

| Name | Default | Meaning |
|---|---|---|
| `--n` | 32 | tasks to generate |
| `--max-rounds` | `max(8, ceil(3·log2(N)))` | novelty-loop ceiling |
| `MAX_WORKERS` | 64 | concurrent calls |
| `DEDUP_THRESHOLD` | 0.80 | min similarity to another generated task at which one is regenerated |
| `EVAL_SIMILARITY_THRESHOLD` | 0.92 | min similarity to a real evaluation task at which a generated task is regenerated |
| `MAX_STALL_ROUNDS` | 3 | rounds without progress before stopping |

## Caveats

### Solvability is not verified programmatically

Structural validation checks shape and colour range. Nothing confirms a rule is
inferable from its training pairs alone. In one hand-checked blind sample of 20 tasks,
19 were solvable. Sample your own before trusting a score.

### Duplicate detection is bounded by wording

Two tasks sharing a mechanism but described differently fall below
`DEDUP_THRESHOLD` and both survive. Which task survives a cluster is arbitrary: the
first in union-find order.

### Generated task ids leak their rules

Names like `diagonal_corner_echo` come from the model. Renumber before showing tasks to
a human solver, and note that `sanity_check.json` contains the intended rule for every
task.

## Attribution

The ARC-AGI-1 evaluation set is downloaded from
[fchollet/ARC-AGI](https://github.com/fchollet/ARC-AGI) (Apache 2.0) and used as the
reference distribution. It is not redistributed here.

MIT licensed.
