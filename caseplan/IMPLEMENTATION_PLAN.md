# caseplan design

## Goal

A small toolset for creating, tracking, and recording the provenance of
Hermes-3 study cases without a hand-maintained lab log.

Design premises, from how these studies are actually run:

- A study is a directory holding a family of related cases.
- Cases differ from a parent case by a small but arbitrary set of BOUT options.
- Many cases are managed at once, so the natural interface is a **table**.
- Case *results* are discussed in analysis scripts, **not** in the plan. The
  plan is only about assembling cases, tracking their status, and provenance.
- Case state is human-owned; it is never inferred from logs.
- Run evidence (progress, provenance, liveness) is extracted from files/logs.
- Existing study directories must be bootstrappable into the tool.

## Core design: one CSV, ownership by column

Everything the user reads and edits lives in a single **`cases.csv`**. It is
both the human-owned plan and the tool-refreshed status table. This deliberately
merges what an earlier draft split across `plan.md` + `index.csv`; the merge is
safe only because ownership is enforced **per column**:

- **Human columns** — authoritative, never overwritten by the tool:
  `case, state, from, restart, hermes_build, changed_options, notes`.
- **Tool columns** — rewritten on every `index`, never hand-edited:
  `running, exists, final_t, runtime`.

`index` reconciles the file by a keyed merge on `case`: human columns are kept
verbatim, tool columns are recomputed from disk, and any run directory found on
disk without a row is appended. Writes are atomic (temp file + `os.replace`)
with a `cases.csv.bak` backup, so the file the user also hand-edits can never be
half-written or silently lost.

Full provenance that does not belong in a narrow table lives in a tool-owned
**`case.json`** per run directory.

### Why not Markdown / a database

Markdown (`plan.md`) was the original source format, chosen so results could be
discussed inline. That need moved to analysis scripts, leaving a tabular,
many-row workflow that CSV serves far better (copy-a-row parameter scans, pandas
/ spreadsheet viewing, `git` diffs). A database was rejected: plain text files
diff, version, and survive tooling changes.

The one field CSV hosts awkwardly is `changed_options` (nested/multi-valued); it
is carried as a single `;`-delimited cell, which is editable if slightly clumsy.

## Files and ownership

`cases.csv`

- Human-owned for its human columns; tool-owned for its four tool columns.
- Read by `gen` (case definitions) and by `index` (merge target).
- Written only by `index` (tool columns + appended rows) and `init`
  (first creation). Never rewritten wholesale once it exists.

`case.json` (one per run directory)

- Tool-owned. Two blocks with different lifecycles:
  - `generation`: written **once** by `casegen` — how the case was made
    (`from`, `restart`, resolved source, grid, `hermes_build`,
    `changed_options`, copied files, applied changes, timestamp, tool version).
    Records a historical event, so it cannot go stale.
  - `evidence`: rewritten by `index` every run — provenance and progress
    (commits, run binary, final_t, runtime, output counts, mtimes, hints,
    `is_running`, `indexed_at`).
- `index` preserves `generation` and replaces only `evidence`.
- Legacy cases discovered by `init` have no real `generation` block (they were
  not made by `casegen`); only `evidence` is written for them.

Dropped from the earlier design: `plan.md`, `.status` (state now lives only in
the `state` column), and the derived columns `state_source`, `outputs`, `path`.

## Columns

| Column | Owner | Meaning |
|---|---|---|
| `case` | human | run directory name; the merge key |
| `state` | human | one state word (below) |
| `running` | tool | live process detected for this case |
| `exists` | tool | run directory exists on disk |
| `from` | human | source case/template for inputs |
| `restart` | human | ``/`caseX`/`caseX:append`, or a bare mode word |
| `hermes_build` | human | Hermes-3 build for this case; hook for the launcher |
| `changed_options` | human | `;`-list of `section:key=value` |
| `notes` | human | short setup notes |
| `final_t` | tool | last simulated time from the log |
| `runtime` | tool | summed wall time from the log |

### State vocabulary

`planned`, `unfinished`, `finished`, `crashed`, `stuck`, `bad`, `ignore`,
`unknown`. `running` is intentionally **not** a state word: liveness is a
machine-detected column, so it is kept out of the human judgement.

### `restart` encoding

One cell carries both the old `restart_from` and `restart_mode`:

- empty / `scratch` → scratch (no restart files copied);
- `caseX` → restart from `caseX` (copy its restart files);
- `caseX:append` → restart from `caseX` and append output on the next run;
- bare `restart` / `restart_append` → mode known, source not yet filled. This
  is what `init` infers from a case-name suffix; `casegen` demands a source
  before it will regenerate such a case.

The grid is deliberately **not** a column: it is a `BOUT.inp` setting, so it is
changed through `changed_options` like any other option.

### `changed_options` encoding

`section:key=value` items separated by `;`, e.g.
`d:gradient_ceiling_D=0.025; d:neutral_lmax=1.0`. Top-level (sectionless) keys
are written bare (`nout=600`). A non-empty item without `=` is an error, not
silently dropped.

## Commands

### `caseplan init STUDY`

Bootstrap `cases.csv` for an existing study.

- Discover case directories (modern `runs/` and/or `--legacy-flat` children).
- Choose a base (central-most by option diff, or `--base`).
- Infer each case's `from` and `changed_options` against that base
  (`--no-smart-diff` to only list observed cases), the `restart` mode from the
  case-name suffix, and fill tool columns from evidence.
- Refuse to overwrite an existing `cases.csv` without `--force`.

Inference is a starting point; after review the file is human-owned.

### `caseplan index STUDY`

Refresh status.

- Read `cases.csv`; discover run directories.
- For each row with a directory: recompute evidence, rewrite its `case.json`
  `evidence` block, update the four tool columns.
- For each row without a directory (a planned/queued case): clear tool columns.
- Append any discovered directory that has no row.
- Write `cases.csv` atomically (+ `.bak`) and print a compact table.

Non-goals (v1): never infer `state` from logs; never launch/queue runs.

### `caseplan gen cases.csv`

Create or update cases from the rows (casegen). Paranoid by default; see Safety.
`--dry-run` prints every copy and the exact `BOUT.inp` diff. `--case NAME`
restricts to one row. A pure scratch row with no `changed_options` is skipped
(it would be an exact copy of its parent).

## Running detection

State is human-owned, but *liveness* is machine-detected and is the column the
user most relies on (manual launches on a workstation, easy to lose track of).

`index` scans `/proc` once and builds the set of resolved data directories of
live processes carrying a BOUT `-d <datadir>` argument. A case is `running` iff
its resolved directory is in that set. The launch form is

```
taskset -c <cores> mpirun -np 10 --bind-to=none $hermes/hermes-3 -d <casedir>
```

Matching is on the exact resolved `-d` path, never a substring — essential given
prefix-sharing names (`st40fllrb4` is a substring of `st40fllrb4fb`). Reading
live cmdlines directly also sidesteps stale/recycled PID files.

## Safety (casegen)

The safe path is to create new cases, never to mutate expensive results.
`casegen` refuses unless the matching flag is given:

- existing generated case without outputs → `--force-existing` / `--update-inputs-only`;
- directory containing outputs (`BOUT.log.*`, `BOUT.dmp.*.nc`,
  `BOUT.restart.*.nc`, `BOUT.squash.nc`) → `--allow-output-dir`;
- directory without `case.json` (hand-made) → `--overwrite-handmade`.

Copy policy excludes run products, PID/scheduler markers, and monitor plots when
copying from `from`; restart files are copied only from the `restart` source
when the mode requires them. `BOUT.inp` edits are minimal-diff (only the matched
value changes, comments preserved) and written atomically after optional
`boutdata.BoutOptionsFile` validation; if validation fails the original is left
untouched.

## Package layout

```
caseplan/
  IMPLEMENTATION_PLAN.md
  README.md
  caseplan/
    __init__.py
    cli.py          # argparse entry point: init / index / gen
    casesfile.py    # CaseRow, cases.csv read/write/merge, restart/changed_options parsing
    index.py        # evidence extraction, running detection, case.json, merge
    generate.py     # casegen: create/update cases from rows
    init.py         # bootstrap cases.csv
    boutinp.py      # minimal-edit BOUT.inp reader/writer + boutdata validation
    filesystem.py   # copy policy, output detection, run discovery
  tests/
    test_casesfile.py
    test_boutinp.py
    test_generate.py
    test_index.py
```

Standard library only (`argparse`, `csv`, `dataclasses`, `datetime`, `pathlib`,
`re`, `shutil`), with `boutdata` used only as an optional validator.

## Status

CSV design implemented and validated end-to-end on a real legacy study;
`init` → `index` → `gen` all work, human columns survive re-index, running
detection avoids the prefix trap, writes are atomic with `.bak`. Test suite
passing.

## Later

- A separate launch/queue stage (`gen` only assembles today). `hermes_build`
  and `restart` are the hooks it will read.
- Scheduler-based liveness beyond local `/proc` validation, if runs ever move
  off the workstation.
