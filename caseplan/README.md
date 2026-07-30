# caseplan

A small toolset for assembling, running-tracking, and recording the provenance
of Hermes-3 study cases from a single **`cases.csv`** — no hand-maintained lab
log, no per-case prose. Case *results* belong in your analysis scripts;
`caseplan` is only about putting cases together, seeing their status, and
knowing where they came from.

## The model

- **`cases.csv`** — the one file you live in. It is *both* the human-owned plan
  (case definitions) *and* the tool-refreshed status table. The two never fight
  because ownership is **by column**:

  | You own (never overwritten) | Tool owns (rewritten each `index`) |
  |---|---|
  | `case, state, from, restart, hermes_build, changed_options, notes` | `running, exists, final_t, runtime` |

- **`case.json`** — one per run directory, tool-owned. Holds the *full*
  provenance the CSV omits: Hermes/BOUT commits, run binary, output counts,
  mtimes, the write-once `generation` record of how the case was made, and an
  `evidence` block refreshed every `index`. You never edit it by hand.

State is human-owned and lives only in the `state` column; it is **never**
inferred from logs. Logs and PID files populate the tool columns / `case.json`
only.

## Columns

| Column | Owner | Meaning |
|---|---|---|
| `case` | you | run directory name (the key) |
| `state` | you | `planned` / `unfinished` / `finished` / `crashed` / `stuck` / `bad` / `ignore` / `unknown` |
| `running` | tool | live process detected for this case right now |
| `exists` | tool | run directory exists on disk |
| `from` | you | source case/template to copy inputs from |
| `restart` | you | `` = scratch, `caseX` = restart from caseX, `caseX:append` = restart & append. A bare `restart`/`restart_append` records the mode with the source still to fill (what `init` infers from the case name) |
| `hermes_build` | you | Hermes-3 build for this case (also feeds the future launcher) |
| `changed_options` | you | `;`-separated `section:key=value` list, e.g. `d:flux_limit=0.4; d:neutral_lmax=1.0` |
| `notes` | you | short setup notes |
| `final_t` | tool | last simulated time from the log |
| `runtime` | tool | summed wall time from the log |

## Commands

`caseplan.py` is on `$PATH` (via `sdtools/cli`). Run it inside the `hermes3`
spack env so `boutdata` is importable (`casegen` validates each generated
`BOUT.inp`; without boutdata that validation quietly no-ops).

```bash
caseplan.py init  STUDY  [--legacy-flat] [--runs-dir runs] [--base CASE] [--no-smart-diff] [--force]
caseplan.py index STUDY  [--legacy-flat] [--runs-dir runs]
caseplan.py gen   cases.csv  [--dry-run] [--case NAME] [--legacy-flat] [--runs-dir runs] [safety flags]
```

- **`init`** — bootstrap `cases.csv` for an existing study: discovers case
  directories, infers `from` / `changed_options` against a central base and the
  `restart` mode from each case-name suffix, and fills the tool columns from
  evidence. Refuses to overwrite an existing `cases.csv` unless `--force`. After
  that the file is yours. (The grid is a `BOUT.inp` setting, so it is not a
  column — change it via `changed_options` if you need to.)
- **`index`** — refresh the tool columns and each `case.json`. Merges on
  `case`: your columns are kept verbatim, and any run directory found on disk
  that is not yet a row is appended (so a run never silently disappears). Writes
  atomically and keeps a `cases.csv.bak`. Prints a compact table.
- **`gen`** (casegen) — create/update the cases described by the rows. Paranoid
  by default (see Safety). Use `--dry-run` to preview every copy and the exact
  `BOUT.inp` diff before touching anything.

### Layouts

Modern layout keeps generated cases under `runs/`:

```
study/
  cases.csv
  grids/grid.nc
  templates/base/BOUT.inp
  runs/<case>/{BOUT.inp, case.json, BOUT.log.*, BOUT.dmp.*.nc, ...}
```

Legacy **flat** studies keep case directories as direct children of the study
dir — pass `--legacy-flat` to `init`/`index`/`gen` for those.

## Running detection

Because you launch runs by hand on your workstation, `index` tells you what is
actually alive. It scans `/proc` once and marks a case `running` when a live
process carries a `-d <datadir>` argument that resolves to that case's
directory — matching your

```bash
taskset -c <cores> mpirun -np 10 --bind-to=none $hermes/hermes-3 -d <casedir>
```

The match is on the exact resolved `-d` path, not a substring, so
`st40fllrb4` is never mistaken for `st40fllrb4fb`.

## Safety (casegen)

The safe path is to create new cases, never to mutate expensive results.
`casegen` refuses, unless you pass the matching deliberately-ugly flag, to:

- touch an existing generated case without outputs → `--force-existing` / `--update-inputs-only`
- touch a directory containing run outputs (`BOUT.log.*`, `BOUT.dmp.*.nc`, `BOUT.restart.*.nc`, `BOUT.squash.nc`) → `--allow-output-dir`
- touch a directory with no `case.json` (hand-made) → `--overwrite-handmade`

When copying inputs from `from:`, run products, PID/scheduler markers, and
monitor plots are excluded; restart files are copied only from the `restart`
source when the mode requires them. `BOUT.inp` edits are minimal (only the
matched value changes) and written atomically after `boutdata` validation.

## Not in scope (yet)

Launching / queueing runs is a separate future stage; for now you launch
manually and `caseplan` tracks. The `hermes_build` column is the hook for that
launcher.
