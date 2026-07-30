"""Comparing a case against the recipe it was supposed to run.

A recipe is a named block of `[solver]` and `[petsc]` settings. `apply_recipe`
replaces those two sections of a BOUT.inp wholesale, so once a recipe has been
applied any difference in them is a later edit -- deliberate or accidental.

That makes the comparison a free consistency check. What the run was declared to
be testing is recorded by hand; what actually differs is measured here. When the
measurement contains something the declaration does not, a setting changed that
nobody intended.
"""

import os
import re

MANAGED_SECTIONS = ("solver", "petsc")


def parse_settings(path, sections=MANAGED_SECTIONS):
    """
    `{"section:key": value}` for the named sections of a BOUT.inp or recipe file.

    A bare line with no `=` is a PETSc flag, which is meaningful by its presence
    alone; it is recorded with an empty value so that adding or removing one
    still shows up as a difference.
    """

    wanted = {s.lower() for s in sections}
    settings = {}
    section = None

    with open(path, errors="ignore") as f:
        for raw in f:
            line = raw.split("#")[0].strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].strip().lower()
                continue
            if section not in wanted:
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                settings[f"{section}:{key.strip()}"] = value.strip()
            else:
                settings[f"{section}:{line}"] = ""

    return settings


def find_recipe(recipe, recipes_dir):
    """Path to a named recipe, or None. Names are as written in the index."""

    if not recipe or not recipes_dir:
        return None
    path = os.path.join(recipes_dir, f"{recipe}.txt")
    return path if os.path.exists(path) else None


def _comparable(value):
    """Numeric values compared as numbers, so 1e-7 and 1.0e-7 are not a diff."""

    text = value.strip().strip("\"'")
    if re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?", text):
        return repr(float(text))
    return text.lower()


def diff_against_recipe(case_dir, recipe_path):
    """
    Every deviation of a case's `[solver]`/`[petsc]` sections from its recipe,
    as `section:key: from -> to` strings, sorted so two runs that deviate the
    same way produce identical text.

    `(absent)` on either side means the setting exists in only one of the two.
    """

    case = parse_settings(os.path.join(case_dir, "BOUT.inp"))
    named = parse_settings(recipe_path)

    diffs = []
    for key in sorted(set(case) | set(named)):
        if key not in named:
            diffs.append(f"{key}: (absent) -> {case[key] or '(set)'}")
        elif key not in case:
            diffs.append(f"{key}: {named[key] or '(set)'} -> (absent)")
        elif _comparable(case[key]) != _comparable(named[key]):
            diffs.append(f"{key}: {named[key]} -> {case[key]}")

    return diffs
