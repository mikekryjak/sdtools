#!/usr/bin/env python3

import argparse
import os
import re
import sys

"""
select_recipe.py - Apply a solver "recipe" to a case's BOUT.inp.

A recipe is a plain text file containing one or both of the [solver] and
[petsc] sections (plus optional leading comment lines describing the recipe).
This tool overwrites everything under [solver] and under [petsc] in the case's
BOUT.inp with the corresponding sections from the recipe file, leaving every
other section untouched.

Behaviour:
  - The [solver] body in BOUT.inp is replaced with the recipe's [solver] body.
  - The [petsc] body in BOUT.inp is replaced with the recipe's [petsc] body.
  - If the recipe omits a section, that section's body in BOUT.inp is emptied.
  - If BOUT.inp lacks a section that the recipe provides, it is inserted
    immediately after the [solver] section.
  - Leading comment lines in the recipe (before the first section) are ignored.

Usage:
    select_recipe.py <case folder> <recipe.txt>
"""

# Section headers look like "[solver]" possibly with a trailing comment.
SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]")

# Sections that this tool manages.
MANAGED = ["solver", "petsc"]


def section_name(line):
    """Return the lowercase section name if line is a header, else None."""
    m = SECTION_RE.match(line)
    if m:
        return m.group(1).strip().lower()
    return None


def extract_section_body(lines, name):
    """Return (found, body_lines) for the named section.

    Body excludes the header line and stops at the next section header.
    Leading and trailing blank lines are stripped.
    """
    name = name.lower()
    in_section = False
    found = False
    body = []
    for line in lines:
        sec = section_name(line)
        if sec is not None:
            if sec == name:
                in_section = True
                found = True
                continue
            if in_section:
                break
            continue
        if in_section:
            body.append(line)

    # Strip surrounding blank lines.
    while body and body[0].strip() == "":
        body.pop(0)
    while body and body[-1].strip() == "":
        body.pop()

    # Ensure every line ends with a newline.
    body = [b if b.endswith("\n") else b + "\n" for b in body]
    return found, body


def select_recipe(case, recipe_path):
    """Overwrite [solver] and [petsc] in case/BOUT.inp from the recipe file."""
    path_inp = os.path.join(case, "BOUT.inp")

    if not os.path.isfile(path_inp):
        sys.exit(f"Error: no BOUT.inp found in '{case}'")
    if not os.path.isfile(recipe_path):
        sys.exit(f"Error: recipe file '{recipe_path}' not found")

    with open(recipe_path) as f:
        recipe_lines = f.readlines()
    with open(path_inp) as f:
        target_lines = f.readlines()

    # Pull the managed sections out of the recipe (empty body if absent).
    recipe_bodies = {}
    for name in MANAGED:
        _, body = extract_section_body(recipe_lines, name)
        recipe_bodies[name] = body

    target_sections = {
        section_name(l) for l in target_lines if section_name(l) is not None
    }

    def render_section(name):
        out = [f"[{name}]\n"]
        out.extend(recipe_bodies[name])
        out.append("\n")
        return out

    # Rebuild BOUT.inp, replacing managed section bodies in place.
    out = []
    replaced = []
    i = 0
    n = len(target_lines)
    while i < n:
        line = target_lines[i]
        sec = section_name(line)

        if sec in MANAGED:
            out.append(line)  # keep the original header line
            out.extend(recipe_bodies[sec])
            replaced.append(sec)
            # Skip the original body up to the next header.
            i += 1
            while i < n and section_name(target_lines[i]) is None:
                i += 1
            # Tidy separation before whatever comes next.
            if i < n:
                out.append("\n")

            # If this is the solver section, insert any managed sections that
            # the recipe provides but the target lacks (e.g. a missing petsc).
            if sec == "solver":
                for name in MANAGED:
                    if name not in target_sections:
                        out.extend(render_section(name))
                        replaced.append(name)
            continue

        out.append(line)
        i += 1

    # If the target had no [solver] at all, append the managed sections.
    if "solver" not in target_sections:
        for name in MANAGED:
            if name not in target_sections:
                out.extend(render_section(name))
                replaced.append(name)

    with open(path_inp, "w") as f:
        f.writelines(out)

    print(f"-> Applied recipe '{os.path.basename(recipe_path)}' to {path_inp}")
    for name in MANAGED:
        nlines = len(recipe_bodies[name])
        state = "inserted" if name not in target_sections else "overwritten"
        if nlines == 0:
            print(f"   [{name}] {state} (emptied - not in recipe)")
        else:
            print(f"   [{name}] {state} ({nlines} lines)")


# ------------------------------------------------------------
# PARSER
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overwrite [solver] and [petsc] in a case's BOUT.inp "
        "from a recipe text file."
    )
    parser.add_argument("case", type=str, help="Case folder containing BOUT.inp")
    parser.add_argument("recipe", type=str, help="Recipe text file to apply")

    args = parser.parse_args()
    select_recipe(args.case, args.recipe)
