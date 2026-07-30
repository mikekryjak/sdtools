"""Minimal-edit reader/writer for BOUT.inp files.

Hardened descendant of ``cli/read_opt.py`` and ``cli/set_opt.py``. The guiding
philosophy is unchanged: replace only the matched value in place and leave the
rest of ``BOUT.inp`` untouched, so generated cases produce small, readable
diffs. ``boutdata.BoutOptionsFile`` is used only as an optional validation /
readback tool, never as the default writer (it canonicalises files and creates
large formatting-only diffs).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OptionChange:
    """A single requested option assignment, e.g. ``d:gradient_ceiling_D = 0.1``."""

    section: str | None
    key: str
    value: str
    raw: str = ""

    @property
    def dotted(self) -> str:
        return f"{self.section}:{self.key}" if self.section else self.key


@dataclass
class LineChange:
    """Record of what ``apply_option_changes`` did for one option."""

    change: OptionChange
    action: str  # "set" | "add_key" | "add_section"
    line_no: int | None  # 1-based line affected, None for appends computed later
    old_line: str | None
    new_line: str
    old_value: str | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_inp(path) -> Path:
    """Accept a case directory, a direct BOUT.inp path, or an absolute path."""
    p = Path(path)
    if p.is_dir():
        return p / "BOUT.inp"
    return p


def _section_of(line: str) -> str | None:
    """Return the section name if ``line`` is a ``[section]`` header, else None."""
    s = line.strip()
    if s.startswith("[") and "]" in s:
        return s[1 : s.index("]")].strip()
    return None


def _split_lhs(lhs: str) -> tuple[str | None, str]:
    lhs = lhs.strip()
    if ":" in lhs:
        section, key = lhs.split(":", 1)
        return section.strip(), key.strip()
    return None, lhs


def _key_of(code_line: str) -> str | None:
    """Real ``key = value`` split: return the lowercased key, or None."""
    code = code_line.split("#", 1)[0]
    if "=" not in code:
        return None
    return code.split("=", 1)[0].strip().lower()


def _value_of(code_line: str) -> str:
    code = code_line.split("#", 1)[0]
    return code.split("=", 1)[1].strip() if "=" in code else ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_bout_options(path) -> dict[str, str]:
    """Parse a BOUT.inp into a ``{"section:key": "value"}`` dict.

    Top-level keys (before the first ``[section]``) appear bare. Keys and
    sections are lowercased to match the historic ``read_opt`` behaviour.
    """
    inp = _resolve_inp(path)
    settings: dict[str, str] = {}
    section = ""
    for line in inp.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        header = _section_of(line)
        if header is not None:
            section = header.lower()
            continue
        key = _key_of(line)
        if key is None:
            continue
        value = _value_of(line)
        full = f"{section}:{key}" if section else key
        settings[full] = value
    return settings


def parse_option_changes(text: str) -> list[OptionChange]:
    """Parse a block of ``section:key = value`` lines.

    Non-option, non-blank, non-comment lines raise ``ValueError`` rather than
    being silently ignored (the plan's ``Changes:`` safety rule).
    """
    changes: list[OptionChange] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            raise ValueError(f"Not an option assignment: {raw!r}")
        lhs, value = s.split("=", 1)
        section, key = _split_lhs(lhs)
        if not key:
            raise ValueError(f"Missing option name: {raw!r}")
        changes.append(
            OptionChange(section=section, key=key, value=value.strip(), raw=s)
        )
    return changes


def _replace_value_in_line(line: str, new_value: str) -> str:
    """Replace only the value field, preserving spacing and any inline comment."""
    newline = "\n" if line.endswith("\n") else ""
    body = line[:-1] if newline else line
    before_eq, after_eq = body.split("=", 1)
    if "#" in after_eq:
        val_part, comment = after_eq.split("#", 1)
        comment = "#" + comment
    else:
        val_part, comment = after_eq, ""
    lead = val_part[: len(val_part) - len(val_part.lstrip())]
    if comment:
        # keep the whitespace that sat between the value and the comment
        gap = val_part[len(val_part.rstrip()) :]
        new_after = lead + str(new_value) + gap + comment
    else:
        new_after = lead + str(new_value)
    return before_eq + "=" + new_after + newline


def _section_bounds(lines: list[str], section: str | None) -> tuple[int, int, int | None]:
    """Return ``(start, end, header_idx)`` for the body of ``section``.

    For the top-level (``section is None``) block, the body runs from line 0 to
    the first ``[section]`` header. ``header_idx`` is None for the top level and
    when the section is absent (then ``start == end == len(lines)``).
    """
    if section is None:
        end = len(lines)
        for i, line in enumerate(lines):
            if _section_of(line) is not None:
                end = i
                break
        return 0, end, None

    header_idx = None
    for i, line in enumerate(lines):
        h = _section_of(line)
        if h is not None and h.lower() == section.lower():
            header_idx = i
            break
    if header_idx is None:
        return len(lines), len(lines), None

    start = header_idx + 1
    end = len(lines)
    for i in range(start, len(lines)):
        if _section_of(lines[i]) is not None:
            end = i
            break
    return start, end, header_idx


def _indent_of(lines: list[str], start: int, end: int) -> str:
    for i in range(start, end):
        code = lines[i].split("#", 1)[0]
        if "=" in code:
            return lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    return ""


def _apply_one(lines: list[str], change: OptionChange) -> LineChange:
    key_lower = change.key.lower()
    start, end, header_idx = _section_bounds(lines, change.section)

    # Section missing entirely: append a fresh section + key.
    if change.section is not None and header_idx is None:
        block = []
        if lines and lines[-1].strip() != "":
            block.append("\n")
        block.append(f"[{change.section}]\n")
        block.append(f"{change.key} = {change.value}\n")
        lines.extend(block)
        return LineChange(
            change=change,
            action="add_section",
            line_no=len(lines),
            old_line=None,
            new_line=f"{change.key} = {change.value}",
            old_value=None,
        )

    # Look for an existing key in the section body.
    for i in range(start, end):
        if _key_of(lines[i]) == key_lower:
            old_line = lines[i]
            old_value = _value_of(lines[i])
            new_line = _replace_value_in_line(lines[i], change.value)
            lines[i] = new_line
            return LineChange(
                change=change,
                action="set",
                line_no=i + 1,
                old_line=old_line.rstrip("\n"),
                new_line=new_line.rstrip("\n"),
                old_value=old_value,
            )

    # Key absent: insert after the last non-blank line of the section body.
    insert_idx = end
    j = end - 1
    while j >= start and lines[j].strip() == "":
        j -= 1
    insert_idx = j + 1
    indent = _indent_of(lines, start, end)
    new_text = f"{indent}{change.key} = {change.value}\n"
    lines.insert(insert_idx, new_text)
    return LineChange(
        change=change,
        action="add_key",
        line_no=insert_idx + 1,
        old_line=None,
        new_line=new_text.rstrip("\n"),
        old_value=None,
    )


def apply_option_changes(path, changes, *, dry_run: bool = False) -> list[LineChange]:
    """Apply ``changes`` to a BOUT.inp.

    Returns a structured ``LineChange`` per change so ``casegen --dry-run`` can
    print exact diffs. When ``dry_run`` is False the write is atomic (temp file,
    optional validation, then replace); if validation fails the original file is
    left untouched.
    """
    inp = _resolve_inp(path)
    text = inp.read_text()
    # Keep line endings so we can rewrite faithfully.
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] = lines[-1] + "\n"

    records = [_apply_one(lines, c) for c in changes]

    if dry_run:
        return records

    new_text = "".join(lines)
    tmp = inp.with_name(inp.name + ".caseplan-tmp")
    tmp.write_text(new_text)
    try:
        validate_with_boutdata(tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, inp)
    return records


def validate_with_boutdata(path) -> None:
    """Best-effort validation/readback via ``boutdata.BoutOptionsFile``.

    A no-op when boutdata is not importable (the plan only requires validation
    "when available"). Raises if boutdata is present and rejects the file.
    """
    inp = _resolve_inp(path)
    try:
        from boutdata.data import BoutOptionsFile  # type: ignore
    except Exception:
        try:
            from boutdata import BoutOptionsFile  # type: ignore
        except Exception:
            return  # boutdata unavailable; skip validation
    BoutOptionsFile(str(inp))


def format_line_change(lc: LineChange) -> str:
    """Human-readable one/two-line diff for dry-run output."""
    head = f"  [{lc.action}] {lc.change.dotted} = {lc.change.value}"
    if lc.action == "set":
        return f"{head}\n      - {lc.old_line}\n      + {lc.new_line}"
    return f"{head}\n      + {lc.new_line}"
