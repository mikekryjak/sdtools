#!/usr/bin/env python3
"""PATH wrapper for the caseplan package.

sdtools ships no packaging; cli/ is on $PATH and its tools are run by name.
This exposes the caseplan package (sdtools/caseplan/caseplan) the same way,
so `caseplan.py index STUDY` works from anywhere with no install step.
"""

import pathlib
import sys

# .../sdtools/caseplan holds the `caseplan` package; put it on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "caseplan"))

from caseplan.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
