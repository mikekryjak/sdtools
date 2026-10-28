"""Which output Load.case_2D reads for each `source`, on empty stand-in files."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hermes3.load import Load, output_path  # noqa: E402


def make(tmp_path, *names):
    for name in names:
        (tmp_path / name).write_bytes(b"")
    return str(tmp_path)


DUMPS = ("BOUT.dmp.0.nc", "BOUT.dmp.1.nc")


def test_auto_prefers_dumps(tmp_path, capsys):
    case = make(tmp_path, *DUMPS, "BOUT.distilled.nc")
    assert output_path(case).endswith("BOUT.dmp.*.nc")
    assert "ignored" in capsys.readouterr().out


def test_auto_falls_back_to_distilled(tmp_path, capsys):
    case = make(tmp_path, "BOUT.distilled.nc")
    assert output_path(case).endswith("BOUT.distilled.nc")
    assert "NaN" in capsys.readouterr().out


def test_auto_with_neither_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="neither"):
        output_path(make(tmp_path, "BOUT.inp"))


def test_explicit_sources(tmp_path):
    case = make(tmp_path, *DUMPS, "BOUT.squash.nc", "BOUT.distilled.nc")
    assert output_path(case, "dumps").endswith("BOUT.dmp.*.nc")
    assert output_path(case, "squash").endswith("BOUT.squash.nc")
    assert output_path(case, "distilled").endswith("BOUT.distilled.nc")


@pytest.mark.parametrize("source", ["dumps", "squash", "distilled"])
def test_explicit_source_missing_fails(tmp_path, source):
    with pytest.raises(FileNotFoundError):
        output_path(make(tmp_path, "BOUT.inp"), source)


def test_unknown_source(tmp_path):
    with pytest.raises(ValueError):
        output_path(make(tmp_path, *DUMPS), "dumpz")


def test_squashed_file_needs_y_guards(tmp_path):
    case = make(tmp_path, "BOUT.distilled.nc")
    with pytest.raises(ValueError, match="keep_yboundaries"):
        Load.case_2D(case, None, keep_yboundaries=False)
