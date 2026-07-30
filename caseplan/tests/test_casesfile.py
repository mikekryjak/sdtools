import textwrap

import pytest

from caseplan import casesfile
from caseplan.casesfile import CaseRow, parse_changed_options
from caseplan.init import _infer_restart


def test_infer_restart_from_name():
    assert _infer_restart("st40fllrb4-newvth_restart") == "restart"
    assert _infer_restart("st40fllrb2-newlimform_scratch") == "scratch"
    assert _infer_restart("caseX_restart_append") == "restart_append"
    assert _infer_restart("st40fllrb4f-grad_ceil_0.3") == ""


def test_restart_spec():
    assert CaseRow(case="c").restart_spec() == (None, "scratch")
    assert CaseRow(case="c", restart="scratch").restart_spec() == (None, "scratch")
    assert CaseRow(case="c", restart="b4fd").restart_spec() == ("b4fd", "restart")
    assert CaseRow(case="c", restart="b4fd:append").restart_spec() == ("b4fd", "restart_append")


def test_restart_spec_bare_mode_words():
    # A name-inferred mode with no source yet: mode known, source None.
    assert CaseRow(case="c", restart="restart").restart_spec() == (None, "restart")
    assert CaseRow(case="c", restart="restart_append").restart_spec() == (None, "restart_append")


def test_parse_changed_options():
    changes = parse_changed_options("d:flux_limit=0.4; d:neutral_lmax=1.0")
    assert [c.dotted for c in changes] == ["d:flux_limit", "d:neutral_lmax"]
    assert [c.value for c in changes] == ["0.4", "1.0"]


def test_parse_changed_options_top_level():
    (change,) = parse_changed_options("nout = 100")
    assert change.section is None
    assert change.key == "nout"
    assert change.value == "100"


def test_parse_changed_options_empty():
    assert parse_changed_options("") == []
    assert parse_changed_options("  ;  ") == []


def test_bad_changed_options_raises():
    with pytest.raises(ValueError):
        parse_changed_options("this is not an assignment")


def test_roundtrip_preserves_human_columns(tmp_path):
    rows = [
        CaseRow(case="a", state="finished", from_="base", restart="a:append",
                hermes_build="/b", changed_options="d:x=1", notes="hi",
                running=True, exists=True, final_t="1.2e7", runtime="42"),
    ]
    path = tmp_path / "cases.csv"
    casesfile.write_cases(path, rows)
    back = casesfile.read_cases(path)
    r = back[0]
    assert (r.case, r.state, r.from_, r.restart, r.hermes_build,
            r.changed_options, r.notes) == (
        "a", "finished", "base", "a:append", "/b", "d:x=1", "hi")
    assert r.running is True and r.exists is True
    assert r.final_t == "1.2e7" and r.runtime == "42"


def test_write_creates_backup(tmp_path):
    path = tmp_path / "cases.csv"
    casesfile.write_cases(path, [CaseRow(case="a")])
    casesfile.write_cases(path, [CaseRow(case="a"), CaseRow(case="b")])
    assert (tmp_path / "cases.csv.bak").exists()


def test_missing_columns_tolerated(tmp_path):
    # A file with only the human columns still reads; tool columns default off.
    path = tmp_path / "cases.csv"
    path.write_text(textwrap.dedent("""\
        case,state,from,changed_options
        a,planned,base,d:x=1
        """))
    (r,) = casesfile.read_cases(path)
    assert r.case == "a" and r.state == "planned"
    assert r.running is False and r.exists is False
    assert r.final_t == ""
