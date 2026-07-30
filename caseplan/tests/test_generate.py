import json
import textwrap

import pytest

from caseplan.generate import GenerationError, GenOptions, casegen

BASE_INP = textwrap.dedent(
    """\
    nout = 100
    timestep = 50.0

    [d]
    gradient_ceiling_D = 0.3
    flux_limit = true
    """
)

CASES_CSV = textwrap.dedent(
    """\
    case,state,from,restart,hermes_build,changed_options,notes
    case-a,planned,templates/base,,,d:gradient_ceiling_D=0.025,
    """
)


@pytest.fixture
def study(tmp_path):
    base = tmp_path / "templates" / "base"
    base.mkdir(parents=True)
    (base / "BOUT.inp").write_text(BASE_INP)
    (base / "submit.sh").write_text("#!/bin/bash\necho run\n")
    # an output file that must NOT be copied
    (base / "BOUT.log.0").write_text("junk")
    cases = tmp_path / "cases.csv"
    cases.write_text(CASES_CSV)
    return tmp_path, cases


def test_dry_run_makes_no_dir(study):
    root, cases = study
    results = casegen(cases, GenOptions(dry_run=True))
    assert not (root / "runs" / "case-a").exists()
    r = results[0]
    assert any("gradient_ceiling_D" in m for m in r.messages)


def test_generate_creates_case(study):
    root, cases = study
    casegen(cases, GenOptions())
    target = root / "runs" / "case-a"
    assert (target / "BOUT.inp").exists()
    assert (target / "submit.sh").exists()
    # outputs excluded
    assert not (target / "BOUT.log.0").exists()
    # change applied
    assert "0.025" in (target / "BOUT.inp").read_text()
    # generation record written
    data = json.loads((target / "case.json").read_text())
    assert data["generation"]["case"] == "case-a"
    assert data["generation"]["from"] == "templates/base"
    assert data["generation"]["applied_changes"][0]["option"] == "d:gradient_ceiling_D"


def test_restart_copies_restart_files(tmp_path):
    base = tmp_path / "templates" / "base"
    base.mkdir(parents=True)
    (base / "BOUT.inp").write_text(BASE_INP)
    parent = tmp_path / "runs" / "parent"
    parent.mkdir(parents=True)
    (parent / "BOUT.inp").write_text(BASE_INP)
    (parent / "BOUT.restart.0.nc").write_text("restartdata")
    cases = tmp_path / "cases.csv"
    cases.write_text(textwrap.dedent("""\
        case,state,from,restart,hermes_build,changed_options,notes
        child,planned,templates/base,parent:append,,d:flux_limit=0.4,
        """))
    casegen(cases, GenOptions())
    target = tmp_path / "runs" / "child"
    assert (target / "BOUT.restart.0.nc").read_text() == "restartdata"
    gen = json.loads((target / "case.json").read_text())["generation"]
    assert gen["restart_mode"] == "restart_append"
    assert gen["restart_from"] == "parent"


def test_refuses_existing_without_flag(study):
    root, cases = study
    casegen(cases, GenOptions())
    with pytest.raises(GenerationError):
        casegen(cases, GenOptions())


def test_force_existing_allows_update(study):
    root, cases = study
    casegen(cases, GenOptions())
    casegen(cases, GenOptions(force_existing=True))  # should not raise


def test_refuses_protected_outputs(study):
    root, cases = study
    casegen(cases, GenOptions())
    target = root / "runs" / "case-a"
    (target / "BOUT.dmp.0.nc").write_text("results")
    with pytest.raises(GenerationError):
        casegen(cases, GenOptions(force_existing=True))


def test_refuses_handmade_dir(study):
    root, cases = study
    target = root / "runs" / "case-a"
    target.mkdir(parents=True)
    (target / "BOUT.inp").write_text("handmade")
    with pytest.raises(GenerationError):
        casegen(cases, GenOptions())
