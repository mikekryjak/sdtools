import csv
import json
import textwrap

import pytest

from caseplan import casesfile
from caseplan.index import collect_evidence, refresh

LOG = textwrap.dedent(
    """\
    BOUT++ version 5.1.2
    Revision: 396f6a7618a6a8dcdc6f9d46e51acb9e38e84d19
    Git Version of Hermes: b021a7dd104b7c214200f6a804f2b9a9f83ed851
    	Command line options for this run : /home/mike/work/hermes-3/build/hermes-3 -d case restart
    	Option input:error_on_unused_options = 1 (default)
    Sim Time  |  RHS evals  | Wall Time |  Calc    Inv   Comm    I/O   SOLVER

    2.395e+06          1       6.35e-02     3.7    0.0    0.0  117.6  -21.3
    2.476e+06         25       9.63e-02    45.4    0.0    0.1   34.9   19.6
    """
)

CASES_CSV = textwrap.dedent(
    """\
    case,state,from,restart,hermes_build,changed_options,notes
    case-a,finished,templates/base,,,d:gradient_ceiling_D=0.025,keep me
    case-planned,planned,templates/base,,,d:gradient_ceiling_D=0.1,
    """
)


@pytest.fixture
def study(tmp_path):
    runs = tmp_path / "runs" / "case-a"
    runs.mkdir(parents=True)
    (runs / "BOUT.inp").write_text("nout = 100\n")
    (runs / "BOUT.log.0").write_text(LOG)
    (runs / "BOUT.dmp.0.nc").write_text("x")
    (runs / "BOUT.dmp.1.nc").write_text("x")
    (tmp_path / "cases.csv").write_text(CASES_CSV)
    return tmp_path


def _csv_rows(study):
    with (study / "cases.csv").open() as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, {r["case"]: r for r in reader}


def test_evidence_from_log(study):
    ev = collect_evidence(study / "runs" / "case-a", running_dirs=set())
    assert ev.bout_version == "5.1.2"
    assert ev.bout_commit.startswith("396f6a")
    assert ev.hermes_commit.startswith("b021a7")
    assert ev.run_binary.endswith("hermes-3")
    assert ev.final_t == pytest.approx(2.476e06)
    assert ev.n_dumps == 2
    assert ev.is_running is False
    # option-echo lines containing "error" must not be flagged as errors
    assert ev.error_hint is None


def test_refresh_merges_csv_and_writes_case_json(study):
    refresh(study)

    # case.json evidence written for the real case
    cj = json.loads((study / "runs" / "case-a" / "case.json").read_text())
    assert "evidence" in cj
    assert cj["evidence"]["bout_version"] == "5.1.2"

    header, rowmap = _csv_rows(study)
    assert header == casesfile.COLUMNS
    assert rowmap["case-a"]["state"] == "finished"
    assert rowmap["case-a"]["exists"] == "yes"
    assert rowmap["case-a"]["running"] == "no"
    assert rowmap["case-planned"]["exists"] == "no"
    # human columns are preserved verbatim
    assert rowmap["case-a"]["changed_options"] == "d:gradient_ceiling_D=0.025"
    assert rowmap["case-a"]["notes"] == "keep me"
    assert rowmap["case-a"]["from"] == "templates/base"


def test_discovered_dir_is_appended(study):
    extra = study / "runs" / "case-b"
    extra.mkdir()
    (extra / "BOUT.inp").write_text("nout = 1\n")
    refresh(study)
    _, rowmap = _csv_rows(study)
    assert "case-b" in rowmap
    assert rowmap["case-b"]["exists"] == "yes"
    assert rowmap["case-b"]["state"] == "unknown"


def test_running_detection_via_dir_set(study):
    run_dir = study / "runs" / "case-a"
    ev = collect_evidence(run_dir, running_dirs={run_dir.resolve()})
    assert ev.is_running is True
    assert "live process" in ev.running_evidence


def test_generation_block_preserved_on_reindex(study):
    cj_path = study / "runs" / "case-a" / "case.json"
    cj_path.write_text(json.dumps({"generation": {"case": "case-a", "marker": 1}}))
    refresh(study)
    data = json.loads(cj_path.read_text())
    assert data["generation"]["marker"] == 1  # untouched
    assert "evidence" in data  # added
