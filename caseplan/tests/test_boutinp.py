import textwrap

import pytest

from caseplan.boutinp import (
    OptionChange,
    apply_option_changes,
    parse_option_changes,
    read_bout_options,
)

SAMPLE = textwrap.dedent(
    """\
    # top-level comment
    nout = 100
    timestep = 50.0   # a comment

    [mesh]
    nx = 10
    ny = 200  # cells

    [mesh:paralleltransform]
    type = shifted

    [d]
    gradient_ceiling_D = 0.3
    flux_limit = true
    """
)


@pytest.fixture
def inp(tmp_path):
    p = tmp_path / "BOUT.inp"
    p.write_text(SAMPLE)
    return p


def test_read_options(inp):
    opts = read_bout_options(inp)
    assert opts["nout"] == "100"
    assert opts["timestep"] == "50.0"
    assert opts["mesh:nx"] == "10"
    assert opts["mesh:paralleltransform:type"] == "shifted"
    assert opts["d:gradient_ceiling_D".lower()] == "0.3"


def test_read_options_accepts_directory(tmp_path):
    (tmp_path / "BOUT.inp").write_text(SAMPLE)
    opts = read_bout_options(tmp_path)
    assert opts["d:flux_limit"] == "true"


def test_parse_option_changes_ok():
    changes = parse_option_changes("d:gradient_ceiling_D = 0.025\n  nout = 50\n# note\n")
    assert changes[0].section == "d"
    assert changes[0].key == "gradient_ceiling_D"
    assert changes[0].value == "0.025"
    assert changes[1].section is None
    assert changes[1].key == "nout"


def test_parse_option_changes_rejects_non_option():
    with pytest.raises(ValueError):
        parse_option_changes("this is just prose\n")


def test_set_existing_key_in_place(inp):
    changes = [OptionChange("d", "gradient_ceiling_D", "0.025")]
    recs = apply_option_changes(inp, changes)
    assert recs[0].action == "set"
    assert recs[0].old_value == "0.3"
    opts = read_bout_options(inp)
    assert opts["d:gradient_ceiling_d"] == "0.025"
    # untouched lines preserved
    assert "flux_limit = true" in inp.read_text()


def test_preserve_inline_comment(inp):
    apply_option_changes(inp, [OptionChange(None, "timestep", "0.05")])
    line = [l for l in inp.read_text().splitlines() if l.startswith("timestep")][0]
    assert "0.05" in line
    assert "# a comment" in line


def test_add_missing_key_to_section(inp):
    apply_option_changes(inp, [OptionChange("d", "neutral_lmax", "1.0")])
    opts = read_bout_options(inp)
    assert opts["d:neutral_lmax"] == "1.0"
    # added inside [d], not after a later section
    text = inp.read_text()
    assert text.index("neutral_lmax") > text.index("[d]")


def test_add_missing_section(inp):
    apply_option_changes(inp, [OptionChange("newsec", "foo", "bar")])
    opts = read_bout_options(inp)
    assert opts["newsec:foo"] == "bar"
    assert "[newsec]" in inp.read_text()


def test_dry_run_does_not_write(inp):
    before = inp.read_text()
    recs = apply_option_changes(inp, [OptionChange("d", "gradient_ceiling_D", "0.0")], dry_run=True)
    assert inp.read_text() == before
    assert recs[0].action == "set"


def test_value_not_required_unique(tmp_path):
    # old value "0.3" also appears elsewhere; must match by key, not substring
    p = tmp_path / "BOUT.inp"
    p.write_text("[d]\na = 0.3\ngradient_ceiling_D = 0.3\n")
    apply_option_changes(p, [OptionChange("d", "gradient_ceiling_D", "9.9")])
    opts = read_bout_options(p)
    assert opts["d:a"] == "0.3"
    assert opts["d:gradient_ceiling_d"] == "9.9"
