"""Tests for the perftest extractor (R23).

The reductions here decide numbers that go into the results store and outlive
the dumps they came from, so the thing worth testing is not that they run but
that they exclude guard cells. A guard cell holds whatever was in that memory
-- on real runs up to 1e15 against an interior value of 1e-3 (F20) -- so a
reduction that includes them reports which field has the worst garbage, and
reports it as a physics result.

The synthetic dataset below is the smallest one xhermes' `clear_guards` will
accept: a single-null topology, two radial guard cells each side, no poloidal
guards.
"""

import numpy as np
import pytest
import xarray as xr

import xhermes  # noqa: F401 -- registers the .hermes accessors

from perftest.extract import (
    Report,
    _ddt_series,
    _interior,
    _interior_mask,
    _residual_shares,
)
from perftest.index import INDEX_COLUMNS, _merge_columns

from hermes3.logparse import build_type

NX, NTHETA, MXG = 8, 6, 2
GUARD_X = (0, 1, NX - 2, NX - 1)

METADATA = {
    "MXG": MXG, "MYG": 0,
    "nxg": NX, "nyg": NTHETA, "ny_inner": NTHETA // 2,
    "ixseps1": 4, "ixseps2": 4, "ixseps1g": 4, "ixseps2g": 4,
    "jyseps1_1g": 1, "jyseps2_1g": 2, "jyseps1_2g": 3, "jyseps2_2g": 4,
    "topology": "single-null",
    "keep_xboundaries": 1, "keep_yboundaries": 0,
}

GARBAGE = 1e15


def _field(interior, guard, nt=2):
    """A (t, x, theta) field holding `interior` everywhere except the radial
    guard columns, which hold `guard`."""
    values = np.full((nt, NX, NTHETA), float(interior))
    for x in GUARD_X:
        values[:, x, :] = float(guard)
    return values


def _dataset(fields, nt=2):
    """A dataset carrying `fields` plus the geometry the mask is built from."""
    data = {name: (("t", "x", "theta"), values) for name, values in fields.items()}
    for name in ("J", "dx", "dy", "dz"):
        data[name] = (("x", "theta"), np.ones((NX, NTHETA)))
    ds = xr.Dataset(data, coords={"t": np.arange(nt, dtype=float)})
    ds.attrs["metadata"] = METADATA
    for name in ds.data_vars:
        ds[name].attrs["metadata"] = METADATA
        ds[name].attrs["conversion"] = 1.0
    return ds


# =============================================================================
# The interior mask -- the thing everything else depends on
# =============================================================================
def test_mask_is_false_exactly_on_guard_cells():
    mask = _interior_mask(_dataset({}))
    assert mask is not None
    assert not mask.isel(x=list(GUARD_X)).values.any()
    assert mask.isel(x=slice(MXG, -MXG)).values.all()
    assert int(mask.values.sum()) == (NX - 2 * MXG) * NTHETA


def test_mask_has_no_time_dimension():
    """It is applied to fields that do have one, so it must broadcast."""
    assert "t" not in _interior_mask(_dataset({})).dims


def test_missing_geometry_is_a_problem_not_a_pass():
    """No mask must fail the extraction. Treating "cannot tell" as "no guards
    to remove" is exactly how the contaminated shares got recorded."""
    ds = xr.Dataset({"resid_A": (("t", "x", "theta"), np.ones((2, NX, NTHETA)))},
                    coords={"t": [0.0, 1.0]})
    report = Report("case")
    assert _interior_mask(ds, report) is None
    assert report.problems


def test_interior_without_a_mask_does_not_silently_trim():
    """`_interior(field, None)` returns the field untouched, so a caller that
    forgets to check gets an obviously wrong number rather than a plausible
    one."""
    ds = _dataset({"resid_A": _field(1.0, GARBAGE)})
    assert _interior(ds["resid_A"], None).equals(ds["resid_A"])


# =============================================================================
# Residual shares (F20)
# =============================================================================
def test_residual_shares_ignore_guard_garbage():
    """A field whose guards hold 1e15 and whose interior holds nothing must not
    take the whole norm from a field that is genuinely large inside."""
    ds = _dataset({
        "resid_loud_guards": _field(0.0, GARBAGE),
        "resid_real": _field(1.0, 0.0),
    })
    shares = _residual_shares(ds)
    assert shares["share_real"].iloc[0] == pytest.approx(1.0)
    assert shares["share_loud_guards"].iloc[0] == pytest.approx(0.0)


def test_residual_shares_sum_to_one():
    ds = _dataset({
        "resid_A": _field(1.0, GARBAGE),
        "resid_B": _field(3.0, 0.0),
    })
    shares = _residual_shares(ds)
    columns = [c for c in shares.columns if c.startswith("share_")]
    assert shares[columns].sum(axis=1).values == pytest.approx(1.0)
    # 1 and 3 in every interior cell -> squares 1 and 9.
    assert shares["share_A"].iloc[0] == pytest.approx(0.1)


def test_residual_total_counts_interior_cells_only():
    ds = _dataset({"resid_A": _field(2.0, GARBAGE)})
    total = _residual_shares(ds)["resid_ss_total"].iloc[0]
    assert total == pytest.approx(4.0 * (NX - 2 * MXG) * NTHETA)


# =============================================================================
# Rate of change
# =============================================================================
def test_ddt_rms_and_peak_ignore_guard_garbage():
    ds = _dataset({"ddt(A)": _field(2.0, GARBAGE), "A": _field(4.0, GARBAGE)})
    row = _ddt_series(ds).iloc[0]
    assert row["rms_ddt"] == pytest.approx(2.0)
    assert row["max_abs_ddt"] == pytest.approx(2.0)
    assert row["rms_state"] == pytest.approx(4.0)


def test_volume_weighting_agrees_with_plain_rms_on_uniform_cells():
    """Every cell has the same volume here, so the two norms cannot differ.
    If they do, the weighting is picking up cells the plain norm is not --
    which on a real grid would be the guards."""
    ds = _dataset({"ddt(A)": _field(3.0, GARBAGE), "A": _field(1.0, 0.0)})
    row = _ddt_series(ds).iloc[0]
    assert row["rms_ddt_vw"] == pytest.approx(row["rms_ddt"])


def test_ddt_carries_the_normalisation_factor():
    ds = _dataset({"ddt(A)": _field(1.0, 0.0), "A": _field(1.0, 0.0)})
    ds["ddt(A)"].attrs["conversion"] = 7.5
    assert _ddt_series(ds)["conversion"].iloc[0] == pytest.approx(7.5)


# =============================================================================
# Build type, read back from the compile flags
# =============================================================================
@pytest.mark.parametrize("flags, expected", [
    ("-Wall -DCHECK=2 -O2 -g -DNDEBUG", "RelWithDebInfo"),
    ("-Wall -O3 -DNDEBUG", "Release"),
    ("-Wall -Os -DNDEBUG", "MinSizeRel"),
    ("-Wall -O0 -g", "Debug"),
    ("-Wall -g", "Debug"),
])
def test_build_type_recognises_the_cmake_defaults(flags, expected):
    assert build_type(flags) == expected


def test_unrecognised_flags_report_themselves_rather_than_guessing():
    assert build_type("-Wall -march=native") == "unknown"
    assert build_type("-O1 -g -DNDEBUG") == "-O1 -g -DNDEBUG"


def test_build_type_of_nothing_is_nothing():
    assert build_type(None) is None
    assert build_type("") is None


# =============================================================================
# Index columns -- a schema that grows must not disturb what is there
# =============================================================================
def test_a_new_canonical_column_lands_in_its_canonical_place():
    without = [c for c in INDEX_COLUMNS if c != "build_type"]
    assert _merge_columns(without) == INDEX_COLUMNS


def test_existing_column_order_is_never_rearranged():
    """The file is hand-edited and read in a spreadsheet, so its own order
    wins."""
    shuffled = ["wall_s", "test_id", "case_dir"]
    merged = _merge_columns(shuffled)
    assert [c for c in merged if c in shuffled] == shuffled


def test_columns_the_schema_has_never_heard_of_are_kept():
    merged = _merge_columns(list(INDEX_COLUMNS) + ["someones_private_note"])
    assert "someones_private_note" in merged
