"""drop_incomplete_final_records on stand-in dumps that hold only t_array."""
import os
import sys

import netCDF4
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hermes3.load import drop_incomplete_final_records  # noqa: E402

FILL = 9.969209968386869e36     # netCDF's default float fill value


def write_dump(path, t):
    with netCDF4.Dataset(path, "w") as f:
        f.createDimension("t", None)
        f.createVariable("t_array", "f8", ("t",))[:] = t


def test_record_broken_on_one_processor_only(tmp_path, capsys):
    # Processor 0 flushed its last timestamp, processor 1 did not.
    write_dump(tmp_path / "BOUT.dmp.0.nc", [0.0, 1.0, 2.0, 3.0])
    write_dump(tmp_path / "BOUT.dmp.1.nc", [0.0, 1.0, 2.0, FILL])
    ds = xr.Dataset(coords={"t": [0.0, 1.0, 2.0, 3.0]})
    assert drop_incomplete_final_records(ds).sizes["t"] == 4
    assert drop_incomplete_final_records(ds, str(tmp_path)).sizes["t"] == 3
    assert "at least one processor" in capsys.readouterr().out


def test_no_dumps_uses_the_loaded_times(tmp_path):
    # A squash file or distillate: only the loaded t counts.
    ds = xr.Dataset(coords={"t": [0.0, 1.0, 2.0, np.nan]})
    assert drop_incomplete_final_records(ds, str(tmp_path)).sizes["t"] == 3
