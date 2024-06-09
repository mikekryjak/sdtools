from xarray import register_dataset_accessor, register_dataarray_accessor
from xbout import BoutDatasetAccessor, BoutDataArrayAccessor
from hermes3.plotting import *
import numpy as np
import xarray as xr


def find_front_position(ds):
    """
    Find front position and add it to the dataset
    It's very slow... maybe it can be optimised
    Currently only doing 5eV, others are also there but commented out
    """
    if "t" not in ds.dims:
        raise Exception("Dataset must contain more than one timestep")
    fl = ds.hermesm.select_custom_sol_ring(int(ds.metadata["ixseps1"]), "outer_lower").squeeze()
    dist = np.cumsum(fl["dl"]).values
    dist_from_target = dist[-1] - dist

    df = pd.DataFrame()
    df.index = range(ds.dims["t"])
    df["t"] = ds["t"]


    ## Can't do a simple argmin() because the data isn't monotonic and could technically cross 5eV 
    # multiple times. So this looks at the final crossing and interpolates in-between.
    # Another benefit of this method is that the front position is interpolated and therefore doesn't jump
    # between cells
    def find_crossing(dist, data, threshold):

        # Find indices where the temperature crosses the threshold
        final_crossing = np.where(np.diff(np.signbit(data - threshold)))[0][-1]

        # Initialize a list to store crossing times
        crossing_times = []

        # Interpolate within each crossing interval to find the exact crossing time
        t1, t2 = dist[final_crossing], dist[final_crossing + 1]
        data1, data2 = data[final_crossing], data[final_crossing + 1]

        # Linear interpolation to find the crossing time
        location = t1 + (threshold - data1) * (t2 - t1) / (data2 - data1)

        return location



    for t in range(ds.dims["t"]):

        timeslice = fl.isel(t=t)
        df.loc[t, "5eV"] = find_crossing(dist, timeslice["Te"].values, 5)
        # df.loc[t, "Ne_peak"] = find_crossing(dist, timeslice["Ne"].values, timeslice["Ne"].values.max())
        # df.loc[t, "Rc_peak"] = find_crossing(dist, timeslice["Rc"].values, timeslice["Rc"].values.max())
        # df.loc[t, "iz_peak"] = find_crossing(dist, timeslice["Sd+_iz"].values, timeslice["Sd+_iz"].values.max())
        # df.loc[t, "rec_peak"] = find_crossing(dist, timeslice["Sd+_rec"].values, timeslice["Sd+_rec"].values.max())



    df["5eV"] = dist[-1] - df["5eV"]

    ds["front_poldist_5eV"] = xr.DataArray(df["5eV"].values, dims = ["t"])
    ds["front_poldist_5eV"].attrs.update(dict(
        short_name = "Front pol. distance from target [m]",
        units = "m",
        origin = "sdtools")
    )

    return ds

