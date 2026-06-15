#!/usr/bin/env python3

"""
Create BOUT++ restart files from a case directory at the time nearest to the
requested milliseconds.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import xbout


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create BOUT++ restart files from the output in a case directory at the "
            "time nearest to the requested milliseconds."
        )
    )
    parser.add_argument("casepath", help="Path to the input case directory")
    parser.add_argument("output_path", help="Directory where restart files will be written")
    parser.add_argument("time_ms", type=float, help="Target simulation time in milliseconds")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing restart files in the output directory",
    )
    parser.add_argument(
        "--copy_input_file",
        action="store_true",
        help="Copy BOUT.inp into the output directory after writing restart files",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    casepath = Path(args.casepath).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    if not casepath.is_dir():
        raise FileNotFoundError(f"Case directory not found: {casepath}")

    inputfilepath = casepath / "BOUT.inp"
    if not inputfilepath.exists():
        raise FileNotFoundError(f"BOUT.inp not found in {casepath}")

    loadfilepath = casepath / "BOUT.dmp.*.nc"
    if not list(casepath.glob("BOUT.dmp.*.nc")):
        raise FileNotFoundError(f"No BOUT.dmp.*.nc files found in {casepath}")

    output_path.mkdir(parents=True, exist_ok=True)

    ds = xbout.load.open_boutdataset(
        datapath=str(loadfilepath),
        inputfilepath=str(inputfilepath),
        keep_xboundaries=True,
        keep_yboundaries=True,
    )

    omega_ci = ds.metadata["Omega_ci"]
    times_ms = np.asarray(ds["t"], dtype=float) * (1000.0 / omega_ci)

    if times_ms.size == 0:
        raise ValueError(f"No time points found in dataset for {casepath}")

    tind = int(np.abs(times_ms - args.time_ms).argmin())
    selected_time_ms = float(times_ms[tind])

    ds.bout.to_restart(
        savepath=str(output_path),
        tind=tind,
        overwrite=args.overwrite,
    )

    if args.copy_input_file:
        shutil.copy2(inputfilepath, output_path / "BOUT.inp")

    print(
        f"Created restart files in {output_path} from {casepath} at "
        f"tind={tind} (requested {args.time_ms:.6g} ms, selected {selected_time_ms:.6g} ms)"
    )

    if args.copy_input_file:
        print(f"Copied {inputfilepath.name} to {output_path}")


if __name__ == "__main__":
    main()
