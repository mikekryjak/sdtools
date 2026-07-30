#!/usr/bin/env python3

import argparse
import ast
import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hermes3.accessors import *  # noqa: F401,F403
from hermes3.case_db import CaseDB
from hermes3.plotting import plot2d


def parse_limit(values):
    if values is None:
        return (None, None)

    if len(values) == 1:
        try:
            parsed = ast.literal_eval(values[0])
        except (SyntaxError, ValueError) as error:
            raise argparse.ArgumentTypeError(
                "limits must be two values or a tuple like '(1.0, 2.0)'"
            ) from error
    elif len(values) == 2:
        parsed = values
    else:
        raise argparse.ArgumentTypeError(
            "limits must be two values or a tuple like '(1.0, 2.0)'"
        )

    if len(parsed) != 2:
        raise argparse.ArgumentTypeError("limits must contain exactly two values")

    return tuple(
        None if value is None or value == "None" else float(value) for value in parsed
    )


def load_case(case_folder):
    case_folder = os.path.abspath(case_folder)
    case_root = os.path.dirname(case_folder)
    casename = os.path.basename(case_folder.rstrip(os.sep))

    db = CaseDB(case_dir=case_root, grid_dir=case_root)
    case = db.load_case_2D(casename, use_squash=False)
    case.extract_2d_tokamak_geometry()
    return case


def plot_case_2d(case_folder, params, xlim=(None, None), ylim=(None, None)):
    case = load_case(case_folder)
    ds = case.ds

    missing = [param for param in params if param not in ds]
    if missing:
        raise KeyError(f"Variables not found in dataset: {', '.join(missing)}")

    toplot = []
    for param in params:
        data = ds[param]
        if "t" in data.dims:
            data = data.isel(t=-1)
        toplot.append({"data": data, "title": param})

    plot2d(toplot, xlim=xlim, ylim=ylim)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot 2D Hermes variables from a case folder."
    )
    parser.add_argument("case_folder", help="Path to the case folder")
    parser.add_argument("params", nargs="+", help="Variables to plot")
    parser.add_argument(
        "--xlim",
        nargs="+",
        default=None,
        metavar=("MIN", "MAX"),
        help="R-axis limits, e.g. --xlim 1.2 1.8 or --xlim '(1.2, 1.8)'",
    )
    parser.add_argument(
        "--ylim",
        nargs="+",
        default=None,
        metavar=("MIN", "MAX"),
        help="Z-axis limits, e.g. --ylim -1.0 1.0 or --ylim '(-1.0, 1.0)'",
    )

    args = parser.parse_args()
    plot_case_2d(
        args.case_folder,
        args.params,
        xlim=parse_limit(args.xlim),
        ylim=parse_limit(args.ylim),
    )


if __name__ == "__main__":
    main()
