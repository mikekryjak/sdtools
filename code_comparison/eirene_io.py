"""Self-contained readers for EIRENE triangle-mesh output (fort.33/34/35/46).

These replace the equivalent routines in the external
``solps_python_scripts`` package so the code_comparison tooling has no
dependency on it. Formats and unit conventions follow the SOLPS-ITER
fort.3x/fort.46 layout (original Fortran/MATLAB readers by W. Dekeyser).

Everything is returned in SI units:
    nodes        [m]
    pden*        [m^-3]            number density
    eden*        [J m^-3]          energy density
    tden*        [eV]              temperature (= eden / pden / 1.5)
    v{x,y,z}den* [kg m^-2 s^-1]    momentum density (already mass-weighted)

The momentum components are (x, y, z) = (R, Z, toroidal). The mean neutral
velocity on a triangle is ``v{x,y,z}den* / pden* / m``; the field-aligned
parallel component requires projecting onto B on the B2 grid (see SOLPScase).
"""

import os
import numpy as np

eV = 1.6022e-19


def _resolve(path, fname):
    """Return the path to ``fname``, falling back to ../baserun/ as SOLPS does."""
    here = os.path.join(path, fname)
    if os.path.exists(here):
        return here
    base = os.path.join(path, "..", "baserun", fname)
    if os.path.exists(base):
        return base
    raise FileNotFoundError(f"{fname} not found in {path} or ../baserun/")


def read_fort33(path):
    """Read fort.33 triangle node coordinates. Returns (nnodes, 2) array [m].

    Serial layout: nnodes R values followed by nnodes Z values, free-wrapped
    across lines, in cm.
    """
    with open(_resolve(path, "fort.33")) as fid:
        nnodes = int(fid.readline().split()[0])
        nodes = np.zeros((nnodes, 2))
        for col in (0, 1):
            i = 0
            while i < nnodes:
                line = fid.readline().split()
                nodes[i:i + len(line), col] = [float(v) for v in line]
                i += len(line)
    return nodes * 1e-2  # cm -> m


def read_fort34(path):
    """Read fort.34 triangle topology. Returns (ntri, 3) int array of 0-based
    node indices into the fort.33 node list."""
    with open(_resolve(path, "fort.34")) as fid:
        ntri = int(fid.readline().split()[0])
        cells = np.zeros((ntri, 3), dtype=np.int32)
        for i in range(ntri):
            line = fid.readline().split()
            # line = idx n1 n2 n3 (1-based nodes); store 0-based
            cells[i, :] = [int(line[j]) - 1 for j in range(1, 4)]
    return cells


def read_fort35(path):
    """Read fort.35 triangle links. Returns dict with nghbr/side/cont (ntri, 3)
    and ixiy (ntri, 2): the B2 (ix, iy) cell index for each triangle.

    Per line: idx, 3x(nghbr, side, cont), ix, iy.
    """
    with open(_resolve(path, "fort.35")) as fid:
        ntri = int(fid.readline().split()[0])
        nghbr = np.zeros((ntri, 3), dtype=np.int32)
        side = np.zeros((ntri, 3), dtype=np.int32)
        cont = np.zeros((ntri, 3), dtype=np.int32)
        ixiy = np.zeros((ntri, 2), dtype=np.int32)
        for i in range(ntri):
            d = fid.readline().split()
            nghbr[i, :] = [int(d[1]), int(d[4]), int(d[7])]
            side[i, :] = [int(d[2]), int(d[5]), int(d[8])]
            cont[i, :] = [int(d[3]), int(d[6]), int(d[9])]
            ixiy[i, :] = [int(d[10]), int(d[11])]
    return {"nghbr": nghbr, "side": side, "cont": cont, "ixiy": ixiy}


def read_triangles(path):
    """Read the full triangle mesh (fort.33/34/35) and compute cell geometry.

    Returns a dict with:
        nodes      (nnodes, 2)  node coords [m]
        cells      (ntri, 3)    0-based node indices
        nghbr/side/cont, ixiy   from fort.35 (ixiy = B2 cell index per triangle)
        area       (ntri,)      triangle area [m^2] (shoelace)
        R_centroid (ntri,)      mean R of the 3 nodes [m]
        vol        (ntri,)      toroidal volume = 2*pi*R_centroid*area [m^3]
        mesh_extent  dict with minr/maxr/minz/maxz

    ``vol`` is the natural weight for aggregating intensive triangle densities
    onto B2 cells; the 2*pi factor cancels in a volume-weighted mean.
    """
    nodes = read_fort33(path)
    cells = read_fort34(path)
    links = read_fort35(path)

    tri = nodes[cells]                       # (ntri, 3, 2)
    x = tri[:, :, 0]
    y = tri[:, :, 1]
    area = 0.5 * np.abs(
        x[:, 0] * (y[:, 1] - y[:, 2])
        + x[:, 1] * (y[:, 2] - y[:, 0])
        + x[:, 2] * (y[:, 0] - y[:, 1])
    )
    R_centroid = x.mean(axis=1)
    vol = 2.0 * np.pi * R_centroid * area

    triangles = {
        "nodes": nodes,
        "cells": cells,
        "area": area,
        "R_centroid": R_centroid,
        "vol": vol,
        "mesh_extent": {
            "minr": nodes[:, 0].min(),
            "maxr": nodes[:, 0].max(),
            "minz": nodes[:, 1].min(),
            "maxz": nodes[:, 1].max(),
        },
    }
    triangles.update(links)
    return triangles


def _read_ft46_field(lines, cursor, name, dims):
    """Find named field starting at ``cursor`` and read prod(dims) values.

    fort.46 (ver >= 20160829) is self-describing: each block starts with a
    header line containing the field name and an element count (column 6 or 7),
    followed by the values free-wrapped across lines. Returns (field, cursor)
    with field reshaped to dims in Fortran order.
    """
    n = len(lines)
    while cursor < n and name not in lines[cursor]:
        cursor += 1
    if cursor >= n:
        raise ValueError(f"EOF reached without finding {name}.")

    toks = lines[cursor].split()
    try:
        numin = int(toks[6])
    except (IndexError, ValueError):
        numin = int(toks[7])
    expected = int(np.array(dims).prod())
    if numin != expected:
        raise ValueError(
            f"read_fort46: inconsistent element count for {name} "
            f"({numin} in file vs {expected} expected)"
        )
    cursor += 1

    field = np.zeros(numin)
    iin = 0
    while iin < numin and cursor < n:
        vals = lines[cursor].split()
        field[iin:iin + len(vals)] = [float(v) for v in vals]
        iin += len(vals)
        cursor += 1
    if iin < numin:
        raise ValueError(f"read_fort46: unexpected EOF while reading {name}")

    try:
        field = np.reshape(field, tuple(dims), order="F")
    except ValueError:
        pass
    return field, cursor


def read_fort46(path):
    """Read fort.46 neutral tallies on the triangle mesh, in SI units.

    Returns a dict keyed by field name (atoms ``*dena``, molecules ``*denm``,
    test ions ``*deni``):
        pden*  density [m^-3]
        eden*  energy density [J m^-3]
        tden*  temperature [eV]
        vxden*/vyden*/vzden*  momentum density [kg m^-2 s^-1], (R, Z, toroidal)
    plus total-neutral pdenn/edenn/tdenn (atoms + 2*molecules).

    Species are the last axis: e.g. pdena has shape (ntri, natm); index 0 is
    the main-ion neutral.
    """
    with open(_resolve(path, "fort.46")) as fid:
        lines = fid.read().splitlines()

    ntri, ver = (int(t) for t in lines[0].split()[:2])
    if ver < 20160829:
        raise ValueError(
            f"read_fort46: file version {ver} not supported "
            "(need self-describing format, ver >= 20160829)"
        )
    natm, nmol, nion = (int(t) for t in lines[1].split()[:3])

    # Skip the natm+nmol+nion species label lines, then parse fields in order.
    cursor = 2 + natm + nmol + nion

    out = {}
    specs = (("a", natm), ("m", nmol), ("i", nion))

    # densities and energy densities
    for prefix, scale in (("pden", 1e6), ("eden", 1e6 * eV)):
        for suf, ns in specs:
            name = prefix + suf
            field, cursor = _read_ft46_field(lines, cursor, name, [ntri, ns])
            out[name] = field * scale

    # momentum densities (x=R, y=Z, z=toroidal)
    for prefix in ("vxden", "vyden", "vzden"):
        for suf, ns in specs:
            name = prefix + suf
            field, cursor = _read_ft46_field(lines, cursor, name, [ntri, ns])
            out[name] = field * 1e1

    # temperatures: e/n/1.5 where density > 0 (matches eirmod_extrab25.F90)
    for suf, _ in specs:
        p, e = out["pden" + suf], out["eden" + suf]
        t = np.zeros_like(e)
        m = p > 0
        t[m] = e[m] / p[m] / eV / 1.5
        out["tden" + suf] = t

    # total neutrals: atoms + 2 * (di-atomic) molecules
    pdenn = out["pdena"].sum(axis=-1) + 2.0 * out["pdenm"].sum(axis=-1)
    edenn = out["edena"].sum(axis=-1) + 2.0 * out["edenm"].sum(axis=-1)
    tdenn = np.zeros_like(edenn)
    m = pdenn > 0
    tdenn[m] = edenn[m] / pdenn[m] / eV / 1.5
    out["pdenn"], out["edenn"], out["tdenn"] = pdenn, edenn, tdenn

    return out
