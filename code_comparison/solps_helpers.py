import numpy as np


def _read_field(fid, fieldname, dims, cast, validate_dims=True):
    """Read a single SOLPS text field from an open file handle.

    Parameters
    ----------
    fid : file object
        Open file positioned anywhere before the requested field.
    fieldname : str
        Name of the field to search for in the file.
    dims : int or list[int]
        Expected output shape. Scalars may be passed as a single integer.
    cast : callable
        Element-wise converter, typically int or float.
    validate_dims : bool
        If True, check that the field size declared in the file matches dims.

    Returns
    -------
    numpy.ndarray
        Parsed field data, reshaped in Fortran order for array-valued fields.
    """

    line = fid.readline()
    while fieldname not in line:
        if line == "":
            raise ValueError(f"EOF reached without finding {fieldname}.")
        line = fid.readline()

    numin = int(line.split()[2])
    expected = int(np.array(dims).prod())
    if validate_dims and numin != expected:
        raise ValueError(f"inconsistent number of input elements for {fieldname}")

    field = np.zeros(numin)
    iin = 0
    while iin < numin:
        values = fid.readline().split()
        if not values:
            break
        for iil, value in enumerate(values):
            field[iin + iil] = cast(value)
        iin += len(values)

    if iin != numin:
        raise ValueError(f"unexpected EOF while reading {fieldname}")

    if isinstance(dims, list) and len(dims) > 1:
        field = np.reshape(field, tuple(dims), order="F")

    return field


def read_ifield(fid=None, fieldname=None, dims=None):
    """Read an integer SOLPS field from a text file."""

    validate_dims = fieldname != "nx,ny"
    return _read_field(fid, fieldname, dims, int, validate_dims=validate_dims)


def read_rfield(fid=None, fieldname=None, dims=None):
    """Read a floating-point SOLPS field from a text file."""

    return _read_field(fid, fieldname, dims, float)