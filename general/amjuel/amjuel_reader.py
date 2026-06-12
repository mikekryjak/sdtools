from __future__ import annotations

import re
from pathlib import Path

import numpy as np


_DFLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:D\s*[+-]?\s*\d+)"
_COEFF_ROW_RE = re.compile(
    rf"^\s*(?P<t_index>\d+)\s+"
    rf"(?P<v0>{_DFLOAT_PATTERN})\s+"
    rf"(?P<v1>{_DFLOAT_PATTERN})\s+"
    rf"(?P<v2>{_DFLOAT_PATTERN})\s*$"
)
_SECTION_RE = re.compile(r"\\section\{(?P<header>.*?)\}(?P<body>.*?)(?=(?:\\section\{|\Z))", re.S)
_SUBSECTION_RE = re.compile(
    r"\\subsection\{\s*Reaction\s+(?P<reaction>\S+)(?P<header>.*?)\}(?P<body>.*?)(?=(?:\\subsection\{|\\section\{|\Z))",
    re.S,
)
_BOUND_RE = re.compile(
    rf"(?P<key>[A-Z]\d(?:MIN|MAX))\s*=\s*(?P<value>{_DFLOAT_PATTERN})\s*(?P<unit>[^\n]*)"
)


def _parse_d_float(value: str) -> float:
    """Convert a Fortran-style float string.

    Inputs:
        value: String containing a number that may use D exponents.

    Returns:
        Parsed floating-point value.
    """
    value = value.replace("D", "E")
    value = re.sub(
        r"E\s*([+-]?)\s*(\d+)",
        lambda match: f"E{match.group(1) or '+'}{match.group(2)}",
        value,
    )
    return float(value)


def _positive_array(name: str, value: float | np.ndarray) -> np.ndarray:
    """Convert an input to a positive float array.

    Inputs:
        name: Label used in the error message.
        value: Scalar or array to validate.

    Returns:
        Numpy array of positive float values.
    """
    value = np.asarray(value, dtype=float)
    if np.any(value <= 0.0):
        raise ValueError(f"{name} must be strictly positive because AMJUEL uses logarithms.")
    return value


def extract_fit(name: str, tex_path: str | Path) -> dict[str, object]:
    """Extract one AMJUEL fit block from the tex file.

    Inputs:
        name: Fit name such as "H.4 2.1.5".
        tex_path: Path to the AMJUEL tex source.

    Returns:
        Dictionary containing metadata, coefficients, and fit bounds.
    """
    name = " ".join(name.split())
    name_match = re.fullmatch(r"(H\.\d+)\s+(\S+)", name)
    if name_match is None:
        raise ValueError(
            "AMJUEL names must look like 'H.4 2.1.5' or 'H.10 2.1.5'."
        )
    family, reaction = name_match.groups()
    text = Path(tex_path).read_text(encoding="utf-8")

    for section_match in _SECTION_RE.finditer(text):
        section_header = re.sub(r"\\cite\{[^}]*\}", "", section_match.group("header"))
        section_header = " ".join(section_header.replace("\\\\", " ").split())
        family_match = re.match(r"(H\.\d+)\s*:", section_header)
        if family_match is None or family_match.group(1) != family:
            continue

        compact_header = section_header.replace(" ", "")
        if "(E,T)" in compact_header:
            second_param = "E"
        elif re.search(r"\(n[^,]*,T", compact_header):
            second_param = "Ne"
        else:
            raise ValueError(f"Could not infer the second AMJUEL parameter from section header: {section_header}")

        for subsection_match in _SUBSECTION_RE.finditer(section_match.group("body")):
            if subsection_match.group("reaction") != reaction:
                continue

            subsection_body = subsection_match.group("body")
            marker = "\\begin{small}\\begin{verbatim}"
            if marker not in subsection_body:
                continue

            description, coeff_text = subsection_body.split(marker, maxsplit=1)
            coeff_text, _ = coeff_text.split("\\end{verbatim}", maxsplit=1)
            description = re.sub(r"\\cite\{[^}]*\}", "", description)
            description = " ".join(description.replace("\\\\", " ").split())

            coeffs = np.zeros((9, 9), dtype=float)
            bounds: dict[str, float] = {}
            bound_units: dict[str, str] = {}
            current_e_indices: list[int] | None = None
            parsed_rows = 0

            for line in coeff_text.splitlines():
                if "E-Index:" in line:
                    current_e_indices = [int(index) for index in re.findall(r"\d+", line)]
                    continue

                bound_match = _BOUND_RE.match(line.strip())
                if bound_match is not None:
                    key = bound_match.group("key")
                    bounds[key] = _parse_d_float(bound_match.group("value"))
                    bound_units[key] = " ".join(bound_match.group("unit").split())
                    continue

                row_match = _COEFF_ROW_RE.match(line)
                if row_match is None or current_e_indices is None:
                    continue

                row_values = [
                    _parse_d_float(row_match.group("v0")),
                    _parse_d_float(row_match.group("v1")),
                    _parse_d_float(row_match.group("v2")),
                ]
                t_index = int(row_match.group("t_index"))
                for e_index, value in zip(current_e_indices, row_values, strict=True):
                    coeffs[t_index, e_index] = value
                parsed_rows += 1

            if parsed_rows != 27:
                raise ValueError(f"Expected 27 coefficient rows, found {parsed_rows}.")

            return {
                "name": name,
                "family": family,
                "reaction": reaction,
                "description": description,
                "second_param": second_param,
                "coeffs": coeffs,
                "bounds": bounds,
                "bound_units": bound_units,
            }

        raise KeyError(f"Reaction {name} was not found in {tex_path}.")

    raise KeyError(f"Family {family} was not found in {tex_path}.")


def evaluate_from_coefficients(
    coeffs: np.ndarray,
    Te: float | np.ndarray,
    second_value: float | np.ndarray | None = None,
    *,
    second_param: str = "Ne",
    mode: str = "default",
    pop_ratio: bool = False,
) -> float | np.ndarray:
    """Evaluate a 2D AMJUEL polynomial from a coefficient table.

    Inputs:
        coeffs: 9x9 AMJUEL coefficient array.
        Te: Electron temperature in eV.
        second_value: Density or energy axis value for the fit.
        second_param: Name of the second axis, usually "Ne" or "E".
        mode: Evaluation mode, either "default" or "coronal".
        pop_ratio: If true, skip the final 1e-6 conversion.

    Returns:
        Scalar or array of evaluated rate values.
    """
    if mode not in {"default", "coronal"}:
        raise ValueError("mode must be 'default' or 'coronal'.")

    log_te = np.log(_positive_array("Te", Te))

    if mode == "coronal":
        log_rate = np.zeros_like(log_te, dtype=float)
        for t_index in range(coeffs.shape[0]):
            log_rate += coeffs[t_index, 0] * np.power(log_te, t_index)
    else:
        if second_value is None:
            raise ValueError("A second parameter value is required in default mode.")

        second_array = _positive_array(second_param, second_value)
        log_te, second_array = np.broadcast_arrays(log_te, second_array)

        if second_param == "Ne":
            second_array = second_array * 1.0e-14

        log_second = np.log(second_array)
        log_rate = np.zeros_like(log_te, dtype=float)
        for t_index in range(coeffs.shape[0]):
            te_term = np.power(log_te, t_index)
            for e_index in range(coeffs.shape[1]):
                log_rate += coeffs[t_index, e_index] * te_term * np.power(log_second, e_index)

    rate = np.exp(log_rate)
    if not pop_ratio:
        rate = rate * 1.0e-6

    return float(rate) if np.ndim(rate) == 0 else rate


def amjuel_2d(
    name: str,
    Te: float | np.ndarray,
    Ne: float | np.ndarray | None = None,
    *,
    E: float | np.ndarray | None = None,
    tex_path: str | Path,
    mode: str = "default",
    pop_ratio: bool = False,
) -> float | np.ndarray:
    """Load a fit from tex and evaluate it for Te and Ne or E.

    Inputs:
        name: Fit name such as "H.4 2.1.5".
        Te: Electron temperature in eV.
        Ne: Electron density in m^-3 for density-based fits.
        E: Beam or process energy in eV for energy-based fits.
        tex_path: Path to the AMJUEL tex source.
        mode: Evaluation mode, either "default" or "coronal".
        pop_ratio: If true, skip the final 1e-6 conversion.

    Returns:
        Scalar or array of evaluated rate values.
    """
    fit = extract_fit(name, tex_path=tex_path)
    second_param = fit["second_param"]
    coeffs = fit["coeffs"]

    if second_param == "Ne":
        if E is not None:
            raise ValueError(f"{fit['name']} is density-based. Pass Ne, not E.")
        return evaluate_from_coefficients(
            coeffs,
            Te,
            None if mode == "coronal" else Ne,
            second_param="Ne",
            mode=mode,
            pop_ratio=pop_ratio,
        )

    if Ne is not None:
        raise ValueError(f"{fit['name']} is energy-based. Pass E, not Ne.")
    return evaluate_from_coefficients(
        coeffs,
        Te,
        E,
        second_param="E",
        mode=mode,
        pop_ratio=pop_ratio,
    )


evaluate_rate = amjuel_2d