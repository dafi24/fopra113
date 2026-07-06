"""Shared post-processing helpers for FoPra ring-resonator fit outputs.

The legacy fitting routine writes a whitespace separated ``*_ParamFile.txt``.
This module turns those fit parameters into explicit resonance tables containing
FWHM, local FSR, finesse and group-index quantities with propagated standard
deviations.
"""

import csv
import math
import re
from pathlib import Path

import numpy as np

C_NM_THZ = 299792.458
PARAM_COLUMNS = [
    "StartWavelength[nm]",
    "EndWavelength[nm]",
    "w_r[nm]",
    "sigma(w_r)[nm]",
    "width[nm]",
    "sigma(width)[nm]",
    "Q_L[-]",
    "sigma(Q_L)[nm]",
    "Amplitude[-]",
    "sigma(Amp)[-]",
    "w_r1[nm]",
    "sigma1(w_r)[nm]",
    "width1[nm]",
    "sigma1(width)[nm]",
    "Q_L1[-]",
    "sigma1(Q_L)[nm]",
    "Amplitude1[-]",
    "sigma1(Amp)[-]",
    "b[-]",
    "c[-]",
    "R2",
]


def finite(value):
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def safe_name(value):
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("_")
    return value or "unknown"


def split_structure_measurement(output_dir):
    name = Path(output_dir).name
    match = re.match(r"^(.*)_([0-9]+)$", name)
    if match:
        return match.group(1), match.group(2)
    return name, ""


def find_param_files(plots_dir):
    plots_dir = Path(plots_dir)
    return sorted(plots_dir.rglob("*_ParamFile.txt"))


def read_param_file(param_file):
    """Read one legacy ParamFile into a list of dictionaries."""
    param_file = Path(param_file)
    rows = []
    if not param_file.is_file():
        return rows
    with open(param_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("StartWavelength"):
                continue
            parts = line.split()
            if len(parts) < len(PARAM_COLUMNS):
                continue
            row = {col: safe_float(parts[i]) for i, col in enumerate(PARAM_COLUMNS)}
            row["param_file"] = str(param_file)
            rows.append(row)
    return rows


def param_records_to_resonances(param_file, exclude_double=True, min_r2=None):
    """Convert legacy fit rows to explicit first-resonance records.

    The legacy format stores FWHM as ``width[nm]``. This function exposes it as
    ``fwhm_nm`` and ``sigma_fwhm_nm`` so later scripts can use the quantity
    explicitly.
    """
    param_file = Path(param_file)
    structure, measurement = split_structure_measurement(param_file.parent)
    records = []
    for fit_index, row in enumerate(read_param_file(param_file)):
        is_double = finite(row.get("w_r1[nm]")) or finite(row.get("width1[nm]"))
        r2 = safe_float(row.get("R2"))
        if min_r2 is not None and finite(r2) and r2 < min_r2:
            continue
        if exclude_double and is_double:
            continue

        wavelength = safe_float(row.get("w_r[nm]"))
        sigma_wavelength = abs(safe_float(row.get("sigma(w_r)[nm]")))
        fwhm = abs(safe_float(row.get("width[nm]")))
        sigma_fwhm = abs(safe_float(row.get("sigma(width)[nm]")))
        q_value = abs(safe_float(row.get("Q_L[-]")))
        sigma_q = abs(safe_float(row.get("sigma(Q_L)[nm]")))
        if not (finite(wavelength) and finite(fwhm) and fwhm > 0):
            continue

        frequency_thz = C_NM_THZ / wavelength
        sigma_frequency_thz = abs(C_NM_THZ * sigma_wavelength / (wavelength ** 2)) if finite(sigma_wavelength) else np.nan
        if not finite(q_value):
            q_value = wavelength / fwhm
            sigma_q = q_value * math.sqrt(
                (sigma_wavelength / wavelength) ** 2 + (sigma_fwhm / fwhm) ** 2
            ) if finite(sigma_wavelength) and finite(sigma_fwhm) else np.nan

        records.append(
            {
                "structure": structure,
                "measurement": measurement,
                "fit_index": fit_index,
                "param_file": str(param_file),
                "is_double_resonance": bool(is_double),
                "resonance_wavelength_nm": wavelength,
                "sigma_resonance_wavelength_nm": sigma_wavelength,
                "resonance_frequency_thz": frequency_thz,
                "sigma_resonance_frequency_thz": sigma_frequency_thz,
                "fwhm_nm": fwhm,
                "sigma_fwhm_nm": sigma_fwhm,
                "Q": q_value,
                "sigma_Q": sigma_q,
                "amplitude": safe_float(row.get("Amplitude[-]")),
                "background_b": safe_float(row.get("b[-]")),
                "background_c": safe_float(row.get("c[-]")),
                "R2": r2,
            }
        )
    return sorted(records, key=lambda item: item["resonance_wavelength_nm"])


def add_local_fsr_and_finesse(resonances, ring_length_um=100.0):
    """Add local FSR and finesse to resonance rows.

    FSR is first calculated for every neighboring resonance pair. For each dip,
    the local FSR is the nearest pair FSR for edge resonances and the average of
    the left/right pair FSRs for interior resonances. Standard deviations are
    propagated from the fitted resonance-center uncertainties.
    """
    resonances = [dict(row) for row in sorted(resonances, key=lambda item: item["resonance_wavelength_nm"])]
    if not resonances:
        return [], []

    l_nm = float(ring_length_um) * 1000.0
    pairs = []
    for i in range(len(resonances) - 1):
        left = resonances[i]
        right = resonances[i + 1]
        wl_left = left["resonance_wavelength_nm"]
        wl_right = right["resonance_wavelength_nm"]
        fsr = wl_right - wl_left
        sigma_left = safe_float(left.get("sigma_resonance_wavelength_nm"), 0.0)
        sigma_right = safe_float(right.get("sigma_resonance_wavelength_nm"), 0.0)
        sigma_fsr = math.sqrt(max(sigma_left, 0.0) ** 2 + max(sigma_right, 0.0) ** 2)
        lambda0 = 0.5 * (wl_left + wl_right)
        sigma_lambda0 = 0.5 * sigma_fsr
        ng = lambda0 ** 2 / (fsr * l_nm) if fsr > 0 else np.nan
        sigma_ng = np.nan
        if fsr > 0 and finite(sigma_fsr):
            sigma_ng = abs(ng) * math.sqrt((2.0 * sigma_lambda0 / lambda0) ** 2 + (sigma_fsr / fsr) ** 2)
        pairs.append(
            {
                "structure": left.get("structure", ""),
                "measurement": left.get("measurement", ""),
                "left_fit_index": left.get("fit_index", i),
                "right_fit_index": right.get("fit_index", i + 1),
                "left_resonance_wavelength_nm": wl_left,
                "right_resonance_wavelength_nm": wl_right,
                "center_wavelength_nm": lambda0,
                "sigma_center_wavelength_nm": sigma_lambda0,
                "center_frequency_thz": C_NM_THZ / lambda0,
                "fsr_nm": fsr,
                "sigma_fsr_nm": sigma_fsr,
                "ng_pair": ng,
                "sigma_ng_pair": sigma_ng,
            }
        )

    for i, row in enumerate(resonances):
        adjacent = []
        if i > 0:
            adjacent.append(pairs[i - 1])
        if i < len(pairs):
            adjacent.append(pairs[i])
        if adjacent:
            fsrs = np.array([item["fsr_nm"] for item in adjacent], dtype=float)
            sigmas = np.array([item["sigma_fsr_nm"] for item in adjacent], dtype=float)
            local_fsr = float(np.nanmean(fsrs))
            local_sigma = float(math.sqrt(np.nansum(sigmas ** 2)) / len(adjacent))
        else:
            local_fsr = np.nan
            local_sigma = np.nan

        row["local_fsr_nm"] = local_fsr
        row["sigma_local_fsr_nm"] = local_sigma
        row["finesse"] = np.nan
        row["sigma_finesse"] = np.nan
        fwhm = safe_float(row.get("fwhm_nm"))
        sigma_fwhm = safe_float(row.get("sigma_fwhm_nm"))
        if finite(local_fsr) and finite(fwhm) and local_fsr > 0 and fwhm > 0:
            finesse = local_fsr / fwhm
            row["finesse"] = finesse
            if finite(local_sigma) and finite(sigma_fwhm):
                row["sigma_finesse"] = abs(finesse) * math.sqrt((local_sigma / local_fsr) ** 2 + (sigma_fwhm / fwhm) ** 2)
    return resonances, pairs


def write_rows_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (float, np.floating)):
                    clean[key] = "" if not finite(value) else f"{float(value):.12g}"
                else:
                    clean[key] = value
            writer.writerow(clean)


def read_rows_csv(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def create_resonance_summary_for_param_file(param_file, ring_length_um=100.0, min_r2=None, exclude_double=True):
    param_file = Path(param_file)
    resonances = param_records_to_resonances(param_file, exclude_double=exclude_double, min_r2=min_r2)
    resonances, pairs = add_local_fsr_and_finesse(resonances, ring_length_um=ring_length_um)
    out_dir = param_file.parent
    resonance_path = out_dir / "resonance_summary.csv"
    pair_path = out_dir / "fsr_pairs.csv"
    write_rows_csv(resonance_path, resonances)
    write_rows_csv(pair_path, pairs)
    return resonance_path, pair_path, len(resonances), len(pairs)


def write_all_resonance_summaries(plots_dir, ring_length_um=100.0, min_r2=None, exclude_double=True):
    summaries = []
    for param_file in find_param_files(plots_dir):
        try:
            resonance_path, pair_path, n_res, n_pairs = create_resonance_summary_for_param_file(
                param_file,
                ring_length_um=ring_length_um,
                min_r2=min_r2,
                exclude_double=exclude_double,
            )
            summaries.append(
                {
                    "param_file": str(param_file),
                    "resonance_summary": str(resonance_path),
                    "fsr_pairs": str(pair_path),
                    "n_resonances": n_res,
                    "n_fsr_pairs": n_pairs,
                }
            )
        except Exception as exc:
            summaries.append({"param_file": str(param_file), "error": str(exc)})
    return summaries
