from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


WL_DATASET_CANDIDATES = (
    "wavelength_scan.wl",
    "wavelength_scan.WL",
    "wavelength_scan.wavelength",
)
RR_DATASET_CANDIDATES = (
    "wavelength_scan.RR.mean",
    "wavelength_scan.rr.mean",
    "wavelength_scan.relative_reflection.mean",
    "wavelength_scan.RelativeReflection.mean",
)
R_DATASET_CANDIDATES = (
    "wavelength_scan.R.mean",
    "wavelength_scan.r.mean",
    "wavelength_scan.reflection.mean",
)
E_DATASET_CANDIDATES = (
    "wavelength_scan.E.mean",
    "wavelength_scan.e.mean",
    "wavelength_scan.excitation.mean",
)
S_DATASET_CANDIDATES = (
    "wavelength_scan.S.mean",
    "wavelength_scan.s.mean",
)
STRUCTURE_DATASET_CANDIDATES = (
    "wavelength_scan.structure_string",
    "wavelength_scan.structure",
)


def clean_sample_name(sample):
    """Return a filesystem-friendly structure name from the HDF5 byte/string value."""
    if isinstance(sample, bytes):
        sample = sample.decode("utf-8", errors="replace")
    sample = str(sample)
    if sample.startswith("b'") and sample.endswith("'"):
        sample = sample[2:-1]
    if sample.startswith('b"') and sample.endswith('"'):
        sample = sample[2:-1]
    return sample.strip().replace("/", "_").replace("\\", "_") or "unknown_structure"


def normalize_measurement_number(measurement_number):
    """Normalize inputs such as 50420, '50420', or '000050420' to '50420'."""
    digits = "".join(ch for ch in str(measurement_number) if ch.isdigit())
    if not digits:
        raise ValueError(f"Invalid measurement number: {measurement_number!r}")
    return str(int(digits))


def measurement_filename(measurement_number, suffix="-FoPraWavelengthScan.h5"):
    """Return the Artiq/FoPra file name for a measurement number."""
    number = normalize_measurement_number(measurement_number)
    return f"{int(number):09d}{suffix}"


def find_measurement_file(measurement_number, data_root=".", suffix="-FoPraWavelengthScan.h5"):
    """Find a measurement HDF5 file inside the repository/data root.

    The original script used an absolute machine-specific path. The uploaded
    repository stores the data under folders such as ``2026-05-07/15`` and
    ``2026-05-07/16``. This helper searches recursively from ``data_root``.
    """
    data_root = Path(data_root).expanduser().resolve()
    wanted = measurement_filename(measurement_number, suffix=suffix)
    direct_matches = list(data_root.rglob(wanted))
    if direct_matches:
        return direct_matches[0]

    number = normalize_measurement_number(measurement_number)
    loose_matches = sorted(data_root.rglob(f"*{number}{suffix}"))
    if loose_matches:
        return loose_matches[0]

    raise FileNotFoundError(
        f"Could not find {wanted} below {data_root}. "
        "Use --data-root or --file to point at the measurement file."
    )


def _first_existing_dataset(group, candidates):
    for key in candidates:
        if key in group:
            return key, np.asarray(group[key])
    return None, None


def _find_dataset_ending(group, endings):
    lower_endings = tuple(ending.lower() for ending in endings)
    for key in group.keys():
        if key.lower().endswith(lower_endings):
            return key, np.asarray(group[key])
    return None, None


def _scalar_string(value, fallback):
    try:
        array = np.asarray(value)
        if array.shape == ():
            value = array.item()
        elif array.size > 0:
            value = array.flat[0]
    except Exception:
        return fallback
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def open_h5_robust(path):
    """Open a FoPra wavelength-scan HDF5 file using all known dataset variants.

    Some uploaded scans do not contain ``wavelength_scan.RR.mean``. Older or
    reference scans may only contain ``wavelength_scan.S.mean`` or separate
    reflected/excitation channels. This reader keeps the automatic batch script
    from crashing on those files:

    1. prefer a precomputed relative-reflection dataset, if present;
    2. otherwise compute R/E when both channels are available;
    3. otherwise fall back to S.mean.
    """
    path = Path(path)
    with h5py.File(path, "r") as fh:
        if "datasets" not in fh:
            raise KeyError(f"{path} has no 'datasets' group.")
        datasets = fh["datasets"]
        available = sorted(datasets.keys())

        wl_key, WL = _first_existing_dataset(datasets, WL_DATASET_CANDIDATES)
        if WL is None:
            wl_key, WL = _find_dataset_ending(datasets, (".wl", ".wavelength"))
        if WL is None:
            raise KeyError(
                f"{path} has no recognized wavelength dataset. Available datasets: {available}"
            )

        signal_key, signal = _first_existing_dataset(datasets, RR_DATASET_CANDIDATES)
        if signal is None:
            signal_key, signal = _find_dataset_ending(datasets, (".rr.mean", ".relative_reflection.mean"))

        if signal is None:
            r_key, reflected = _first_existing_dataset(datasets, R_DATASET_CANDIDATES)
            e_key, excitation = _first_existing_dataset(datasets, E_DATASET_CANDIDATES)
            if reflected is not None and excitation is not None:
                n = min(len(reflected), len(excitation))
                reflected = np.asarray(reflected[:n], dtype=float)
                excitation = np.asarray(excitation[:n], dtype=float)
                if np.any(np.isclose(excitation, 0.0)):
                    raise ZeroDivisionError(f"{path}: excitation channel {e_key} contains zero values.")
                signal = np.abs(reflected) / np.abs(excitation)
                signal_key = f"abs({r_key})/abs({e_key})"

        if signal is None:
            signal_key, signal = _first_existing_dataset(datasets, S_DATASET_CANDIDATES)
        if signal is None:
            signal_key, signal = _find_dataset_ending(datasets, (".s.mean", ".r.mean"))

        if signal is None:
            raise KeyError(
                f"{path} has no recognized reflection signal dataset. Available datasets: {available}"
            )

        sample_key, sample_value = _first_existing_dataset(datasets, STRUCTURE_DATASET_CANDIDATES)
        sample = _scalar_string(sample_value, path.stem) if sample_value is not None else path.stem

    WL = np.asarray(WL, dtype=float).reshape(-1)
    signal = np.asarray(signal, dtype=float).reshape(-1)
    n_points = min(WL.size, signal.size)
    WL = WL[:n_points]
    signal = signal[:n_points]

    # Preserve the behavior of the original analysis script, which discarded
    # the final wavelength/signal point from Artiq wavelength scans.
    if n_points > 1:
        WL = WL[:-1]
        signal = signal[:-1]

    print(f"Loaded {path.name}: wavelength={wl_key}, signal={signal_key}")
    return WL, signal, sample


def supports_wavelength_scan(path):
    """Return (True, '') when open_h5_robust can read the scan, else (False, reason)."""
    try:
        open_h5_robust(path)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _odd_window(window, length):
    window = int(window)
    if window < 3:
        window = 3
    if window > length:
        window = length if length % 2 == 1 else length - 1
    if window % 2 == 0:
        window += 1
    return max(3, window)


def moving_average(y, window):
    """Centered moving average used only to suppress short variations before the sine fit."""
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return y.copy()
    window = _odd_window(window, y.size)
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(y, pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def mirror_sine_model(wavelength, offset, amplitude, period_nm, phase, slope, center_nm):
    centered = np.asarray(wavelength, dtype=float) - center_nm
    return offset + amplitude * np.sin(2 * np.pi * centered / period_nm + phase) + slope * centered


def fit_mirror_sine(WL, RR, smooth_window=401, min_period_factor=0.75, max_period_factor=4.0):
    """Fit the calibration-mirror trace with one long-period sinusoidal trend.

    The fit is intentionally constrained to long periods and is performed on a
    smoothed copy of the mirror data so the baseline captures only the slow
    interference/envelope trend visible in the mirror measurement, not the short
    variations/noise/resonance-scale features.
    """
    WL = np.asarray(WL, dtype=float)
    RR = np.asarray(RR, dtype=float)
    finite = np.isfinite(WL) & np.isfinite(RR)
    WL = WL[finite]
    RR = RR[finite]
    if WL.size < 10:
        raise ValueError("Need at least 10 finite calibration points for the sine fit.")

    order = np.argsort(WL)
    WL = WL[order]
    RR = RR[order]
    center_nm = float(np.mean(WL))
    span_nm = float(np.max(WL) - np.min(WL))
    if span_nm <= 0:
        raise ValueError("Calibration wavelengths must span a non-zero range.")

    smooth = moving_average(RR, smooth_window)
    y_min = float(np.nanpercentile(smooth, 5))
    y_max = float(np.nanpercentile(smooth, 95))
    offset0 = float(np.nanmedian(smooth))
    amp0 = max((y_max - y_min) / 2.0, 1e-6)
    period0 = 1.2 * span_nm
    max_idx = int(np.nanargmax(smooth))
    phase0 = np.pi / 2 - 2 * np.pi * (WL[max_idx] - center_nm) / period0
    slope0 = float((smooth[-1] - smooth[0]) / span_nm)

    min_period = max(1e-9, min_period_factor * span_nm)
    max_period = max(max_period_factor * span_nm, min_period * 1.01)
    amp_bound = max(10 * amp0, abs(y_max - y_min), 1e-3)
    slope_bound = max(10 * abs(slope0), 10 * amp0 / span_nm, 1e-6)
    lower = [y_min - amp_bound, -amp_bound, min_period, -4 * np.pi, -slope_bound]
    upper = [y_max + amp_bound, amp_bound, max_period, 4 * np.pi, slope_bound]

    def model_for_fit(x, offset, amplitude, period_nm, phase, slope):
        return mirror_sine_model(x, offset, amplitude, period_nm, phase, slope, center_nm)

    popt, pcov = curve_fit(
        model_for_fit,
        WL,
        smooth,
        p0=[offset0, amp0, period0, phase0, slope0],
        bounds=(lower, upper),
        maxfev=20000,
    )

    fit = model_for_fit(WL, *popt)
    params = {
        "offset": float(popt[0]),
        "amplitude": float(popt[1]),
        "period_nm": float(popt[2]),
        "phase": float(popt[3]),
        "slope": float(popt[4]),
        "center_nm": center_nm,
        "smooth_window": int(_odd_window(smooth_window, WL.size)),
        "min_period_nm": float(min_period),
        "max_period_nm": float(max_period),
    }
    return WL, RR, smooth, fit, params, pcov


def evaluate_mirror_sine(WL, params):
    return mirror_sine_model(
        WL,
        params["offset"],
        params["amplitude"],
        params["period_nm"],
        params["phase"],
        params["slope"],
        params["center_nm"],
    )


def apply_mirror_calibration(
    WL,
    RR,
    mirror_WL,
    mirror_RR,
    output_dir=None,
    smooth_window=401,
    min_period_factor=0.75,
    max_period_factor=4.0,
    plot=True,
):
    """Divide raw structure data by a long-period sine fit to mirror data.

    Returns ``RR_calibrated, baseline_on_WL, params`` and optionally saves the
    fitted mirror curve and calibrated trace to ``output_dir``.
    """
    mirror_WL_fit, mirror_RR_sorted, mirror_smooth, mirror_fit, params, _ = fit_mirror_sine(
        mirror_WL,
        mirror_RR,
        smooth_window=smooth_window,
        min_period_factor=min_period_factor,
        max_period_factor=max_period_factor,
    )

    WL = np.asarray(WL, dtype=float)
    RR = np.asarray(RR, dtype=float)
    baseline_on_WL = evaluate_mirror_sine(WL, params)
    if np.any(np.isclose(baseline_on_WL, 0.0)):
        raise ZeroDivisionError("Mirror calibration sine fit evaluates to zero at one or more data points.")
    RR_calibrated = RR / baseline_on_WL

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            output_dir / "mirror_calibration_fit.txt",
            np.column_stack((mirror_WL_fit, mirror_RR_sorted, mirror_smooth, mirror_fit)),
            header="Wavelength[nm] MirrorRR_raw MirrorRR_smoothed MirrorRR_sine_fit",
        )
        np.savetxt(
            output_dir / "calibrated_trace.txt",
            np.column_stack((WL, RR, baseline_on_WL, RR_calibrated)),
            header="Wavelength[nm] RR_raw MirrorSineFit_at_Wavelength RR_calibrated",
        )
        with open(output_dir / "mirror_calibration_parameters.txt", "w", encoding="utf-8") as f:
            for key, value in params.items():
                f.write(f"{key}: {value}\n")

        if plot:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(mirror_WL_fit, mirror_RR_sorted, ".", markersize=2, label="Mirror data")
            ax.plot(mirror_WL_fit, mirror_smooth, linewidth=1.5, label="Smoothed mirror data")
            ax.plot(mirror_WL_fit, mirror_fit, linewidth=2.0, label="Long-period sine fit")
            ax.set_xlabel("Wavelength [nm]")
            ax.set_ylabel("Relative reflection [-]")
            ax.set_title("Mirror reflection calibration")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "mirror_calibration_fit.png", dpi=600)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(WL, RR, ".", markersize=2, label="Raw structure data")
            ax.plot(WL, baseline_on_WL, linewidth=2.0, label="Mirror sine baseline")
            ax.plot(WL, RR_calibrated, ".", markersize=2, label="Calibrated data")
            ax.set_xlabel("Wavelength [nm]")
            ax.set_ylabel("Relative reflection [-]")
            ax.set_title("Mirror-calibrated structure data")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "calibrated_spectrum.png", dpi=600)
            plt.close(fig)

    return RR_calibrated, baseline_on_WL, params
