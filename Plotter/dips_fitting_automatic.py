"""Run calibrated dip fitting for every FoPra wavelength-scan data file.

This is a convenience wrapper around ``plotter.py``. It recursively finds all
``*-FoPraWavelengthScan.h5`` files in the repository/data root, skips the mirror
calibration scan by default, and launches the calibrated fitting workflow for
all remaining readable scans in one command.

Typical usage from the repository root or from Plotter/:

    python Plotter/dips_fitting_automatic.py
    python3 dips_fitting_automatic.py

Run only selected structure scans:

    python Plotter/dips_fitting_automatic.py --measurements 50397 50405 50417
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from calibration import open_h5_robust


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PLOTTER_SCRIPT = SCRIPT_DIR / "plotter.py"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "Plots"
DEFAULT_CALIBRATION_MEASUREMENT = "50420"
SCAN_NAME_RE = re.compile(r"^0*(\d+)-FoPraWavelengthScan\.h5$")


def normalize_measurement_number(value):
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if not digits:
        raise ValueError(f"Invalid measurement number: {value!r}")
    return str(int(digits))


def measurement_from_path(path):
    match = SCAN_NAME_RE.match(Path(path).name)
    if match is None:
        return None
    return normalize_measurement_number(match.group(1))


def discover_wavelength_scans(data_root):
    scans = []
    for path in Path(data_root).expanduser().resolve().rglob("*-FoPraWavelengthScan.h5"):
        measurement = measurement_from_path(path)
        if measurement is None:
            continue
        scans.append((measurement, path))
    return sorted(scans, key=lambda item: int(item[0]))


def resolve_calibration_file(scans, calibration_measurement, calibration_file):
    if calibration_file:
        path = Path(calibration_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Calibration file does not exist: {path}")
        return path, measurement_from_path(path)

    calibration_measurement = normalize_measurement_number(calibration_measurement)
    matches = [path for measurement, path in scans if measurement == calibration_measurement]
    if not matches:
        raise FileNotFoundError(
            f"Could not find mirror calibration measurement {calibration_measurement}. "
            "Pass --calibration-file or --calibration-measurement."
        )
    return matches[0], calibration_measurement


def validate_scan(path):
    try:
        open_h5_robust(path)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def build_plotter_command(args, measurement, scan_path, calibration_path):
    cmd = [
        sys.executable,
        str(PLOTTER_SCRIPT),
        "--measurement",
        measurement,
        "--file",
        str(scan_path),
        "--calibration-file",
        str(calibration_path),
        "--data-root",
        str(args.data_root),
        "--output-dir",
        str(args.output_dir),
        "--prominence",
        str(args.prominence),
        "--distance",
        str(args.distance),
        "--window-nm",
        str(args.window_nm),
        "--detection-smooth-window",
        str(args.detection_smooth_window),
        "--min-dip-depth-fraction",
        str(args.min_dip_depth_fraction),
        "--min-dip-depth-sigma",
        str(args.min_dip_depth_sigma),
        "--duplicate-merge-nm",
        str(args.duplicate_merge_nm),
        "--edge-reject-fraction",
        str(args.edge_reject_fraction),
        "--calibration-smooth-window",
        str(args.calibration_smooth_window),
        "--min-calibration-period-factor",
        str(args.min_calibration_period_factor),
        "--max-calibration-period-factor",
        str(args.max_calibration_period_factor),
        "--overview-dpi",
        str(args.overview_dpi),
        "--fit-plot-dpi",
        str(args.fit_plot_dpi),
        "--no-interactive",
    ]
    if args.keep_existing:
        cmd.append("--keep-existing")
    if args.stop_on_fit_error:
        cmd.append("--stop-on-fit-error")
    return cmd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run calibrated FoPra dip fitting for all readable wavelength-scan HDF5 files."
    )
    parser.add_argument(
        "--data-root",
        default=str(REPO_ROOT),
        help="Root folder to search recursively for *-FoPraWavelengthScan.h5 files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output folder passed to plotter.py. Defaults to Plotter/Plots.",
    )
    parser.add_argument(
        "--calibration-measurement",
        default=DEFAULT_CALIBRATION_MEASUREMENT,
        help="Measurement number of the mirror calibration scan. Defaults to 50420.",
    )
    parser.add_argument(
        "--calibration-file",
        default=None,
        help="Direct path to the mirror calibration .h5 file. Overrides --calibration-measurement.",
    )
    parser.add_argument(
        "--measurements",
        nargs="*",
        default=None,
        help="Optional list of structure measurement numbers to fit. If omitted, fit all readable scans except calibration.",
    )
    parser.add_argument(
        "--include-calibration-scan",
        action="store_true",
        help="Also run plotter.py on the mirror scan itself. By default it is skipped.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plotter.py commands without executing them.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch when one measurement subprocess fails. By default failures are logged and the batch continues.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Deprecated compatibility flag; continuing is now the default.",
    )
    parser.add_argument(
        "--stop-on-fit-error",
        action="store_true",
        help="Forward --stop-on-fit-error to plotter.py. By default individual failed peak fits are skipped.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Forward --keep-existing to plotter.py, so old output folders are not removed first.",
    )
    parser.add_argument("--prominence", type=float, default=0.08)
    parser.add_argument("--distance", type=int, default=200)
    parser.add_argument("--window-nm", type=float, default=0.5)
    parser.add_argument("--detection-smooth-window", type=int, default=11)
    parser.add_argument("--min-dip-depth-fraction", type=float, default=0.08)
    parser.add_argument("--min-dip-depth-sigma", type=float, default=5.0)
    parser.add_argument("--duplicate-merge-nm", type=float, default=0.35)
    parser.add_argument("--edge-reject-fraction", type=float, default=0.15)
    parser.add_argument("--calibration-smooth-window", type=int, default=401)
    parser.add_argument("--min-calibration-period-factor", type=float, default=0.75)
    parser.add_argument("--max-calibration-period-factor", type=float, default=4.0)
    parser.add_argument("--overview-dpi", type=int, default=300)
    parser.add_argument("--fit-plot-dpi", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    args.data_root = Path(args.data_root).expanduser().resolve()
    args.output_dir = Path(args.output_dir).expanduser().resolve()

    scans = discover_wavelength_scans(args.data_root)
    if not scans:
        raise FileNotFoundError(f"No *-FoPraWavelengthScan.h5 files found below {args.data_root}")

    calibration_path, calibration_measurement = resolve_calibration_file(
        scans,
        args.calibration_measurement,
        args.calibration_file,
    )
    calibration_ok, calibration_reason = validate_scan(calibration_path)
    if not calibration_ok:
        raise RuntimeError(f"Mirror calibration file is not readable: {calibration_reason}")

    selected_measurements = None
    if args.measurements:
        selected_measurements = {normalize_measurement_number(item) for item in args.measurements}

    jobs = []
    skipped = []
    for measurement, scan_path in scans:
        if selected_measurements is not None and measurement not in selected_measurements:
            continue
        if not args.include_calibration_scan and calibration_measurement == measurement:
            skipped.append((measurement, scan_path, "calibration scan"))
            continue
        ok, reason = validate_scan(scan_path)
        if not ok:
            skipped.append((measurement, scan_path, reason))
            continue
        jobs.append((measurement, scan_path))

    print(f"Data root: {args.data_root}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mirror calibration file: {calibration_path}")
    print(f"Queued {len(jobs)} readable structure scans for calibrated fitting.")
    if skipped:
        print("Skipped scans:")
        for measurement, scan_path, reason in skipped:
            print(f"  {measurement}: {scan_path} ({reason})")

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    failures = []
    for index, (measurement, scan_path) in enumerate(jobs, start=1):
        print(f"\n[{index}/{len(jobs)}] Fitting measurement {measurement}: {scan_path}")
        cmd = build_plotter_command(args, measurement, scan_path, calibration_path)
        print("Command:", " ".join(str(part) for part in cmd))
        if args.dry_run:
            continue

        completed = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
        if completed.returncode != 0:
            failures.append((measurement, completed.returncode))
            message = f"Measurement {measurement} failed with return code {completed.returncode}."
            if args.stop_on_error:
                raise RuntimeError(message)
            print(message)
            print("Continuing with the next measurement. Use --stop-on-error to fail immediately.")
            continue

    if failures:
        print("\nFinished with measurement-level failures:")
        for measurement, returncode in failures:
            print(f"  {measurement}: return code {returncode}")
        print("Batch completed despite failures.")
    else:
        print("\nAll requested calibrated dip-fitting runs completed.")


if __name__ == "__main__":
    main()
