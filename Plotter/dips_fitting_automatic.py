"""Run calibrated dip fitting for every FoPra wavelength-scan data file.

This is a convenience wrapper around ``plotter.py``. It recursively finds all
``*-FoPraWavelengthScan.h5`` files in the repository/data root, skips the mirror
calibration scan by default, and launches the calibrated fitting workflow for
all remaining scans in one command.

Typical usage from the repository root:

    python Plotter/dips_fitting_automatic.py

Use a different mirror scan:

    python Plotter/dips_fitting_automatic.py --calibration-measurement 50420

Run only selected structure scans:

    python Plotter/dips_fitting_automatic.py --measurements 50397 50405 50417
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


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
        "--calibration-smooth-window",
        str(args.calibration_smooth_window),
        "--min-calibration-period-factor",
        str(args.min_calibration_period_factor),
        "--max-calibration-period-factor",
        str(args.max_calibration_period_factor),
    ]
    if args.keep_existing:
        cmd.append("--keep-existing")
    return cmd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run calibrated FoPra dip fitting for all wavelength-scan HDF5 files."
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
        help="Optional list of structure measurement numbers to fit. If omitted, fit all scans except calibration.",
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
        "--continue-on-error",
        action="store_true",
        help="Continue with the next scan if one fitting run fails.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Forward --keep-existing to plotter.py, so old output folders are not removed first.",
    )
    parser.add_argument("--prominence", type=float, default=0.02)
    parser.add_argument("--distance", type=int, default=200)
    parser.add_argument("--window-nm", type=float, default=0.5)
    parser.add_argument("--calibration-smooth-window", type=int, default=401)
    parser.add_argument("--min-calibration-period-factor", type=float, default=0.75)
    parser.add_argument("--max-calibration-period-factor", type=float, default=4.0)
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

    selected_measurements = None
    if args.measurements:
        selected_measurements = {normalize_measurement_number(item) for item in args.measurements}

    jobs = []
    for measurement, scan_path in scans:
        if selected_measurements is not None and measurement not in selected_measurements:
            continue
        if not args.include_calibration_scan and calibration_measurement == measurement:
            continue
        jobs.append((measurement, scan_path))

    print(f"Data root: {args.data_root}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mirror calibration file: {calibration_path}")
    print(f"Queued {len(jobs)} structure scans for calibrated fitting.")

    failures = []
    for index, (measurement, scan_path) in enumerate(jobs, start=1):
        print(f"\n[{index}/{len(jobs)}] Fitting measurement {measurement}: {scan_path}")
        cmd = build_plotter_command(args, measurement, scan_path, calibration_path)
        print("Command:", " ".join(str(part) for part in cmd))
        if args.dry_run:
            continue

        completed = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if completed.returncode != 0:
            failures.append((measurement, completed.returncode))
            message = f"Measurement {measurement} failed with return code {completed.returncode}."
            if args.continue_on_error:
                print(message)
                continue
            raise RuntimeError(message)

    if failures:
        print("\nFinished with failures:")
        for measurement, returncode in failures:
            print(f"  {measurement}: return code {returncode}")
        raise SystemExit(1)

    print("\nAll requested calibrated dip-fitting runs completed.")


if __name__ == "__main__":
    main()
