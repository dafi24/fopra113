import argparse
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from test_callback import SelectiveFitter, assign_filename
from functions_double import Pixner_fit, detect_peaks
from calibration import (
    apply_mirror_calibration,
    clean_sample_name,
    find_measurement_file,
    normalize_measurement_number,
    open_h5_robust,
)

# Defaults chosen for the repository layout uploaded with the FoPra data.
# Override from the command line, for example:
#   python Plotter/plotter.py --measurement 50407 --calibration-measurement 50420
DEFAULT_MEASUREMENT_NUMBER = "50405"
DEFAULT_CALIBRATION_MEASUREMENT_NUMBER = "50420"


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_ROOT = REPO_ROOT
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "Plots"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot and fit FoPra ring-resonator wavelength scans. "
            "Optionally calibrate the raw trace with a long-period sine fit to a mirror scan."
        )
    )
    parser.add_argument(
        "--measurement",
        default=DEFAULT_MEASUREMENT_NUMBER,
        help="FoPra wavelength-scan measurement number for the structure data, e.g. 50405.",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Direct path to the structure .h5 file. If omitted, the script searches --data-root recursively.",
    )
    parser.add_argument(
        "--calibration-measurement",
        default=DEFAULT_CALIBRATION_MEASUREMENT_NUMBER,
        help=(
            "FoPra wavelength-scan measurement number for the calibration mirror data. "
            "Use 'none' or --no-calibration to skip mirror calibration."
        ),
    )
    parser.add_argument(
        "--calibration-file",
        default=None,
        help="Direct path to the calibration mirror .h5 file. Overrides --calibration-measurement.",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Disable mirror calibration and fit the raw structure data directly.",
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Repository/data root. Defaults to the repository root inferred from Plotter/plotter.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for plots and parameter files. Defaults to Plotter/Plots.",
    )
    parser.add_argument(
        "--prominence",
        type=float,
        default=0.02,
        help="Prominence passed to scipy.signal.find_peaks after normalization. Increase for noisier data.",
    )
    parser.add_argument(
        "--distance",
        type=int,
        default=200,
        help="Minimum distance, in data points, between detected dips.",
    )
    parser.add_argument(
        "--window-nm",
        type=float,
        default=0.5,
        help="Half-width of the fitting window around each detected dip in nm.",
    )
    parser.add_argument(
        "--calibration-smooth-window",
        type=int,
        default=401,
        help="Moving-average window used before fitting the mirror sine trend.",
    )
    parser.add_argument(
        "--min-calibration-period-factor",
        type=float,
        default=0.75,
        help="Lower bound for the sine period as a fraction of the mirror wavelength span.",
    )
    parser.add_argument(
        "--max-calibration-period-factor",
        type=float,
        default=4.0,
        help="Upper bound for the sine period as a fraction of the mirror wavelength span.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete an existing output folder before processing.",
    )
    return parser.parse_args()


def resolve_measurement_path(measurement, direct_file, data_root):
    if direct_file:
        path = Path(direct_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Provided file does not exist: {path}")
        return path
    return find_measurement_file(measurement, data_root=data_root)


def plot_overview(WL, RR, title, save_path):
    fig = plt.figure(num=None, figsize=(10, 4))
    ax = fig.add_subplot(111)
    graph, = ax.plot(WL, RR, ".")
    ax.set_title(title)
    ax.set_ylabel("Relative reflection [-]")
    ax.set_xlabel("Wavelength [nm]")
    fig.tight_layout()
    fig.savefig(save_path, dpi=1200)
    return fig, ax, graph


def write_processing_config(output_dir, args, data_path, calibration_path, calibration_params):
    with open(output_dir / "processing_config.txt", "w", encoding="utf-8") as f:
        f.write(f"structure_measurement: {args.measurement}\n")
        f.write(f"structure_file: {data_path}\n")
        f.write(f"calibration_measurement: {args.calibration_measurement}\n")
        f.write(f"calibration_file: {calibration_path}\n")
        f.write(f"data_root: {Path(args.data_root).expanduser().resolve()}\n")
        f.write(f"prominence: {args.prominence}\n")
        f.write(f"distance: {args.distance}\n")
        f.write(f"window_nm: {args.window_nm}\n")
        f.write(f"calibration_smooth_window: {args.calibration_smooth_window}\n")
        f.write(f"min_calibration_period_factor: {args.min_calibration_period_factor}\n")
        f.write(f"max_calibration_period_factor: {args.max_calibration_period_factor}\n")
        if calibration_params:
            f.write("\n[mirror_sine_calibration]\n")
            for key, value in calibration_params.items():
                f.write(f"{key}: {value}\n")


def main():
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()

    measurement_number = normalize_measurement_number(args.measurement)
    data_path = resolve_measurement_path(measurement_number, args.file, data_root)
    WL, RR_raw, sample_nr = open_h5_robust(data_path)
    sample_name = clean_sample_name(sample_nr)

    print("Sample number:", sample_nr)
    print("Resolved sample name:", sample_name)
    print("Structure data:", data_path)

    output_dir = output_root / f"{sample_name}_{measurement_number}"
    if output_dir.exists() and not args.keep_existing:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Always keep a raw overview to make the calibration step auditable.
    plot_overview(WL, RR_raw, "Structure raw data", output_dir / f"plot_{sample_name}_raw.png")
    plt.close("all")

    RR = RR_raw.copy()
    calibration_path = None
    calibration_params = None
    calibration_number = None
    if args.calibration_file:
        calibration_number = "file"
    elif args.calibration_measurement and args.calibration_measurement.strip().lower() not in {"none", "no", "false"}:
        calibration_number = normalize_measurement_number(args.calibration_measurement)

    if args.no_calibration:
        calibration_number = None

    if calibration_number is not None:
        if calibration_number != "file" and calibration_number == measurement_number:
            print("Calibration measurement equals the structure measurement; skipping mirror calibration.")
        else:
            calibration_path = resolve_measurement_path(calibration_number, args.calibration_file, data_root)
            print("Mirror calibration data:", calibration_path)
            mirror_WL, mirror_RR, mirror_sample = open_h5_robust(calibration_path)
            RR, calibration_curve, calibration_params = apply_mirror_calibration(
                WL,
                RR_raw,
                mirror_WL,
                mirror_RR,
                output_dir=output_dir,
                smooth_window=args.calibration_smooth_window,
                min_period_factor=args.min_calibration_period_factor,
                max_period_factor=args.max_calibration_period_factor,
                plot=True,
            )
            print("Mirror calibration sample:", mirror_sample)
            print("Applied mirror sine calibration.")
    else:
        print("Mirror calibration disabled.")

    write_processing_config(output_dir, args, data_path, calibration_path, calibration_params)

    # Legacy overview file name now contains the calibrated data if calibration was enabled.
    fig1, ax1, graph = plot_overview(WL, RR, "Structure", output_dir / f"plot_{sample_name}.png")
    if calibration_params is not None:
        fig1.savefig(output_dir / f"plot_{sample_name}_calibrated.png", dpi=1200)

    timestamp = assign_filename(str(output_dir), f"_ParamFile_{measurement_number}.txt")
    selectivefitter = SelectiveFitter(graph, 0.7, timestamp)

    print("Full save path:", output_dir / f"plot_{sample_name}.png")

    # %% Automated Peak Detection and Fitting
    print("\n--- Starting Automated Fitting ---")
    peak_indices = detect_peaks(RR, find_dips=True, prominence=args.prominence, distance=args.distance)
    print(f"Found {len(peak_indices)} dips. Proceeding to fit...")

    modified_sample_nr = f"{sample_name}_{measurement_number}"
    fit_save_path = str(output_root) + os.sep

    for counter, idx in enumerate(peak_indices):
        f_guess = WL[idx]
        f_range = [f_guess - args.window_nm, f_guess + args.window_nm]
        print(f"\nFitting peak #{counter} at ~{f_guess:.2f} nm...")
        Pixner_fit(
            f_guess=f_guess,
            WL=WL,
            RR=RR,
            sample_nr=modified_sample_nr,
            save_path=fit_save_path,
            counter=counter,
            f_range=f_range,
        )

    print("\n--- Automated Fitting Complete! ---")


if __name__ == "__main__":
    main()
