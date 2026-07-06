# FoPra 113 silicon-chip ring-resonator analysis

This repository contains measurement data and Python scripts for the FoPra 113 experiment, **Trapping light on a silicon chip**. The analysis scripts load Artiq/HDF5 wavelength scans, optionally calibrate the measured spectrum using a calibration-mirror scan, detect suitable resonance dips, and fit the dips with single- or double-Lorentzian models.

The current analysis scripts live in `Plotter/`:

```text
Plotter/
├── dips_fitting_automatic.py   # batch runner for all wavelength scans
├── plotter.py                  # single-measurement calibrated plotting and dip fitting
├── calibration.py              # shared calibration and HDF5-loading helpers
├── functions_double.py         # legacy fitting and plotting functions
└── test_callback.py            # interactive/manual matplotlib fitting helper
```

The uploaded data is stored under date/hour folders, for example:

```text
2026-05-07/14/000050397-FoPraWavelengthScan.h5
2026-05-07/16/000050420-FoPraWavelengthScan.h5
```

The scripts no longer assume a hard-coded absolute path. By default they infer the repository root from the location of `Plotter/` and search recursively for `*-FoPraWavelengthScan.h5` files.

## Installation

Use the FoPra Python environment that already contains the scientific Python packages used by the scripts. The required packages are:

```text
numpy
matplotlib
h5py
scipy
scikit-learn
```

Typical usage from the repository root:

```bash
python3 Plotter/dips_fitting_automatic.py
```

Typical usage from inside `Plotter/`:

```bash
python3 dips_fitting_automatic.py
```

## Recommended workflow

For normal processing, run the batch script:

```bash
python3 Plotter/dips_fitting_automatic.py
```

This will:

1. Find all readable `*-FoPraWavelengthScan.h5` files.
2. Use the mirror scan `50420` as the default calibration reference.
3. Skip the mirror scan itself.
4. Call `plotter.py` once for each structure scan.
5. Save calibrated spectra, mirror calibration fits, individual dip fits, fit parameter files, and peak-selection logs in `Plotter/Plots/`.

For a dry run that only prints the commands:

```bash
python3 Plotter/dips_fitting_automatic.py --dry-run
```

For a lighter, lower-memory batch run:

```bash
python3 Plotter/dips_fitting_automatic.py --overview-dpi 150 --fit-plot-dpi 100
```

For selected measurements only:

```bash
python3 Plotter/dips_fitting_automatic.py --measurements 50397 50405 50417
```

## Output structure

Each processed measurement gets an output directory like:

```text
Plotter/Plots/<structure_name>_<measurement_number>/
```

Typical files in each output folder:

```text
plot_<structure>_raw.png              # uncalibrated raw spectrum
plot_<structure>.png                  # calibrated spectrum used for fitting
plot_<structure>_calibrated.png       # same calibrated overview, explicit name
calibrated_spectrum.png               # raw/calibration/calibrated comparison
calibrated_trace.txt                  # wavelength, raw data, calibration curve, calibrated data
mirror_calibration_fit.png            # mirror data and long-period sine fit
mirror_calibration_fit.txt            # mirror data, smoothed mirror data, sine fit
mirror_calibration_parameters.txt     # fitted sine calibration parameters
processing_config.txt                 # full command/settings record
peak_selection_log.tsv                # accepted/rejected dip candidates and reasons
failed_peak_fits.txt                  # only present if individual dip fits failed
<structure>_<measurement>_ParamFile.txt
_Fit_<start>to<end>nm_id<counter>.png # individual dip fit plots
```

`ParamFile.txt` contains one row per fitted resonance window. For single dips, the second-resonance columns are `nan`. For suspected split/double resonances, the double-Lorentzian parameters are written.

## `Plotter/dips_fitting_automatic.py`

### Purpose

Batch runner for all FoPra wavelength scans. It recursively discovers all `*-FoPraWavelengthScan.h5` files, validates that they can be read, chooses a mirror calibration scan, and launches `plotter.py` as a subprocess for every selected structure scan.

Use this script for normal full-repository processing.

### What it does

- Searches `--data-root` recursively for wavelength-scan HDF5 files.
- Uses measurement `50420` as the mirror calibration scan by default.
- Skips the calibration scan itself unless `--include-calibration-scan` is used.
- Validates scans using the robust HDF5 reader from `calibration.py`.
- Sets `MPLBACKEND=Agg` for safer non-interactive batch plotting.
- Runs each measurement separately, so one measurement can fail without stopping the whole batch by default.
- Passes all calibration, dip-selection, plot-resolution, and fitting options down to `plotter.py`.

### Basic commands

Run all readable structure scans:

```bash
python3 Plotter/dips_fitting_automatic.py
```

Run only selected scans:

```bash
python3 Plotter/dips_fitting_automatic.py --measurements 50397 50405 50417
```

Use a different mirror scan:

```bash
python3 Plotter/dips_fitting_automatic.py --calibration-measurement 50415
```

Use a direct mirror file path:

```bash
python3 Plotter/dips_fitting_automatic.py --calibration-file 2026-05-07/16/000050420-FoPraWavelengthScan.h5
```

Print commands without running:

```bash
python3 Plotter/dips_fitting_automatic.py --dry-run
```

Stop immediately if one measurement fails:

```bash
python3 Plotter/dips_fitting_automatic.py --stop-on-error
```

### Options

| Option | Default | Meaning |
|---|---:|---|
| `--data-root PATH` | repository root | Root folder searched recursively for `*-FoPraWavelengthScan.h5`. |
| `--output-dir PATH` | `Plotter/Plots` | Output folder passed to `plotter.py`. |
| `--calibration-measurement N` | `50420` | Measurement number of the mirror calibration scan. |
| `--calibration-file PATH` | none | Direct path to the mirror calibration `.h5`; overrides `--calibration-measurement`. |
| `--measurements N ...` | all readable scans | Process only the listed measurement numbers. |
| `--include-calibration-scan` | off | Also run fitting on the mirror calibration scan itself. Usually not needed. |
| `--dry-run` | off | Print the generated `plotter.py` commands without executing them. |
| `--stop-on-error` | off | Stop the batch when one measurement subprocess fails. By default the batch continues. |
| `--continue-on-error` | compatibility flag | Kept for old command compatibility; continuing is now the default. |
| `--stop-on-fit-error` | off | Forwarded to `plotter.py`; stop a measurement if one individual peak fit fails. |
| `--keep-existing` | off | Do not delete existing output folders before writing new output. |
| `--prominence FLOAT` | `0.08` | Initial dip-detection prominence after normalization. Higher means fewer candidate dips. |
| `--distance INT` | `200` | Minimum point spacing between initial dip candidates. |
| `--window-nm FLOAT` | `0.5` | Half-width of the wavelength window used for each accepted dip fit. |
| `--detection-smooth-window INT` | `11` | Moving-average window in points used only for dip detection. Use `1` to disable. |
| `--min-dip-depth-fraction FLOAT` | `0.08` | Reject candidate dips whose local depth is smaller than this fraction of the local baseline. |
| `--min-dip-depth-sigma FLOAT` | `5.0` | Reject candidate dips whose depth is less than this many robust local-noise sigmas. |
| `--duplicate-merge-nm FLOAT` | `0.35` | Merge detections whose actual local minima are closer than this wavelength spacing. |
| `--edge-reject-fraction FLOAT` | `0.15` | Reject candidates whose local minimum lies too close to the fit-window edge. |
| `--calibration-smooth-window INT` | `401` | Moving-average window used before fitting the mirror calibration sine curve. |
| `--min-calibration-period-factor FLOAT` | `0.75` | Lower bound for mirror sine period as a fraction of the mirror wavelength span. |
| `--max-calibration-period-factor FLOAT` | `4.0` | Upper bound for mirror sine period as a fraction of the mirror wavelength span. |
| `--overview-dpi INT` | `300` | DPI for overview plots. Lower this for faster/lower-memory runs. |
| `--fit-plot-dpi INT` | `200` | Maximum DPI for individual fit plots saved by legacy fitting code. Use `<=0` to disable capping. |

### Tuning notes

If noise or shallow baseline fluctuations are still being fit, increase:

```bash
--prominence 0.10 --min-dip-depth-fraction 0.10 --min-dip-depth-sigma 7
```

If real resonances are missed, decrease:

```bash
--prominence 0.05 --min-dip-depth-fraction 0.05
```

If the same resonance is still fitted more than once, increase:

```bash
--duplicate-merge-nm 0.5
```

If two genuinely separate neighboring resonances are being merged too often, decrease:

```bash
--duplicate-merge-nm 0.15
```

## `Plotter/plotter.py`

### Purpose

Single-measurement calibrated plotting and dip-fitting script. This is the core processing script called by `dips_fitting_automatic.py`.

### What it does

For one structure scan, `plotter.py`:

1. Loads the wavelength scan using the robust HDF5 loader.
2. Saves a raw overview plot.
3. Loads the mirror calibration scan unless calibration is disabled.
4. Fits the mirror trace with a constrained long-period sine curve.
5. Divides the raw structure trace by the fitted mirror baseline.
6. Saves calibration plots and text files.
7. Detects dip candidates on a smoothed copy of the calibrated trace.
8. Filters out shallow/noisy/edge candidates.
9. Merges duplicate detections of the same physical dip.
10. Fits each accepted dip using the legacy `Pixner_fit` function.
11. Logs failed individual peak fits and continues by default.

### Basic commands

Fit one measurement using default mirror calibration `50420`:

```bash
python3 Plotter/plotter.py --measurement 50405
```

Fit one direct file with one direct mirror file:

```bash
python3 Plotter/plotter.py \
  --file 2026-05-07/15/000050405-FoPraWavelengthScan.h5 \
  --calibration-file 2026-05-07/16/000050420-FoPraWavelengthScan.h5
```

Fit without mirror calibration:

```bash
python3 Plotter/plotter.py --measurement 50405 --no-calibration
```

Run with stricter peak selection:

```bash
python3 Plotter/plotter.py --measurement 50405 --prominence 0.10 --min-dip-depth-fraction 0.10
```

Run with lighter plots:

```bash
python3 Plotter/plotter.py --measurement 50405 --overview-dpi 150 --fit-plot-dpi 100
```

### Options

| Option | Default | Meaning |
|---|---:|---|
| `--measurement N` | `50405` | FoPra wavelength-scan measurement number for the structure data. |
| `--file PATH` | none | Direct path to the structure `.h5`; if omitted, `--data-root` is searched. |
| `--calibration-measurement N` | `50420` | Measurement number for the mirror calibration scan. |
| `--calibration-file PATH` | none | Direct path to the mirror calibration `.h5`; overrides `--calibration-measurement`. |
| `--no-calibration` | off | Fit the raw structure data directly without mirror calibration. |
| `--data-root PATH` | repository root | Root folder searched recursively for measurement files. |
| `--output-dir PATH` | `Plotter/Plots` | Folder where plots, text outputs, and fit parameters are saved. |
| `--prominence FLOAT` | `0.08` | Initial `find_peaks` prominence after normalization. |
| `--distance INT` | `200` | Minimum point spacing between initial dip candidates. |
| `--window-nm FLOAT` | `0.5` | Half-width of each fitting window in nm. |
| `--detection-smooth-window INT` | `11` | Moving-average smoothing in points before detection only. |
| `--min-dip-depth-fraction FLOAT` | `0.08` | Minimum accepted local dip depth divided by local baseline. |
| `--min-dip-depth-sigma FLOAT` | `5.0` | Minimum accepted local dip depth in robust noise sigmas. |
| `--duplicate-merge-nm FLOAT` | `0.35` | Merge candidates closer than this in actual local-minimum wavelength. |
| `--edge-reject-fraction FLOAT` | `0.15` | Reject dips whose minimum is too close to the fit-window edge. |
| `--calibration-smooth-window INT` | `401` | Smoothing window before the mirror sine fit. |
| `--min-calibration-period-factor FLOAT` | `0.75` | Minimum allowed mirror sine period relative to mirror wavelength span. |
| `--max-calibration-period-factor FLOAT` | `4.0` | Maximum allowed mirror sine period relative to mirror wavelength span. |
| `--overview-dpi INT` | `300` | DPI for overview spectrum plots. |
| `--fit-plot-dpi INT` | `200` | Maximum DPI for individual dip plots. Use `<=0` to preserve legacy DPI. |
| `--no-interactive` | off | Disable the interactive matplotlib `SelectiveFitter` callback. Batch mode enables this. |
| `--stop-on-fit-error` | off | Stop when one individual peak fit fails. By default the failed peak is logged and skipped. |
| `--keep-existing` | off | Do not delete an existing output folder before processing. |

## `Plotter/calibration.py`

### Purpose

Shared helper library for robust HDF5 loading, file discovery, sample-name cleanup, and mirror calibration.

This is not normally run directly from the command line. It is imported by `plotter.py` and `dips_fitting_automatic.py`.

### Important functions

#### `open_h5_robust(path)`

Reads a FoPra wavelength scan even when the available HDF5 datasets differ between files. It tries:

1. Precomputed relative reflection datasets such as `wavelength_scan.RR.mean`.
2. Derived `abs(R.mean) / abs(E.mean)` if reflected and excitation channels are available.
3. Fallback signal datasets such as `wavelength_scan.S.mean`.

It also reads the wavelength dataset and a structure/sample string when available.

#### `find_measurement_file(measurement_number, data_root='.')`

Builds a FoPra filename such as `000050420-FoPraWavelengthScan.h5` and searches for it recursively below `data_root`.

#### `fit_mirror_sine(WL, RR, ...)`

Fits the mirror reflection trace with a long-period sine model plus a linear slope. The data is smoothed first so the fit follows the broad trend rather than short variations.

#### `apply_mirror_calibration(WL, RR, mirror_WL, mirror_RR, ...)`

Fits the mirror sine curve and divides the structure trace by that fitted calibration baseline. It also saves calibration diagnostic plots and text files if an output directory is supplied.

### Programmatic example

```python
from Plotter.calibration import open_h5_robust, apply_mirror_calibration

WL, RR, sample = open_h5_robust('2026-05-07/15/000050405-FoPraWavelengthScan.h5')
mirror_WL, mirror_RR, _ = open_h5_robust('2026-05-07/16/000050420-FoPraWavelengthScan.h5')
RR_calibrated, baseline, params = apply_mirror_calibration(WL, RR, mirror_WL, mirror_RR)
```

## `Plotter/functions_double.py`

### Purpose

Legacy analysis and fitting library. It contains the original HDF5 reader, peak detection helper, Lorentzian models, double-Lorentzian models, Fourier filtering utilities, and the main fitting functions used by `plotter.py`.

This file is not normally run directly. It is imported by `plotter.py`.

### Important functions

#### `detect_peaks(data, find_dips=True, **kwargs)`

Normalizes a trace, optionally inverts it so dips become peaks, and calls `scipy.signal.find_peaks`. `plotter.py` calls this first, then applies additional filtering to reject shallow/noisy/duplicate candidates.

#### `Lorentzian_sq(...)` and `double_Lorentzian_sq(...)`

Model functions for single and double squared-Lorentzian dip fits with a linear background.

#### `fitting_Lor(...)` and `fitting_doubleLor(...)`

Call `scipy.optimize.curve_fit` and calculate loaded quality factors using:

```text
Q_L = resonance wavelength / fitted width
```

#### `param(...)` and `param_double(...)`

Use differential evolution to generate initial guesses for the single- and double-Lorentzian fits.

#### `Pixner_fit(f_guess, WL, RR, sample_nr, save_path, counter, f_range)`

The main automatic dip fitter used by `plotter.py`. It crops the data to `f_range`, tries single and double Lorentzian fits, decides whether a double fit is justified, saves an individual fit plot, and appends the fit parameters to the measurement parameter file.

#### `fourier_filtering(...)` and `plot_Fourier_ops(...)`

Utilities for Fourier-domain filtering and diagnostic plotting. They are present for analysis experiments but are not part of the current default batch fitting pipeline.

### Outputs produced by `Pixner_fit`

`Pixner_fit` saves:

```text
<structure>_<measurement>_ParamFile.txt
_Fit_<start>to<end>nm_id<counter>.png
```

The parameter file stores resonance center, center uncertainty, fitted width, width uncertainty, loaded Q, Q uncertainty, amplitude, background parameters, and R². For split/double peaks it also stores the second resonance parameters.

## `Plotter/test_callback.py`

### Purpose

Interactive/manual fitting helper for matplotlib figures. It defines the `SelectiveFitter` class, which lets a user zoom/select a region in an open matplotlib plot and trigger fitting on that currently visible region.

This file is not normally run directly. `plotter.py` can create a `SelectiveFitter` instance unless `--no-interactive` is used. The batch script always forwards `--no-interactive`.

### Important components

#### `assign_filename(direc, stamp)`

Creates a timestamped filename path.

#### `SelectiveFitter(graph, r2_thresh, filename)`

Stores the plotted data from a matplotlib line object and connects a key-press callback. When triggered, it:

1. Reads the currently visible x/y limits of the plot.
2. Extracts only the visible data points.
3. Fits both single and double Lorentzian models.
4. Chooses a split/double fit only when the double fit is meaningfully better.
5. Overlays the fit on the current plot and prints Q/R² information.

### Options

`test_callback.py` has no command-line options. Use `plotter.py` without `--no-interactive` if you want the callback behavior available while viewing one measurement interactively.

## Legacy or non-default helpers

Some functions in `functions_double.py`, such as `write_analysis_h5`, `UnpackAnalysisH5`, `normalization`, `correct_finder`, `remove_peaks`, and `findFitRange`, are retained from earlier workflows. They are useful for manual or follow-up analysis, but the default automated pipeline is:

```text
dips_fitting_automatic.py
  -> plotter.py
      -> calibration.py for loading and mirror calibration
      -> functions_double.py for dip fitting
      -> test_callback.py only for optional interactive mode
```

## Troubleshooting

### The script fits too many shallow/noisy features

Use stricter detection settings:

```bash
python3 Plotter/dips_fitting_automatic.py \
  --prominence 0.10 \
  --min-dip-depth-fraction 0.10 \
  --min-dip-depth-sigma 7
```

### The script misses real dips

Use looser detection settings:

```bash
python3 Plotter/dips_fitting_automatic.py \
  --prominence 0.05 \
  --min-dip-depth-fraction 0.05 \
  --min-dip-depth-sigma 3
```

### The same resonance is fitted more than once

Increase the duplicate merge window:

```bash
python3 Plotter/dips_fitting_automatic.py --duplicate-merge-nm 0.5
```

### Neighboring resonances are incorrectly merged

Decrease the duplicate merge window:

```bash
python3 Plotter/dips_fitting_automatic.py --duplicate-merge-nm 0.15
```

### A long batch run is killed or runs out of memory

Use lower plot DPI values:

```bash
python3 Plotter/dips_fitting_automatic.py --overview-dpi 150 --fit-plot-dpi 100
```

The batch runner already sets `MPLBACKEND=Agg` and `plotter.py` closes figures after each peak fit.

### One measurement fails but the batch should continue

This is the default behavior in `dips_fitting_automatic.py`. Do not use `--stop-on-error`.

### One individual peak fit fails but the measurement should continue

This is the default behavior in `plotter.py`. Do not use `--stop-on-fit-error`. Failed individual peak fits are written to `failed_peak_fits.txt` in that measurement output folder.

## Notes on final FoPra analysis

The scripts extract fitted resonance centers, widths, quality factors, and split/double-resonance information. The final FoPra report quantities such as FSR, group index, finesse histograms, loss coefficient, and waveguide-ring coupling comparisons may require additional post-processing of the generated `ParamFile.txt` files.
