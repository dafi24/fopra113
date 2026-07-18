"""Estimate resonator parameters from FoPra dip-fit outputs.

This script implements the section 4.3 post-processing tasks from the FoPra 113
handout: FSR, group index, finesse, Q trends, first-order effective-index
estimates, and a cross-structure loss/coupling estimate.

Run after ``dips_fitting_automatic.py`` has produced ``*_ParamFile.txt`` files:

    python3 Plotter/resonator_parameters_estimation.py
"""

import argparse
import math
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from analysis_utils import (
    add_local_fsr_and_finesse,
    find_param_files,
    param_records_to_resonances,
    safe_name,
    write_all_resonance_summaries,
    write_rows_csv,
)


EXCLUDED_MEASUREMENTS = {
    "Sample 73 Diameter 100 um Coupling Distance 0.05 um_50413",
    "Sample 76 Diameter 100 um Coupling Distance 0.09 SECOND MEASUREMENT_50399",
    "Sample 78 Diameter 100 um Coupling Distance 0.1 um_50401",
}


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


def robust_sigma(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    if mad > 0:
        return 1.4826 * mad
    return np.nanstd(values)


def robust_z_mask(values, max_z):
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    if np.count_nonzero(mask) < 5:
        return mask
    median = np.nanmedian(values[mask])
    sigma = robust_sigma(values[mask])
    if not finite(sigma) or sigma <= 0:
        return mask
    return mask & (np.abs(values - median) <= max_z * sigma)


def mean_std_sem(values):
    arr = np.asarray([safe_float(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
    sem = float(std / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, std, sem, int(arr.size)


def extract_coupling_distance_um(structure):
    match = re.search(r"Coupling\s*Distance\s*([0-9]+(?:\.[0-9]+)?)", structure, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    return np.nan


def filter_resonances(resonances, args):
    filtered = []
    rejected = []
    for row in resonances:
        reason = None
        fwhm = safe_float(row.get("fwhm_nm"))
        sigma_fwhm = safe_float(row.get("sigma_fwhm_nm"))
        wl_sigma = safe_float(row.get("sigma_resonance_wavelength_nm"))
        q_value = safe_float(row.get("Q"))
        sigma_q = safe_float(row.get("sigma_Q"))
        r2 = safe_float(row.get("R2"))
        if row.get("is_double_resonance"):
            reason = "double_resonance"
        elif finite(r2) and r2 < args.min_r2:
            reason = "low_R2"
        elif not finite(fwhm) or fwhm <= 0:
            reason = "invalid_FWHM"
        elif finite(sigma_fwhm) and sigma_fwhm / fwhm > args.max_relative_fwhm_sigma:
            reason = "large_FWHM_uncertainty"
        elif finite(wl_sigma) and wl_sigma > args.max_center_sigma_nm:
            reason = "large_center_uncertainty"
        elif finite(q_value) and q_value <= 0:
            reason = "invalid_Q"
        elif finite(sigma_q) and finite(q_value) and q_value > 0 and sigma_q / q_value > args.max_relative_q_sigma:
            reason = "large_Q_uncertainty"

        if reason is None:
            filtered.append(dict(row))
        else:
            tmp = dict(row)
            tmp["rejection_reason"] = reason
            rejected.append(tmp)

    if len(filtered) >= 5:
        fwhm_values = np.log([row["fwhm_nm"] for row in filtered])
        q_values = np.log([row["Q"] for row in filtered if finite(row.get("Q")) and row.get("Q") > 0])
        fwhm_mask = robust_z_mask(fwhm_values, args.max_robust_z)
        if len(q_values) == len(filtered):
            q_mask = robust_z_mask(q_values, args.max_robust_z)
        else:
            q_mask = np.ones(len(filtered), dtype=bool)
        keep = fwhm_mask & q_mask
        kept = []
        for row, ok in zip(filtered, keep):
            if ok:
                kept.append(row)
            else:
                tmp = dict(row)
                tmp["rejection_reason"] = "robust_FWHM_or_Q_outlier"
                rejected.append(tmp)
        filtered = kept

    return filtered, rejected


def filter_after_fsr(resonances, args):
    if len(resonances) < 5:
        return resonances, []
    fsr_values = np.array([safe_float(row.get("local_fsr_nm")) for row in resonances], dtype=float)
    fsr_sigma_values = np.array([safe_float(row.get("sigma_local_fsr_nm")) for row in resonances], dtype=float)
    mask = np.isfinite(fsr_values) & (fsr_values > 0)
    rel_sigma = np.divide(fsr_sigma_values, fsr_values, out=np.full_like(fsr_values, np.nan), where=fsr_values > 0)
    mask &= (~np.isfinite(rel_sigma)) | (rel_sigma <= args.max_relative_fsr_sigma)
    mask &= robust_z_mask(fsr_values, args.max_robust_z)
    kept = []
    rejected = []
    for row, ok in zip(resonances, mask):
        if ok:
            kept.append(row)
        else:
            tmp = dict(row)
            tmp["rejection_reason"] = "FSR_outlier_or_uncertain"
            rejected.append(tmp)
    return kept, rejected


def fit_group_index(pairs, ring_length_um):
    if not pairs:
        return np.nan, np.nan
    lambda0 = np.array([safe_float(row["center_wavelength_nm"]) for row in pairs], dtype=float)
    fsr = np.array([safe_float(row["fsr_nm"]) for row in pairs], dtype=float)
    sigma = np.array([safe_float(row.get("sigma_fsr_nm")) for row in pairs], dtype=float)
    mask = np.isfinite(lambda0) & np.isfinite(fsr) & (fsr > 0)
    lambda0 = lambda0[mask]
    fsr = fsr[mask]
    sigma = sigma[mask]
    if lambda0.size == 0:
        return np.nan, np.nan
    l_nm = ring_length_um * 1000.0
    initial = float(np.nanmedian(lambda0 ** 2 / (fsr * l_nm)))

    def model(lam, ng):
        return lam ** 2 / (ng * l_nm)

    try:
        sigma_fit = sigma if np.all(np.isfinite(sigma)) and np.all(sigma > 0) else None
        popt, pcov = curve_fit(model, lambda0, fsr, p0=[initial], sigma=sigma_fit, absolute_sigma=sigma_fit is not None, bounds=(1.0, 10.0), maxfev=10000)
        ng = float(popt[0])
        sigma_ng = float(math.sqrt(pcov[0, 0])) if pcov.size else np.nan
        return ng, sigma_ng
    except Exception:
        ng_values = lambda0 ** 2 / (fsr * l_nm)
        return mean_std_sem(ng_values)[0:2]


def ra_from_finesse(finesse):
    finesse = float(finesse)
    if finesse <= 0 or not finite(finesse):
        return np.nan
    x = (-math.pi + math.sqrt(math.pi ** 2 + 4.0 * finesse ** 2)) / (2.0 * finesse)
    return x ** 2


def sigma_ra_from_sigma_f(finesse, sigma_finesse):
    if not (finite(finesse) and finite(sigma_finesse)):
        return np.nan
    delta = max(abs(finesse) * 1e-6, 1e-6)
    deriv = (ra_from_finesse(finesse + delta) - ra_from_finesse(finesse - delta)) / (2.0 * delta)
    return abs(deriv) * sigma_finesse


def estimate_neff_from_ng(ng, sigma_ng, wavelengths_nm, neff_reference, neff_reference_nm):
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    neff = ng + (neff_reference - ng) * wavelengths_nm / neff_reference_nm
    sigma = abs(1.0 - wavelengths_nm / neff_reference_nm) * sigma_ng if finite(sigma_ng) else np.full_like(wavelengths_nm, np.nan)
    return neff, sigma


def make_structure_plots(structure, resonances, pairs, ng_fit, sigma_ng_fit, output_dir, args):
    output_dir = Path(output_dir)
    plot_name = safe_name(structure)
    plot_structure = "\n".join(textwrap.wrap(structure, width=60))
    wavelength = np.array([row["resonance_wavelength_nm"] for row in resonances], dtype=float)

    finesse = np.array([row.get("finesse", np.nan) for row in resonances], dtype=float)
    sigma_finesse = np.array([row.get("sigma_finesse", np.nan) for row in resonances], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(wavelength, finesse, yerr=sigma_finesse, fmt=".", capsize=2)
    ax.set_xlabel("Resonance wavelength [nm]")
    ax.set_ylabel("Finesse [-]")
    ax.set_title(f"Finesse vs resonance wavelength\n{plot_structure}")
    fig.tight_layout()
    fig.savefig(output_dir / f"finesse_vs_wavelength_{plot_name}.png", dpi=args.plot_dpi)
    plt.close(fig)

    q = np.array([row.get("Q", np.nan) for row in resonances], dtype=float)
    sigma_q = np.array([row.get("sigma_Q", np.nan) for row in resonances], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(wavelength, q, yerr=sigma_q, fmt=".", capsize=2)
    ax.set_xlabel("Resonance wavelength [nm]")
    ax.set_ylabel("Loaded quality factor Q [-]")
    ax.set_title(f"Q vs resonance wavelength\n{plot_structure}")
    fig.tight_layout()
    fig.savefig(output_dir / f"Q_vs_wavelength_{plot_name}.png", dpi=args.plot_dpi)
    plt.close(fig)

    finite_finesse = finesse[np.isfinite(finesse) & (finesse > 0)]
    if finite_finesse.size:
        fig, ax = plt.subplots(figsize=(7, 4))
        counts, bin_edges, _ = ax.hist(
            finite_finesse,
            bins=20,
            label="Data",
        )

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = float(np.mean(np.diff(bin_edges)))

        log_finesse = np.log(finite_finesse)

        # Maximum-likelihood estimates provide stable initial parameters.
        m_initial = float(np.mean(log_finesse))
        s_initial = float(np.std(log_finesse, ddof=0))

        if not np.isfinite(s_initial) or s_initial <= 0:
            s_initial = 0.1

        # The normalization converts the log-normal probability density
        # into expected histogram counts.
        histogram_normalization = finite_finesse.size * bin_width

        def log_normal_counts(finesse_value, m, s):
            return (
                histogram_normalization
                / (finesse_value * s * np.sqrt(2.0 * np.pi))
                * np.exp(
                    -(np.log(finesse_value) - m) ** 2
                    / (2.0 * s ** 2)
                )
            )

        # Restrict the fit to the range supported by the measured data.
        m_lower = float(np.min(log_finesse))
        m_upper = float(np.max(log_finesse))

        # Avoid identical bounds for structures containing equal values.
        if m_upper <= m_lower:
            m_lower = m_initial - 0.5
            m_upper = m_initial + 0.5

        log_range = m_upper - m_lower
        s_lower = 1e-4
        s_upper = max(2.0 * log_range, 2.0 * s_initial, 0.1)

        fit_mask = (
            np.isfinite(bin_centers)
            & np.isfinite(counts)
            & (bin_centers > 0)
        )
        xdata = bin_centers[fit_mask]
        ydata = counts[fit_mask]

        # Poisson-like uncertainties for histogram counts.
        count_sigma = np.sqrt(ydata + 1.0)

        fit_succeeded = False

        try:
            popt, pcov = curve_fit(
                log_normal_counts,
                xdata,
                ydata,
                p0=[
                    np.clip(m_initial, m_lower, m_upper),
                    np.clip(s_initial, s_lower, s_upper),
                ],
                sigma=count_sigma,
                absolute_sigma=True,
                bounds=(
                    [m_lower, s_lower],
                    [m_upper, s_upper],
                ),
                max_nfev=50000,
            )

            m_fit, s_fit = popt
            sigma_m, sigma_s = np.sqrt(np.diag(pcov))

            fit_succeeded = (
                np.isfinite(m_fit)
                and np.isfinite(s_fit)
                and np.isfinite(sigma_m)
                and np.isfinite(sigma_s)
                and m_lower <= m_fit <= m_upper
                and s_lower < s_fit < s_upper
            )

        except (
            RuntimeError,
            ValueError,
            TypeError,
            np.linalg.LinAlgError,
        ):
            fit_succeeded = False

        if not fit_succeeded:
            # Stable maximum-likelihood fallback using the raw finesse data.
            m_fit = m_initial
            s_fit = s_initial

            # Approximate standard errors of the log-normal MLE parameters.
            sigma_m = s_fit / np.sqrt(finite_finesse.size)
            sigma_s = (
                s_fit
                / np.sqrt(2.0 * finite_finesse.size)
            )

            print(
                f"Using direct log-normal estimates for finesse "
                f"histogram of {structure}."
            )

        xfit = np.linspace(
            max(bin_edges[0], np.finfo(float).tiny),
            bin_edges[-1],
            500,
        )

        ax.plot(
            xfit,
            log_normal_counts(xfit, m_fit, s_fit),
            label=(
                fr"Log-normal fit: "
                fr"$m={m_fit:.3g}\pm{sigma_m:.2g}$, "
                fr"$s={s_fit:.3g}\pm{sigma_s:.2g}$"
            ),
        )

        ax.set_xlabel("Finesse [-]")
        ax.set_ylabel("Count")
        ax.set_title(f"Finesse histogram\n{plot_structure}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            output_dir / f"finesse_histogram_{plot_name}.png",
            dpi=args.plot_dpi,
        )
        plt.close(fig)

    if pairs:
        lambda0 = np.array([row["center_wavelength_nm"] for row in pairs], dtype=float)
        fsr = np.array([row["fsr_nm"] for row in pairs], dtype=float)
        sigma_fsr = np.array([row["sigma_fsr_nm"] for row in pairs], dtype=float)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(lambda0, fsr, yerr=sigma_fsr, fmt=".", capsize=2)
        ax.set_xlabel("Pair center wavelength lambda0 [nm]")
        ax.set_ylabel("FSR [nm]")
        ax.set_title(f"FSR vs wavelength\n{plot_structure}")
        fig.tight_layout()
        fig.savefig(output_dir / f"FSR_vs_wavelength_{plot_name}.png", dpi=args.plot_dpi)
        plt.close(fig)

        if finite(ng_fit):
            xfit = np.linspace(np.nanmin(lambda0), np.nanmax(lambda0), 300)
            yfit = xfit ** 2 / (ng_fit * args.ring_length_um * 1000.0)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.errorbar(
                lambda0,
                fsr,
                yerr=sigma_fsr,
                fmt=".",
                capsize=2,
                label="Neighboring-dip FSR",
            )
            ax.plot(
                xfit,
                yfit,
                label=fr"Eq. 5 fit: $n_g={ng_fit:.4g}\pm{sigma_ng_fit:.2g}$",
            )
            ax.set_xlabel("Pair center wavelength lambda0 [nm]")
            ax.set_ylabel("FSR [nm]")
            ax.set_title(f"Group-index fit\n{plot_structure}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / f"ng_fit_{plot_name}.png", dpi=args.plot_dpi)
            plt.close(fig)


def process_structure(param_file, args, output_dir):
    raw = param_records_to_resonances(param_file, exclude_double=True, min_r2=None)
    filtered, rejected = filter_resonances(raw, args)
    with_fsr, pairs = add_local_fsr_and_finesse(filtered, ring_length_um=args.ring_length_um)
    with_fsr, fsr_rejected = filter_after_fsr(with_fsr, args)
    with_fsr, pairs = add_local_fsr_and_finesse(with_fsr, ring_length_um=args.ring_length_um)
    pairs = [pair for pair in pairs if 1.7 <= safe_float(pair.get("fsr_nm")) <= 2.2]
    rejected.extend(fsr_rejected)

    if not with_fsr:
        return None, [], [], rejected

    structure = with_fsr[0]["structure"]
    measurement = with_fsr[0]["measurement"]
    ng_fit, sigma_ng_fit = fit_group_index(pairs, args.ring_length_um)
    wavelengths = np.array([row["resonance_wavelength_nm"] for row in with_fsr], dtype=float)
    neff_values, sigma_neff_values = estimate_neff_from_ng(ng_fit, sigma_ng_fit, wavelengths, args.neff_reference, args.neff_reference_nm)
    for row, neff, sigma_neff in zip(with_fsr, neff_values, sigma_neff_values):
        row["ng_fit"] = ng_fit
        row["sigma_ng_fit"] = sigma_ng_fit
        row["neff_estimated"] = neff
        row["sigma_neff_estimated"] = sigma_neff

    for pair in pairs:
        pair["ng_fit"] = ng_fit
        pair["sigma_ng_fit"] = sigma_ng_fit

    make_structure_plots(structure,with_fsr,pairs,ng_fit,sigma_ng_fit,output_dir,args)

    q_mean, q_std, q_sem, q_n = mean_std_sem([row.get("Q") for row in with_fsr])
    finesse_mean, finesse_std, finesse_sem, finesse_n = mean_std_sem([row.get("finesse") for row in with_fsr])
    fsr_mean, fsr_std, fsr_sem, fsr_n = mean_std_sem([row.get("fsr_nm") for row in pairs])
    ng_pair_mean, ng_pair_std, ng_pair_sem, ng_pair_n = mean_std_sem([row.get("ng_pair") for row in pairs])
    fwhm_mean, fwhm_std, fwhm_sem, fwhm_n = mean_std_sem([row.get("fwhm_nm") for row in with_fsr])
    neff_mean, neff_std, neff_sem, neff_n = mean_std_sem([row.get("neff_estimated") for row in with_fsr])

    summary = {
        "structure": structure,
        "measurement": measurement,
        "param_file": str(param_file),
        "coupling_distance_um": extract_coupling_distance_um(structure),
        "n_clean_resonances": len(with_fsr),
        "n_fsr_pairs": len(pairs),
        "mean_Q": q_mean,
        "std_Q": q_std,
        "sem_Q": q_sem,
        "mean_finesse": finesse_mean,
        "std_finesse": finesse_std,
        "sem_finesse": finesse_sem,
        "mean_FSR_nm": fsr_mean,
        "std_FSR_nm": fsr_std,
        "sem_FSR_nm": fsr_sem,
        "mean_FWHM_nm": fwhm_mean,
        "std_FWHM_nm": fwhm_std,
        "sem_FWHM_nm": fwhm_sem,
        "ng_fit": ng_fit,
        "sigma_ng_fit": sigma_ng_fit,
        "mean_ng_pair": ng_pair_mean,
        "std_ng_pair": ng_pair_std,
        "sem_ng_pair": ng_pair_sem,
        "mean_neff_estimated": neff_mean,
        "std_neff_estimated": neff_std,
        "sem_neff_estimated": neff_sem,
    }
    return summary, with_fsr, pairs, rejected


def estimate_loss_and_coupling(summary_rows, args):
    valid = [row for row in summary_rows if finite(row.get("mean_finesse")) and finite(row.get("coupling_distance_um"))]
    if not valid:
        return [], {}
    for row in valid:
        row["ra_product"] = ra_from_finesse(row["mean_finesse"])
        row["sigma_ra_product"] = sigma_ra_from_sigma_f(row["mean_finesse"], row.get("sem_finesse", row.get("std_finesse")))

    weak = max(valid, key=lambda row: row["coupling_distance_um"])
    a = weak["ra_product"]
    sigma_a = weak.get("sigma_ra_product", np.nan)
    length_cm = args.ring_length_um * 1e-4
    alpha_db_cm = -20.0 * math.log10(a) / length_cm if finite(a) and a > 0 else np.nan
    sigma_alpha = abs(-20.0 / (math.log(10.0) * length_cm) * sigma_a / a) if finite(a) and finite(sigma_a) and a > 0 else np.nan

    rows = []
    for row in valid:
        p = row["ra_product"]
        sigma_p = row.get("sigma_ra_product", np.nan)
        r = p / a if finite(p) and finite(a) and a > 0 else np.nan
        clipped = False
        if finite(r) and r > 1.0:
            r = 1.0
            clipped = True
        if finite(r) and r >= 0:
            k = math.sqrt(max(0.0, 1.0 - r ** 2))
        else:
            k = np.nan
        sigma_r = np.nan
        sigma_k = np.nan
        if finite(p) and finite(sigma_p) and finite(a) and finite(sigma_a) and a > 0:
            sigma_r = abs(r) * math.sqrt((sigma_p / p) ** 2 + (sigma_a / a) ** 2) if p > 0 else np.nan
            sigma_k = abs(r / k) * sigma_r if finite(k) and k > 0 and finite(sigma_r) else np.nan
        out = dict(row)
        out.update(
            {
                "weakest_coupled_reference_structure": weak["structure"],
                "a_from_weakest_ring": a,
                "sigma_a_from_weakest_ring": sigma_a,
                "alpha_db_per_cm": alpha_db_cm,
                "sigma_alpha_db_per_cm": sigma_alpha,
                "self_coupling_r": r,
                "sigma_self_coupling_r": sigma_r,
                "cross_coupling_k": k,
                "sigma_cross_coupling_k": sigma_k,
                "r_clipped_to_one": clipped,
            }
        )
        rows.append(out)
    return rows, {"alpha_db_per_cm": alpha_db_cm, "sigma_alpha_db_per_cm": sigma_alpha, "reference_structure": weak["structure"]}


def make_comparison_plots(coupling_rows, output_dir, args):
    if not coupling_rows:
        return
    output_dir = Path(output_dir)
    dist = np.array([row["coupling_distance_um"] for row in coupling_rows], dtype=float)
    finesse = np.array([row["mean_finesse"] for row in coupling_rows], dtype=float)
    finesse_err = np.array([row.get("std_finesse", np.nan) for row in coupling_rows], dtype=float)
    order = np.argsort(dist)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(dist[order], finesse[order], yerr=finesse_err[order], fmt=".", capsize=2)
    ax.set_xlabel("Coupling distance [um]")
    ax.set_ylabel("Mean finesse [-]")
    ax.set_title("Finesse vs coupling distance")
    fig.tight_layout()
    fig.savefig(output_dir / "finesse_vs_coupling_distance.png", dpi=args.plot_dpi)
    plt.close(fig)

    k = np.array([row.get("cross_coupling_k", np.nan) for row in coupling_rows], dtype=float)
    kerr = np.array([row.get("sigma_cross_coupling_k", np.nan) for row in coupling_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(dist[order], k[order], yerr=kerr[order], fmt=".", capsize=2)
    ax.set_xlabel("Coupling distance [um]")
    ax.set_ylabel("Estimated cross-coupling k [-]")
    ax.set_title("Estimated ring-bus coupling vs coupling distance")
    fig.tight_layout()
    fig.savefig(output_dir / "k_vs_coupling_distance.png", dpi=args.plot_dpi)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate FoPra ring-resonator parameters from fit outputs.")
    parser.add_argument("--plots-dir", default=str(Path(__file__).resolve().parent / "Plots"), help="Folder containing per-measurement output folders.")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "ResonatorParameters"), help="Folder for derived tables and plots.")
    parser.add_argument("--ring-length-um", type=float, default=100.0, help="Ring resonator length L used in Eq. 5.")
    parser.add_argument("--min-r2", type=float, default=0.80, help="Minimum fit R2 retained for post-processing.")
    parser.add_argument("--max-center-sigma-nm", type=float, default=0.02, help="Reject fits with center uncertainty above this value.")
    parser.add_argument("--max-relative-fwhm-sigma", type=float, default=0.50, help="Reject fits with sigma(FWHM)/FWHM above this value.")
    parser.add_argument("--max-relative-q-sigma", type=float, default=1.0, help="Reject fits with sigma(Q)/Q above this value.")
    parser.add_argument("--max-relative-fsr-sigma", type=float, default=0.20, help="Reject local FSR values with relative uncertainty above this value.")
    parser.add_argument("--max-robust-z", type=float, default=4.5, help="Robust z-score threshold for FWHM/Q/FSR outlier rejection.")
    parser.add_argument("--neff-reference", type=float, default=2.6, help="First estimate of neff at the reference wavelength from Fig. 2a hint.")
    parser.add_argument("--neff-reference-nm", type=float, default=1550.0, help="Reference wavelength for the neff estimate.")
    parser.add_argument("--plot-dpi", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    plots_dir = Path(args.plots_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_structure_dir = output_dir / "per_structure"
    per_structure_dir.mkdir(parents=True, exist_ok=True)

    write_all_resonance_summaries(plots_dir, ring_length_um=args.ring_length_um, exclude_double=True)

    summary_rows = []
    all_clean_resonances = []
    all_clean_pairs = []
    all_rejected = []
    for param_file in find_param_files(plots_dir):
        if any(measurement in str(param_file) for measurement in EXCLUDED_MEASUREMENTS):
            continue
        result = process_structure(param_file, args, per_structure_dir)
        summary, clean_resonances, clean_pairs, rejected = result
        all_rejected.extend(rejected)
        if summary is None:
            continue
        summary_rows.append(summary)
        all_clean_resonances.extend(clean_resonances)
        all_clean_pairs.extend(clean_pairs)
        base = safe_name(f"{summary['structure']}_{summary['measurement']}")
        write_rows_csv(per_structure_dir / f"clean_resonances_{base}.csv", clean_resonances)
        write_rows_csv(per_structure_dir / f"clean_fsr_pairs_{base}.csv", clean_pairs)

    coupling_rows, coupling_meta = estimate_loss_and_coupling(summary_rows, args)
    make_comparison_plots(coupling_rows, output_dir, args)

    write_rows_csv(output_dir / "all_clean_resonances.csv", all_clean_resonances)
    write_rows_csv(output_dir / "all_clean_fsr_pairs.csv", all_clean_pairs)
    write_rows_csv(output_dir / "rejected_fit_outliers.csv", all_rejected)
    write_rows_csv(output_dir / "structure_summary.csv", summary_rows)
    write_rows_csv(output_dir / "coupling_loss_summary.csv", coupling_rows)

    with open(output_dir / "analysis_notes.txt", "w", encoding="utf-8") as f:
        f.write("FoPra resonator parameter estimation\n")
        f.write(f"plots_dir: {plots_dir}\n")
        f.write(f"ring_length_um: {args.ring_length_um}\n")
        f.write("Equations used: FSR=lambda0^2/(ng L), Q=lambda_res/FWHM, finesse=FSR/FWHM, F=pi*sqrt(ra)/(1-ra).\n")
        f.write("Double-resonance fits and high-uncertainty/outlier fits are excluded before structure-level calculations.\n")
        f.write(f"neff_reference: {args.neff_reference} at {args.neff_reference_nm} nm.\n")
        if coupling_meta:
            f.write(f"weakest-coupled reference structure: {coupling_meta['reference_structure']}\n")
            f.write(f"alpha_db_per_cm: {coupling_meta['alpha_db_per_cm']} +/- {coupling_meta['sigma_alpha_db_per_cm']}\n")

    print(f"Processed {len(summary_rows)} structure output folders.")
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
