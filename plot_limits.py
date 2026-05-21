#!/usr/bin/env python3
# file: limits_plot.py

"""
Standalone script to extract HybridNew limits from ROOT files and plot expected limits vs theory,
and estimate mass limit(s) from intersections of theory cross-section with expected median limit.

Usage:
  python3 limits_plot.py --preset all_systs --misc all_systs --strategy Ph
  python3 limits_plot.py --root-dir /path/to/dir --out-dir ./out --modes SpinZero_PhotonFusion
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import mplhep
import numpy as np
import uproot


production_modes: Dict[str, Dict[int, float]] = {
    "SpinZero_PhotonFusion": {
        1000: 1.09e4,
        1500: 8.62e2,
        2000: 9.13e1,
        2500: 1.05e1,
        3000: 1.15e0,
        3500: 1.07e-1,
        4000: 7.62e-3,
        4500: 3.50e-4,
    },
    "SpinZero_DrellYan": {
        1000: 1.45e1,
        1500: 8.12e-1,
        2000: 6.04e-2,
        2500: 4.99e-3,
        3000: 4.42e-4,
        3500: 4.17e-5,
        4000: 3.69e-6,
        4500: 2.37e-7,
    },
    "SpinHalf_PhotonFusion": {
        1000: 5.10e4,
        1500: 3.65e3,
        2000: 3.53e2,
        2500: 3.71e1,
        3000: 3.73e0,
        3500: 3.22e-1,
        4000: 2.12e-2,
        4500: 9.05e-3,
    },
    "SpinHalf_DrellYan": {
        1000: 2.40e2,
        1500: 1.63e1,
        2000: 1.47e0,
        2500: 1.43e-1,
        3000: 1.44e-2,
        3500: 1.54e-3,
        4000: 1.65e-4,
        4500: 1.38e-5,
    },
}

quantiles: Dict[str, int] = {
    "0.025": 0,
    "0.160": 1,
    "0.500": 2,
    "0.840": 3,
    "0.975": 4,
}

ROOT_DIR_PRESETS: Dict[str, str] = {
    "20Feb": "/eos/user/t/tmenezes/Monopole_Ntuples/Files_Limits/20Feb_monopole-combine-output-v2/",
    "wout_systs": "/eos/user/t/tmenezes/Monopole_Ntuples/Files_Limits/monopole-combine-output-freezenui-wout-systematics",
    "v3": "/eos/user/t/tmenezes/Monopole_Ntuples/Files_Limits/monopole-combine-output-v3/",
    "Photon_all_systs": "/eos/user/t/tmenezes/Monopole_Ntuples/Files_Limits/Photon-monopole-combine-output-all-systs/",
    "PFMET_all_systs": "/eos/user/t/tmenezes/Monopole_Ntuples/Files_Limits/PFMET-monopole-combine-output-all-systs/",
    "Combined_all_systs": "/eos/user/t/tmenezes/Monopole_Ntuples/Files_Limits/Combined-monopole-combine-output-all-systs/",
    "Updated_Combined_all_systs": "/eos/user/t/tmenezes/Monopole_Ntuples/Files_Limits/Updated_Combined-monopole-combine-output/",
    "rateParam_PFMET_all_systs": "/eos/user/t/tmenezes/Monopole_Ntuples/Files_Limits/rateParam_PFMET-monopole-combine-output/",
    "rateParam_Combined_all_systs": "/eos/user/t/tmenezes/Monopole_Ntuples/Files_Limits/rateParam_Combined-monopole-combine-output/",
}


DEFAULT_OUT_DIR = "/eos/user/t/tmenezes/www/Monopoles/Limits/rateParam"

# Brazil band colors (match reference)
BRAZIL_YELLOW_2SIGMA = "#FFF200"
BRAZIL_GREEN_1SIGMA = "#00A651"


def extract_limits(production_mode: str, root_dir: str) -> Dict[int, List[Optional[float]]]:
    """
    Returns:
      mass -> [q2.5, q16, q50, q84, q97.5] (None if missing)
    """
    limits: defaultdict[int, List[Optional[float]]] = defaultdict(lambda: [None] * 5)

    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"root_dir does not exist: {root_dir}")

    pattern = re.compile(r"higgsCombine\.(.+?)\.HybridNew\.mH(\d+)\.quant0\.(\d+)\.root$")

    for p in root_path.iterdir():
        if not p.name.endswith(".root"):
            continue
        if production_mode not in p.name:
            continue

        m = pattern.match(p.name)
        if not m:
            continue

        mass = int(m.group(2))
        quant = f"0.{m.group(3)}"
        idx = quantiles.get(quant)
        if idx is None:
            continue

        try:
            with uproot.open(str(p)) as f:
                tree = f["limit"]
                limit_val = tree["limit"].array(library="np")[0]
                limits[mass][idx] = float(limit_val)
        except Exception as e:
            print(f"Skipping {p.name} due to error: {e}")

    return dict(sorted(limits.items()))


def _ensure_out_dir(out_dir: str) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def estimate_mass_intersections(
    masses: Sequence[float],
    expected: Sequence[float],
    theory: Sequence[float],
) -> List[float]:
    """
    Estimate intersection mass(es) where theory(m) == expected(m).

    Uses log-ratio: f(m) = ln(theory/expected) and linear interpolation in m
    on each segment where f changes sign.

    Returns:
      Sorted list of intersection masses (can be empty or have multiple roots).
    """
    if not (len(masses) == len(expected) == len(theory)):
        raise ValueError("masses/expected/theory must have same length")

    xs: List[float] = []
    for i in range(len(masses) - 1):
        x1, x2 = float(masses[i]), float(masses[i + 1])
        e1, e2 = float(expected[i]), float(expected[i + 1])
        t1, t2 = float(theory[i]), float(theory[i + 1])

        if e1 <= 0 or e2 <= 0 or t1 <= 0 or t2 <= 0:
            continue

        f1 = math.log(t1 / e1)
        f2 = math.log(t2 / e2)

        if f1 == 0.0:
            xs.append(x1)
            continue

        if f1 * f2 < 0.0:
            # linear interpolation f(x) between (x1,f1) and (x2,f2)
            frac = f1 / (f1 - f2)
            x0 = x1 + frac * (x2 - x1)
            xs.append(x0)

    xs = sorted(set(round(x, 6) for x in xs))
    return xs


def interp_logy(x: Sequence[float], y: Sequence[float], x0: float) -> float:
    """
    Interpolate y(x0) assuming y varies linearly in log-space between points.
    Requires x sorted ascending and positive y.
    """
    if x0 <= x[0]:
        return float(y[0])
    if x0 >= x[-1]:
        return float(y[-1])

    for i in range(len(x) - 1):
        x1, x2 = float(x[i]), float(x[i + 1])
        if not (x1 <= x0 <= x2):
            continue

        y1, y2 = float(y[i]), float(y[i + 1])
        if y1 <= 0 or y2 <= 0:
            # fallback to linear if something is non-positive
            t = (x0 - x1) / (x2 - x1)
            return y1 + t * (y2 - y1)

        ly1, ly2 = math.log(y1), math.log(y2)
        t = (x0 - x1) / (x2 - x1)
        return math.exp(ly1 + t * (ly2 - ly1))

    raise ValueError("x0 not bracketed; check x sorting")


def xs_limit_at_mass(
    masses: Sequence[float],
    exp_median: Sequence[float],
    mass0: float,
) -> float:
    return interp_logy(masses, exp_median, mass0)


def theory_xs_at_mass(
    masses: Sequence[float],
    theory_vals: Sequence[float],
    mass0: float,
) -> float:
    return interp_logy(masses, theory_vals, mass0)



def plot_limits(
    production_mode: str,
    theory_xs_dict: Mapping[int, float],
    root_dir: str,
    out_dir: str,
    misc: str,
    strategy: str,
    *,
    ylim: Tuple[float, float] = (1e-3, 1e2),
    lumi: float = 137.0,
    annotate_intersections: bool = True,
) -> None:
    mplhep.style.use("CMS")

    limits = extract_limits(production_mode, root_dir)
    masses_all = sorted(limits.keys())

    valid_mass_points: List[int] = []
    exp: List[float] = []
    err_1sigma_up: List[float] = []
    err_1sigma_dn: List[float] = []
    err_2sigma_up: List[float] = []
    err_2sigma_dn: List[float] = []
    theory_vals: List[float] = []

    for m in masses_all:
        central = limits[m][2]
        if central is None or m not in theory_xs_dict:
            continue

        q025, q160, q500, q840, q975 = limits[m]
        q025 = q025 if q025 is not None else central
        q160 = q160 if q160 is not None else central
        q840 = q840 if q840 is not None else central
        q975 = q975 if q975 is not None else central

        valid_mass_points.append(m)
        exp.append(float(central))
        err_1sigma_up.append(float(q840 - central))
        err_1sigma_dn.append(float(central - q160))
        err_2sigma_up.append(float(q975 - central))
        err_2sigma_dn.append(float(central - q025))
        theory_vals.append(float(theory_xs_dict[m]))

    print(f"\nQuantile values for {production_mode}:\n{'Mass':<10}  2.5%     16%      50%      84%      97.5%")
    for m in valid_mass_points:
        q_vals = limits[m]

        def fmt(x: Optional[float]) -> str:
            return "None" if x is None else f"{x:.3f}"

        print(
            f"{m:<10}  {fmt(q_vals[0]):<8} {fmt(q_vals[1]):<8} {fmt(q_vals[2]):<8} {fmt(q_vals[3]):<8} {fmt(q_vals[4]):<8}"
        )

    if not valid_mass_points:
        print(f"No valid data points for {production_mode}")
        return

    x = np.array(valid_mass_points, dtype=float)
    y = np.array(exp, dtype=float)
    y_th = np.array(theory_vals, dtype=float)

    intersections = estimate_mass_intersections(x, y, y_th)
    if intersections:
        msg = ", ".join(f"{m:.1f} GeV" for m in intersections)
        print(f"Intersection mass(es) theory=expected for {production_mode}: {msg}")
    else:
        print(f"No theory=expected intersection found for {production_mode}")


    if intersections:
        for m0 in intersections:
            xs_lim = xs_limit_at_mass(x, y, m0)
            xs_th = theory_xs_at_mass(x, y_th, m0)
            print(f"At m* = {m0:.1f} GeV: σ95(exp) ≈ {xs_lim:.4g} fb, σ_theory ≈ {xs_th:.4g} fb")
    

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.text(
        0.02,
        0.90,
        production_mode,
        transform=ax.transAxes,
        fontsize=14,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.fill_between(
        x,
        y - np.array(err_2sigma_dn),
        y + np.array(err_2sigma_up),
        facecolor=BRAZIL_YELLOW_2SIGMA,
        edgecolor="none",
        alpha=1.0,
        label="Expected ± 2σ",
        zorder=1,
    )
    ax.fill_between(
        x,
        y - np.array(err_1sigma_dn),
        y + np.array(err_1sigma_up),
        facecolor=BRAZIL_GREEN_1SIGMA,
        edgecolor="none",
        alpha=1.0,
        label="Expected ± 1σ",
        zorder=2,
    )

    ax.plot(x, y, "k--", label="Expected Median 95% CL", zorder=3)
    ax.plot(x, y_th, "r-", label="Model cross-section", zorder=4)

    if annotate_intersections and intersections:
        for j, m0 in enumerate(intersections, start=1):
            ax.axvline(m0, linestyle=":", linewidth=1.5, color="k", zorder=5)
            ax.text(
                m0,
                ylim[1] / (1.5 + 0.2 * j),
                f"m ≈ {m0:.0f} GeV",
                rotation=90,
                va="top",
                ha="right",
                fontsize=11,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                zorder=6,
            )

    ax.set_yscale("log")
    #ax.set_ylim(*ylim)
    ax.set_ylim(1e-2,1e3)
    ax.set_xlabel("Mass (GeV)")
    ax.set_ylabel("95% CL limit on σ (fb)")

    mplhep.cms.label("Simulation Preliminary", data=True, loc=0, ax=ax, lumi=lumi)
    ax.legend(loc="best", fontsize=15)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    out_path = _ensure_out_dir(out_dir)
    stem = f"{misc}_{strategy}_PreliminaryLimits_{production_mode}"
    pdf_path = out_path / f"{stem}.pdf"
    png_path = out_path / f"{stem}.png"

    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot expected limits from Combine HybridNew ROOT outputs.")
    parser.add_argument("--preset", choices=sorted(ROOT_DIR_PRESETS.keys()), help="Choose a predefined root_dir.")
    parser.add_argument("--root-dir", help="Path to directory containing ROOT outputs. Overrides --preset.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory to save plots.")
    parser.add_argument("--misc", default="all_systs", help="Label used in output filename.")
    parser.add_argument("--strategy", default="Ph", help="Label used in output filename.")
    parser.add_argument("--modes", nargs="*", default=None, help="Subset of production modes to run (default: all).")
    parser.add_argument("--lumi", type=float, default=137.0, help="Integrated luminosity label.")
    parser.add_argument("--ymin", type=float, default=1e-3)
    parser.add_argument("--ymax", type=float, default=1e2)
    parser.add_argument(
        "--no-annotate-intersections",
        action="store_true",
        help="Do not draw intersection mass lines/labels on the plot.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    root_dir = args.root_dir
    if not root_dir:
        if not args.preset:
            raise SystemExit("Provide either --root-dir or --preset.")
        root_dir = ROOT_DIR_PRESETS[args.preset]

    if args.modes:
        unknown = [m for m in args.modes if m not in production_modes]
        if unknown:
            raise SystemExit(f"Unknown mode(s): {unknown}. Known: {sorted(production_modes.keys())}")
        selected_modes: Iterable[str] = args.modes
    else:
        selected_modes = production_modes.keys()

    for mode in selected_modes:
        print(f"\nProcessing mode: {mode}")
        plot_limits(
            mode,
            production_modes[mode],
            root_dir=root_dir,
            out_dir=args.out_dir,
            misc=args.misc,
            strategy=args.strategy,
            ylim=(args.ymin, args.ymax),
            lumi=args.lumi,
            annotate_intersections=not args.no_annotate_intersections,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
