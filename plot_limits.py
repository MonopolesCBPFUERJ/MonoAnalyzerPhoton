#!/usr/bin/env python3
# file: limits_plot.py

"""
Standalone script to extract HybridNew limits from ROOT files and plot expected
limits vs theory, with mass-limit estimation from theory/expected intersections.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import mplhep
import numpy as np
import uproot


production_modes: Dict[str, Dict[int, float]] = {
    "SpinZero_PhotonFusion": {
        1000: 1.09e4, 1500: 8.62e2, 2000: 9.13e1, 2500: 1.05e1,
        3000: 1.15e0, 3500: 1.07e-1, 4000: 7.62e-3, 4500: 3.50e-4,
    },
    "SpinZero_DrellYan": {
        1000: 1.45e1, 1500: 8.12e-1, 2000: 6.04e-2, 2500: 4.99e-3,
        3000: 4.42e-4, 3500: 4.17e-5, 4000: 3.69e-6, 4500: 2.37e-7,
    },
    "SpinHalf_PhotonFusion": {
        1000: 5.10e4, 1500: 3.65e3, 2000: 3.53e2, 2500: 3.71e1,
        3000: 3.73e0, 3500: 3.22e-1, 4000: 2.12e-2, 4500: 9.05e-3,
    },
    "SpinHalf_DrellYan": {
        1000: 2.40e2, 1500: 1.63e1, 2000: 1.47e0, 2500: 1.43e-1,
        3000: 1.44e-2, 3500: 1.54e-3, 4000: 1.65e-4, 4500: 1.38e-5,
    },
}

quantiles: Dict[str, int] = {
    "0.025": 0, "0.160": 1, "0.500": 2, "0.840": 3, "0.975": 4,
}

ROOT_DIR_PRESETS: Dict[str, str] = {
    "20Feb": "/eos/user/t/tmenezes/Monopole_Ntuples/Files_Limits/20Feb_monopole-combine-output-v2/",
    "wout_systs": "/eos/user/t/tmenezes/Monopole_Ntuples/Files_Limits/monopole-combine-output-freezenui-wout-systematics",
    "v3": "/eos/user/t/tmenezes/Monopole_Ntuples/Files_Limits/monopole-combine-output-v3/",
    "Photon_all_systs": "/eos/user/t/tmenezes/Monopole_Ntuples/Files_Limits/Photon-monopole-combine-output-all-systs/",
    "PFMET_all_systs": "/eos/user/t/tmenezes/Monopole_Ntuples/Files_Limits/PFMET-monopole-combine-output-all-systs/",
    "Combined_all_systs": "/eos/user/t/tmenezes/Monopole_Ntuples/Files_Limits/Combined-monopole-combine-output-all-systs/",
    "Updated_Combined_all_systs": "/eos/user/t/tmenezes/Monopole_Ntuples/Files_Limits/Updated_Combined-monopole-combine-output/",
    "rateParam_Photon_all_systs": "/eos/user/t/tmenezes/Monopole_Ntuples/Files_Limits/rateParam_Photon-monopole-combine-output/",
    "rateParam_PFMET_all_systs": "/eos/user/t/tmenezes/Monopole_Ntuples/Files_Limits/rateParam_PFMET-monopole-combine-output/",
    "rateParam_Combined_all_systs": "/eos/user/t/tmenezes/Monopole_Ntuples/Files_Limits/rateParam_Combined-monopole-combine-output/",
}

DEFAULT_OUT_DIR = "/eos/user/t/tmenezes/www/Monopoles/Limits/rateParam"

# Brazil band colors (match reference)
#BRAZIL_YELLOW_2SIGMA = "#FFF200"
#BRAZIL_GREEN_1SIGMA = "#00A651"

# Official CMS CAT recommendations (CMS DP-2024/040)
BRAZIL_YELLOW_2SIGMA = "#F5BB54" # 95% expected
BRAZIL_GREEN_1SIGMA = "#607641"  # 68% expected

def _pretty_mode_name(mode: str) -> str:
    """SpinZero_DrellYan -> 'SpinZero Drell-Yan'."""
    return mode.replace("_", " ").replace("DrellYan", "Drell-Yan").replace("PhotonFusion", "Photon Fusion")


def extract_limits(production_mode: str, root_dir: str) -> Dict[int, List[Optional[float]]]:
    limits: defaultdict[int, List[Optional[float]]] = defaultdict(lambda: [None] * 5)
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"root_dir does not exist: {root_dir}")

    pattern = re.compile(r"higgsCombine\.(.+?)\.HybridNew\.mH(\d+)\.quant0\.(\d+)\.root$")

    for p in root_path.iterdir():
        if not p.name.endswith(".root") or production_mode not in p.name:
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
    masses: Sequence[float], expected: Sequence[float], theory: Sequence[float],
) -> List[float]:
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
            frac = f1 / (f1 - f2)
            xs.append(x1 + frac * (x2 - x1))
    return sorted(set(round(x, 6) for x in xs))


def interp_logy(x: Sequence[float], y: Sequence[float], x0: float) -> float:
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
            t = (x0 - x1) / (x2 - x1)
            return y1 + t * (y2 - y1)
        ly1, ly2 = math.log(y1), math.log(y2)
        t = (x0 - x1) / (x2 - x1)
        return math.exp(ly1 + t * (ly2 - ly1))
    raise ValueError("x0 not bracketed; check x sorting")


def xs_limit_at_mass(masses, exp_median, mass0):
    return interp_logy(masses, exp_median, mass0)


def theory_xs_at_mass(masses, theory_vals, mass0):
    return interp_logy(masses, theory_vals, mass0)


def plot_limits(
    production_mode: str,
    theory_xs_dict: Mapping[int, float],
    root_dir: str,
    out_dir: str,
    misc: str,
    strategy: str,
    *,
    ylim: Tuple[float, float] = (1e-1, 2e2),
    lumi: float = 137.2,
    annotate_intersections: bool = False,
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
        fmt = lambda x: "None" if x is None else f"{x:.3f}"
        print(f"{m:<10}  {fmt(q_vals[0]):<8} {fmt(q_vals[1]):<8} {fmt(q_vals[2]):<8} {fmt(q_vals[3]):<8} {fmt(q_vals[4]):<8}")

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
        for m0 in intersections:
            xs_lim = xs_limit_at_mass(x, y, m0)
            xs_th = theory_xs_at_mass(x, y_th, m0)
            print(f"At m* = {m0:.1f} GeV: σ95(exp) ≈ {xs_lim:.4g} fb, σ_theory ≈ {xs_th:.4g} fb")
    else:
        print(f"No theory=expected intersection found for {production_mode}")

    # ---------- plotting ----------
    fig, ax = plt.subplots(figsize=(10, 7.5))

    # Brazil bands
    ax.fill_between(
        x, y - np.array(err_2sigma_dn), y + np.array(err_2sigma_up),
        facecolor=BRAZIL_YELLOW_2SIGMA, edgecolor="none", linewidth=0, zorder=1,
    )
    ax.fill_between(
        x, y - np.array(err_1sigma_dn), y + np.array(err_1sigma_up),
        facecolor=BRAZIL_GREEN_1SIGMA, edgecolor="none", linewidth=0, zorder=2,
    )

    # Expected median + theory
    exp_line, = ax.plot(x, y, color="black", linestyle="--", linewidth=2, zorder=4)
    th_line,  = ax.plot(x, y_th, color="red", linestyle="-", linewidth=2, zorder=5)

    if annotate_intersections and intersections:
        for j, m0 in enumerate(intersections, start=1):
            ax.axvline(m0, linestyle=":", linewidth=1.2, color="k", zorder=3)
            ax.text(
                m0, ylim[1] / (1.5 + 0.2 * j), f"m ≈ {m0:.0f} GeV",
                rotation=90, va="top", ha="right", fontsize=11,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), zorder=6,
            )

    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_xlim(x.min(), x.max())
    ax.set_xlabel("m [GeV]")
    ax.set_ylabel(r"$\sigma$ [fb]")

    mplhep.cms.label("Preliminary", data=True, loc=0, ax=ax, lumi=lumi)

    # Legend: mode name as header, then entries in reference order
    header = Line2D([], [], color="none", label=_pretty_mode_name(production_mode))
    handles = [
        header,
        Line2D([], [], color="black", linestyle="--", linewidth=2, label="Exp. 95% CL limit"),
        Patch(facecolor=BRAZIL_GREEN_1SIGMA, edgecolor="none", label="Exp. (68%)"),
        Patch(facecolor=BRAZIL_YELLOW_2SIGMA, edgecolor="none", label="Exp. (95%)"),
        Line2D([], [], color="red", linestyle="-", linewidth=2, label=r"$|g| = 1\,g_{D}$"),
    ]
    leg = ax.legend(
        handles=handles, loc="upper right", fontsize=14,
        frameon=False, handlelength=2.2, borderpad=0.6, labelspacing=0.5,
    )
    # Make the header line bold-ish and left-aligned with no marker
    leg.get_texts()[0].set_fontweight("bold")

    ax.grid(True, which="major", linestyle="-", linewidth=0.4, alpha=0.3)
    ax.tick_params(which="both", direction="in", top=True, right=True)

    out_path = _ensure_out_dir(out_dir)
    stem = f"{misc}_{strategy}_PreliminaryLimits_{production_mode}"
    fig.savefig(out_path / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_path / f"{stem}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"Saved: {out_path / f'{stem}.pdf'}")
    print(f"Saved: {out_path / f'{stem}.png'}")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot expected limits from Combine HybridNew ROOT outputs.")
    parser.add_argument("--preset", choices=sorted(ROOT_DIR_PRESETS.keys()))
    parser.add_argument("--root-dir")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--misc", default="all_systs")
    parser.add_argument("--strategy", default="Ph")
    parser.add_argument("--modes", nargs="*", default=None)
    parser.add_argument("--lumi", type=float, default=137.2)
    parser.add_argument("--ymin", type=float, default=1e-1)
    parser.add_argument("--ymax", type=float, default=2e2)
    parser.add_argument("--annotate-intersections", action="store_true",
                        help="Draw vertical lines/labels at theory=expected crossings.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    root_dir = args.root_dir or (ROOT_DIR_PRESETS[args.preset] if args.preset else None)
    if not root_dir:
        raise SystemExit("Provide either --root-dir or --preset.")

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
            mode, production_modes[mode],
            root_dir=root_dir, out_dir=args.out_dir,
            misc=args.misc, strategy=args.strategy,
            ylim=(args.ymin, args.ymax), lumi=args.lumi,
            annotate_intersections=args.annotate_intersections,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

