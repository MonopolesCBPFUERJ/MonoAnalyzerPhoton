# path: abcd_analysis.py
"""
Standalone ABCD analysis runner (lxplus-friendly).

Example:
  python abcd_analysis.py \
    --year 2018 \
    --unblind 0 \
    --regions-scheme default \
    --flavour MET \
    --trigger HLT_PFMET300 \
    --tree monopoles \
    --outdir ./out \
    /path/to/file1.root /path/to/file2.root
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import uproot

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mplhep as hep
from scipy.stats import binom

LOGGER = logging.getLogger("abcd")


# -----------------------------
# ROOT I/O
# -----------------------------
def _autodetect_tree_name(root_file: uproot.ReadOnlyFile) -> str:
    for key, obj in root_file.items():
        classname = getattr(obj, "classname", "")
        if "TTree" in classname or hasattr(obj, "arrays"):
            return key.split(";")[0]
    raise ValueError("Could not autodetect a TTree in the ROOT file. Pass --tree explicitly.")


def read_root_file(
    file_path: str | Path,
    branches: list[str],
    tree_name: Optional[str] = "monopoles",
    allow_missing_branches: bool = True,
) -> pd.DataFrame:
    """
    Read a single ROOT file into a pandas DataFrame via uproot.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"ROOT file not found: {file_path}")

    with uproot.open(file_path) as f:
        tname = tree_name or _autodetect_tree_name(f)
        if tname not in f:
            keys = [k.split(";")[0] for k in f.keys()]
            raise KeyError(f"Tree '{tname}' not found in {file_path}. Available: {keys}")

        tree = f[tname]
        available = set(tree.keys())

        missing = [b for b in branches if b not in available]
        if missing and not allow_missing_branches:
            raise KeyError(f"Missing branches in {file_path}: {missing}")

        wanted = [b for b in branches if b in available]
        df = tree.arrays(wanted, library="pd") if wanted else pd.DataFrame()

        if missing:
            for b in missing:
                df[b] = np.nan

        return df[branches]


def read_multiple_root_files(
    file_paths: Iterable[str | Path],
    branches: list[str],
    tree_name: Optional[str] = "monopoles",
    allow_missing_branches: bool = True,
) -> pd.DataFrame:
    """
    Read multiple ROOT files and concatenate.
    """
    frames = [
        read_root_file(fp, branches, tree_name=tree_name, allow_missing_branches=allow_missing_branches)
        for fp in file_paths
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=branches)


# -----------------------------
# Significance utilities
# -----------------------------
def add_significance_columns(df: pd.DataFrame, p: float = 0.07) -> pd.DataFrame:
    """
    Adds:
      - binomial_result = P(X>=k) where X~Bin(n,p)
      - Significance = sqrt(-log(binomial_result))
    """
    out = df.copy()

    n = out["cand_SubHits"].astype("int64").to_numpy()
    k = out["cand_SatSubHits"].astype("int64").to_numpy()

    n = np.where(n < 0, 0, n)
    k = np.where(k < 0, 0, k)
    k = np.minimum(k, n)

    # P(X>=k) = sf(k-1)
    tail = binom.sf(k - 1, n, p)
    tail = np.maximum(tail, 1e-300)

    out["binomial_result"] = tail
    out["Significance"] = np.sqrt(-np.log(tail))
    return out


# -----------------------------
# Your analysis code
# -----------------------------
def do_analysis(data: pd.DataFrame, year: str) -> pd.DataFrame:
    if year in ["2016", "2016APV"]:
        condition = (
            (data["cand_dist"] < 0.5)
            & (data["cand_HIso"] < 10)
            & (np.abs(data["cand_XYPar0"]) < 0.6)
            & (np.abs(data["cand_XYPar1"]) < 10)
            & (np.abs(data["cand_XYPar2"]) > 1000)
            & (np.abs(data["cand_RZPar0"]) < 10)
            & (np.abs(data["cand_RZPar1"]) < 999)
            & (np.abs(data["cand_RZPar2"]) < 0.005)
            & (data["cand_e55"] > 175)
        )
    elif year in [
        "2017",
        "2018",
        "2017-2018",
        "2017-2018_HEM",
        "2018BC",
        "2018D",
        "2018D_HEM",
        "2018C_HEM",
        "2018C",
    ]:
        condition = (
            (data["cand_dist"] < 0.5)
            & (data["cand_HIso"] < 10)
            & (np.abs(data["cand_XYPar0"]) < 0.6)
            & (np.abs(data["cand_XYPar1"]) < 10)
            & (np.abs(data["cand_XYPar2"]) > 1000)
            & (np.abs(data["cand_RZPar0"]) < 10)
            & (np.abs(data["cand_RZPar1"]) < 999)
            & (np.abs(data["cand_RZPar2"]) < 0.005)
            & (data["cand_e55"] > 200)
        )
    else:
        raise ValueError(f"Unsupported year: {year}")

    filtered = data.loc[condition].copy()
    filtered = add_significance_columns(filtered, p=0.07)

    LOGGER.info("Before do_analysis: events=%d", len(data))
    LOGGER.info("After do_analysis:  events=%d", len(filtered))
    return filtered


def do_analysis_PFMET(data: pd.DataFrame, year: str) -> pd.DataFrame:
    if year in ["2016", "2016APV", "2016-2016APV"]:
        condition = (
            (data["cand_dist"] < 0.5)
            & (data["cand_HIso"] < 10)
            & (np.abs(data["cand_XYPar0"]) < 0.6)
            & (np.abs(data["cand_XYPar1"]) < 10)
            & (np.abs(data["cand_XYPar2"]) > 1000)
            & (np.abs(data["cand_RZPar0"]) < 10)
            & (np.abs(data["cand_RZPar1"]) < 999)
            & (np.abs(data["cand_RZPar2"]) < 0.005)
            & (np.abs(data["PFMET_pt"]) > 500)
        )
    elif year in [
        "2017",
        "2018",
        "2017-2018",
        "2017-2018_HEM",
        "2018BC",
        "2018D",
        "2018D_HEM",
        "2018C_HEM",
        "2018C",
    ]:
        condition = (
            (data["cand_dist"] < 0.5)
            & (data["cand_HIso"] < 10)
            & (np.abs(data["cand_XYPar0"]) < 0.6)
            & (np.abs(data["cand_XYPar1"]) < 10)
            & (np.abs(data["cand_XYPar2"]) > 1000)
            & (np.abs(data["cand_RZPar0"]) < 10)
            & (np.abs(data["cand_RZPar1"]) < 999)
            & (np.abs(data["cand_RZPar2"]) < 0.005)
            & (np.abs(data["PFMET_pt"]) > 400)
        )
    else:
        raise ValueError(f"Unsupported year: {year}")

    filtered = data.loc[condition].copy()
    filtered = add_significance_columns(filtered, p=0.07)

    LOGGER.info("After do_analysis_PFMET: events=%d", len(filtered))
    return filtered


def select_highest_significance_rows(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("eventNumber")["Significance"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def get_loose_values(year: str) -> tuple[float, float]:
    if year in ["2016", "2016APV", "2016-2016APV"]:
        return 0.60, 6.5
    if year in [
        "2017",
        "2018",
        "2017-2018",
        "2017-2018_HEM",
        "2018BC",
        "2018D",
        "2018D_HEM",
        "2018C_HEM",
        "2018C",
    ]:
        return 0.75, 7.0
    raise ValueError(f"Invalid year: {year}")


def count_events_in_regions(
    data: pd.DataFrame, regions: pd.DataFrame, x_key: str, y_key: str
) -> dict[str, int]:
    counts: dict[str, int] = {str(row["region"]): 0 for _, row in regions.iterrows()}
    for _, row in regions.iterrows():
        cond = (
            (data[x_key] >= row["xmin"])
            & (data[x_key] <= row["xmax"])
            & (data[y_key] >= row["ymin"])
            & (data[y_key] <= row["ymax"])
        )
        counts[str(row["region"])] = int(data.loc[cond].shape[0])
    return counts


def calculate_region_count(numerator_factors: list[float], denominator_factors: list[float]) -> float:
    numerator = float(np.prod([n + 1 for n in numerator_factors]))
    denominator = float(np.prod([d + 1 for d in denominator_factors]))
    return numerator / denominator if denominator != 0 else float("nan")


def calculate_error(n_value: float, factors: list[float]) -> float:
    return float(n_value) * math.sqrt(sum([1.0 / (factor + 1.0) for factor in factors]))


def old_get_regions(regions_scheme: str, f51_loose: float, dEdx_loose: float) -> pd.DataFrame:
    base_data = {
        "region": ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
        "xmin": [0.0, f51_loose, 0.85, 0.0, f51_loose, 0.85, 0.0, f51_loose, 0.85],
        "xmax": [f51_loose, 0.85, 1.05, f51_loose, 0.85, 1.05, f51_loose, 0.85, 1.05],
        "ymin": [0.0, 0.0, 0.0, dEdx_loose, dEdx_loose, dEdx_loose, 9.0, 9.0, 9.0],
        "ymax": [dEdx_loose, dEdx_loose, dEdx_loose, 9.0, 9.0, 9.0, 30.0, 30.0, 30.0],
    }
    df = pd.DataFrame(base_data)

    if regions_scheme == "spike_out":
        df.loc[[2, 5, 8], "xmax"] = 0.95
    elif regions_scheme == "spike_only":
        #df.loc[[1, 4, 7], "xmax"] = 0.95
        #df.loc[[2, 5, 8], "xmax"] = 0.95
        df.loc[2, "xmin"] = 0.95
        df.loc[1, "xmax"] = 0.95
        df.loc[2, "xmax"] = 1.05
        df.loc[4, "xmax"] = 0.95
        df.loc[5, "xmax"] = 1.05
        df.loc[7, "xmax"] = 0.95
        df.loc[8, "xmax"] = 1.05
    elif regions_scheme in ["default", "nominal"]:
        pass
    else:
        raise ValueError(f"Unknown regions_scheme: {regions_scheme}")

    return df


def get_regions(regions_scheme: str, f51_loose: float, dEdx_loose: float) -> pd.DataFrame:
    if regions_scheme in ["default", "nominal"]:
        x_edges = [0.0, f51_loose, 0.85, 1.05]
    elif regions_scheme == "spike_only":
        x_edges = [0.0, f51_loose, 0.95, 1.05]
    elif regions_scheme == "spike_out":
        x_edges = [0.0, f51_loose, 0.85, 0.95]
    else:
        raise ValueError(f"Unknown regions_scheme: {regions_scheme}")

    y_edges = [0.0, dEdx_loose, 9.0, 30.0]

    rows = []
    region = 1
    for iy in range(3):
        for ix in range(3):
            rows.append(
                {
                    "region": str(region),
                    "xmin": x_edges[ix],
                    "xmax": x_edges[ix + 1],
                    "ymin": y_edges[iy],
                    "ymax": y_edges[iy + 1],
                }
            )
            region += 1

    return pd.DataFrame(rows)


def ABCD_plot(
    data: pd.DataFrame,
    year: str,
    unblind: int,
    regions_scheme: str,
    flavour: str,
    trigger: str,
    outdir: Path,
) -> pd.DataFrame:
    unblind = int(unblind)

    f51_loose, dEdx_loose = get_loose_values(year)
    LOGGER.info(
        "Year=%s f51_loose=%.3f dEdx_loose=%.3f",
        year,
        f51_loose,
        dEdx_loose,
    )

    regions = get_regions(regions_scheme, f51_loose, dEdx_loose)
    data = data.drop_duplicates(subset=["cand_f51", "Significance"]).copy()

    # -------------------------------------------------------------------------
    # Fixed lower f51 boundary for the unblind scan approaches.
    #
    # This is intentionally independent of the year-dependent f51_loose.
    # -------------------------------------------------------------------------
    f51_unblind_min = 0.85

    # -------------------------------------------------------------------------
    # 01/02 family:
    #
    # 9501, 9601, ..., 9901 -> high-f51 strategy
    # 9502, 9602, ..., 9902 -> intermediate-f51 strategy
    #
    # In this family, regions 5,6,8,9 must remain blinded.
    # -------------------------------------------------------------------------
    f51_unblind_map = {
        9501: {"separator": 0.95, "region": "high"},
        9502: {"separator": 0.95, "region": "intermediate"},

        9601: {"separator": 0.96, "region": "high"},
        9602: {"separator": 0.96, "region": "intermediate"},

        9701: {"separator": 0.97, "region": "high"},
        9702: {"separator": 0.97, "region": "intermediate"},

        9801: {"separator": 0.98, "region": "high"},
        9802: {"separator": 0.98, "region": "intermediate"},

        9901: {"separator": 0.99, "region": "high"},
        9902: {"separator": 0.99, "region": "intermediate"},
    }

    # -------------------------------------------------------------------------
    # 03/04 family:
    #
    # 9503, 9603, ..., 9903 -> intermediate contribution, CRs unblinded
    # 9504, 9604, ..., 9904 -> spike contribution, CRs unblinded
    #
    # In this family:
    #   - regions 5,6,8 are unblinded
    #   - region 9 remains blinded
    # -------------------------------------------------------------------------
    f51_unblind_568_map = {
        9503: {"separator": 0.95, "component": "intermediate"},
        9504: {"separator": 0.95, "component": "spike"},

        9603: {"separator": 0.96, "component": "intermediate"},
        9604: {"separator": 0.96, "component": "spike"},

        9703: {"separator": 0.97, "component": "intermediate"},
        9704: {"separator": 0.97, "component": "spike"},

        9803: {"separator": 0.98, "component": "intermediate"},
        9804: {"separator": 0.98, "component": "spike"},

        9903: {"separator": 0.99, "component": "intermediate"},
        9904: {"separator": 0.99, "component": "spike"},
    }

    # Legacy aliases.
    legacy_unblind_aliases = {
        21: 9501,
        22: 9502,
    }

    resolved_unblind = legacy_unblind_aliases.get(unblind, unblind)

    # -------------------------------------------------------------------------
    # Region-mask helpers.
    #
    # These are important because the region blinding should be based on the
    # ABCD region definitions, not only on f51 threshold cuts.
    # -------------------------------------------------------------------------
    def make_region_mask(region_id: str, frame=None) -> pd.Series:
        if frame is None:
            frame = data

        region_row = regions[regions["region"].astype(str) == str(region_id)]

        if len(region_row) != 1:
            raise ValueError(
                f"Expected exactly one definition for region {region_id}, "
                f"found {len(region_row)}"
            )

        row = region_row.iloc[0]
        eps = 1e-12

        return (
            (frame["cand_f51"] >= row["xmin"] - eps)
            & (frame["cand_f51"] <= row["xmax"] + eps)
            & (frame["Significance"] >= row["ymin"] - eps)
            & (frame["Significance"] <= row["ymax"] + eps)
        )

    def make_regions_mask(region_ids, frame=None) -> pd.Series:
        if frame is None:
            frame = data

        mask = pd.Series(False, index=frame.index)

        for region_id in region_ids:
            mask = mask | make_region_mask(region_id, frame=frame)

        return mask

    def make_region9_unblind_mask(frame=None) -> pd.Series:
        """
        Region 9 mask for the unblind scan.

        The lower f51 boundary is fixed to f51_unblind_min = 0.85,
        not to the year-dependent f51_loose.
        """
        if frame is None:
            frame = data

        region_row = regions[regions["region"].astype(str) == "9"]

        if len(region_row) != 1:
            raise ValueError(
                f"Expected exactly one definition for region 9, found {len(region_row)}"
            )

        row = region_row.iloc[0]
        eps = 1e-12

        return (
            (frame["cand_f51"] >= f51_unblind_min - eps)
            & (frame["cand_f51"] <= row["xmax"] + eps)
            & (frame["Significance"] >= row["ymin"] - eps)
            & (frame["Significance"] <= row["ymax"] + eps)
        )

    def log_remaining_regions(region_ids, frame=None) -> None:
        if frame is None:
            frame = data

        remaining = {
            str(region_id): int(make_region_mask(str(region_id), frame=frame).sum())
            for region_id in region_ids
        }

        LOGGER.info("Remaining events in regions %s: %s", region_ids, remaining)

    # -------------------------------------------------------------------------
    # Apply blinding / unblinding configuration.
    # -------------------------------------------------------------------------
    if unblind == 0:
        # Default fully blinded ABCD mode:
        # regions 5,6,8,9 are blinded.
        regions_5689_blind = make_regions_mask(["5", "6", "8", "9"], frame=data)

        LOGGER.info(
            "unblind=0: removing regions 5/6/8/9, events removed=%d",
            int(regions_5689_blind.sum()),
        )

        data = data.loc[~regions_5689_blind].copy()
        log_remaining_regions(["5", "6", "8", "9"], frame=data)

    elif unblind == 1:
        # Regions 5,6,8 unblinded.
        # Region 9 remains blinded.
        region9_blind = make_region9_unblind_mask(frame=data)

        LOGGER.info(
            "unblind=1: removing region 9 only, events removed=%d",
            int(region9_blind.sum()),
        )

        data = data.loc[~region9_blind].copy()
        log_remaining_regions(["9"], frame=data)

    elif resolved_unblind in f51_unblind_map:
        # 9501/9502/.../9901/9902:
        # regions 5,6,8,9 are blinded.
        config = f51_unblind_map[resolved_unblind]

        f51_separator = config["separator"]
        f51_region = config["region"]

        LOGGER.info(
            "Using f51 scan with regions 5/6/8/9 blinded: "
            "unblind=%s resolved_unblind=%s separator=%.2f region=%s "
            "f51_unblind_min=%.2f",
            unblind,
            resolved_unblind,
            f51_separator,
            f51_region,
            f51_unblind_min,
        )

        if f51_region == "high":
            extra_blind = (
                (data["cand_f51"] > f51_separator)
                & (data["cand_f51"] <= 1.0)
                & (data["Significance"] < dEdx_loose)
            )

        elif f51_region == "intermediate":
            extra_blind = (
                (data["cand_f51"] >= f51_unblind_min)
                & (data["cand_f51"] <= f51_separator)
                & (data["Significance"] < 9.0)
            )

        else:
            raise ValueError(f"Unknown f51 region configuration: {f51_region}")

        # Explicitly blind the ABCD regions, independent of f51_unblind_min.
        # This is needed because Region 5 may start below f51 = 0.85.
        regions_5689_blind = make_regions_mask(["5", "6", "8", "9"], frame=data)

        total_blind = regions_5689_blind | extra_blind

        LOGGER.info(
            "Events removed from blinded regions 5/6/8/9: %d",
            int(regions_5689_blind.sum()),
        )
        LOGGER.info(
            "Events removed by f51 sideband extra_blind: %d",
            int(extra_blind.sum()),
        )

        data = data.loc[~total_blind].copy()
        log_remaining_regions(["5", "6", "8", "9"], frame=data)

    elif resolved_unblind in f51_unblind_568_map:
        # 9503/9504/.../9903/9904:
        # regions 5,6,8 are unblinded.
        # region 9 is blinded.
        config = f51_unblind_568_map[resolved_unblind]

        f51_separator = config["separator"]
        f51_component = config["component"]

        LOGGER.info(
            "Using f51 scan with regions 5/6/8 unblinded and region 9 blinded: "
            "unblind=%s resolved_unblind=%s separator=%.2f component=%s "
            "f51_unblind_min=%.2f",
            unblind,
            resolved_unblind,
            f51_separator,
            f51_component,
            f51_unblind_min,
        )

        region9_blind = make_region9_unblind_mask(frame=data)

        # The f51 scan window starts at 0.85.
        # Below 0.85 is kept as the ordinary low-f51 side.
        f51_scan_window = (
            (data["cand_f51"] >= f51_unblind_min)
            & (data["cand_f51"] <= 1.0)
        )

        if f51_component == "intermediate":
            # Keep only the intermediate contribution inside the scan window:
            #
            #   0.85 <= f51 <= separator
            #
            # Therefore remove the spike side:
            #
            #   f51 > separator
            remove_other_component = (
                f51_scan_window
                & (data["cand_f51"] > f51_separator)
            )

        elif f51_component == "spike":
            # Keep only the spike contribution inside the scan window:
            #
            #   f51 > separator
            #
            # Therefore remove the intermediate side:
            #
            #   0.85 <= f51 <= separator
            remove_other_component = (
                f51_scan_window
                & (data["cand_f51"] <= f51_separator)
            )

        else:
            raise ValueError(f"Unknown f51 component configuration: {f51_component}")

        total_blind = region9_blind | remove_other_component

        LOGGER.info(
            "Events removed from region 9: %d",
            int(region9_blind.sum()),
        )
        LOGGER.info(
            "Events removed outside selected f51 component: %d",
            int(remove_other_component.sum()),
        )

        data = data.loc[~total_blind].copy()

        log_remaining_regions(["9"], frame=data)

        # Regions 5,6,8 should be allowed to remain populated.
        remaining_568 = {
            region_id: int(make_region_mask(region_id, frame=data).sum())
            for region_id in ["5", "6", "8"]
        }
        LOGGER.info(
            "Remaining events in unblinded regions 5/6/8: %s",
            remaining_568,
        )

    elif unblind == 2:
        # Fully unblinded mode.
        LOGGER.info("unblind=2: no additional blinding applied.")

    else:
        LOGGER.warning(
            "Unrecognized unblind value %s. No unblind-specific filtering was applied.",
            unblind,
        )

    # -------------------------------------------------------------------------
    # Count events.
    # -------------------------------------------------------------------------
    region_counts = count_events_in_regions(
        data,
        regions,
        x_key="cand_f51",
        y_key="Significance",
    )

    N1, N2, N3, N4, N7 = (
        region_counts.get("1", 0),
        region_counts.get("2", 0),
        region_counts.get("3", 0),
        region_counts.get("4", 0),
        region_counts.get("7", 0),
    )

    # -------------------------------------------------------------------------
    # Calculate predicted counts.
    # -------------------------------------------------------------------------
    N5 = calculate_region_count([N2, N4], [N1])
    N6 = calculate_region_count([N3, N4], [N1])
    N8 = calculate_region_count([N2, N7], [N1])

    error_calculated_counts = {
        "5": calculate_error(N5, [N2, N4, N1]),
        "6": calculate_error(N6, [N3, N4, N1]),
        "8": calculate_error(N8, [N2, N7, N1]),
    }

    if unblind == 2:
        N9 = calculate_region_count([N3 + N6], [N1 + N2 + N4 + N5])
        error_calculated_counts["9"] = calculate_error(
            N9,
            [N3 + N6, N7 + N8, N1 + N2 + N4 + N5],
        )
    else:
        N9 = calculate_region_count([N3, N7], [N1])
        error_calculated_counts["9"] = calculate_error(N9, [N3, N7, N1])

    calculated_counts = {
        "5": N5,
        "6": N6,
        "8": N8,
        "9": N9,
    }

    # -------------------------------------------------------------------------
    # Decide what is allowed to be shown on the plot.
    #
    # Rule:
    #   - If a region is blinded, show only its predicted count.
    #   - If a region is unblinded, show its actual count.
    #   - If a region is unblinded and also has a prediction, show both.
    # -------------------------------------------------------------------------
    all_regions = {str(r) for r in regions["region"]}

    if unblind == 0 or resolved_unblind in f51_unblind_map:
        # Regions 5,6,8,9 are blinded.
        visible_observed_regions = {"1", "2", "3", "4", "7"}

    elif unblind == 1 or resolved_unblind in f51_unblind_568_map:
        # Regions 5,6,8 are unblinded.
        # Region 9 is blinded.
        visible_observed_regions = {"1", "2", "3", "4", "5", "6", "7", "8"}

    elif unblind == 2:
        # Fully unblinded.
        visible_observed_regions = all_regions

    else:
        visible_observed_regions = all_regions

    visible_predicted_regions = set(calculated_counts.keys())

    redacted_region_counts = {
        str(region): count if str(region) in visible_observed_regions else "BLINDED"
        for region, count in region_counts.items()
    }

    LOGGER.info("unblind=%s resolved_unblind=%s", unblind, resolved_unblind)
    LOGGER.info("Visible observed regions: %s", sorted(visible_observed_regions))
    LOGGER.info("Visible predicted regions: %s", sorted(visible_predicted_regions))
    LOGGER.info(
        "Region Counts, with blinded observed counts redacted: %s",
        redacted_region_counts,
    )
    LOGGER.info("Calculated Counts: %s", calculated_counts)

    # -------------------------------------------------------------------------
    # Plot.
    # -------------------------------------------------------------------------
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.text(
        0.00,
        1.01,
        "CMS",
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=20,
        ha="left",
        va="bottom",
    )
    ax.text(
        0.10,
        1.01,
        "Preliminary",
        transform=ax.transAxes,
        fontsize=20,
        ha="left",
        va="bottom",
    )
    ax.text(
        0.48,
        1.01,
        f"{trigger}",
        transform=ax.transAxes,
        fontsize=14,
        ha="left",
        va="bottom",
    )
    ax.text(
        1.00,
        1.01,
        f"{year}",
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=16,
        ha="right",
        va="bottom",
    )

    ax.hexbin(
        data["cand_f51"],
        data["Significance"],
        gridsize=50,
        cmap="viridis",
        mincnt=1,
        edgecolors="k",
        linewidths=1.0,
    )

    for _, row in regions.iterrows():
        ax.add_patch(
            patches.Rectangle(
                (row["xmin"], row["ymin"]),
                row["xmax"] - row["xmin"],
                row["ymax"] - row["ymin"],
                fill=None,
                edgecolor="black",
                linestyle="--",
                linewidth=1.0,
            )
        )

        region = str(row["region"])
        xmid = (row["xmin"] + row["xmax"]) / 2
        ymin = row["ymin"]
        ymax = row["ymax"]
        height = ymax - ymin
        ymid = 0.5 * (ymin + ymax)
        #ymid = 0.5 + (row["ymin"] + row["ymax"]) / 2

        show_observed = region in visible_observed_regions
        show_predicted = (
            region in visible_predicted_regions
            and region in calculated_counts
            and region in error_calculated_counts
        )

        obs_y = ymid
        pred_y = ymid

        if show_observed and show_predicted:
            pred_y = ymin + 0.72 * height
            obs_y = ymin + 0.28 * height 

        if show_predicted:
            ax.text(
                xmid,
                pred_y,
                f"{calculated_counts[region]:.2f} ±\n{error_calculated_counts[region]:.2f}",
                fontsize=14,
                color="red",
                ha="center",
                va="center",
            )

        if show_observed:
            ax.text(
                xmid,
                obs_y,
                f"{region_counts.get(region, 0)}",
                fontsize=14,
                color="black",
                ha="center",
                va="center",
            )

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 30)
    ax.set_xlabel("f51", fontsize=20)
    ax.set_ylabel(r"$dE/dX_{Significance}$", fontsize=20)

    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{flavour}_ABCD_plot_{regions_scheme}_{year}_{unblind}"
    fig.savefig(outdir / f"{stem}.pdf")
    fig.savefig(outdir / f"{stem}.png")
    plt.show()

    return data



def runABCD(
    unblind: int,
    year: str,
    file_paths: list[str],
    regions_scheme: str,
    flavour: str,
    trigger: str,
    tree: Optional[str],
    outdir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    branches = [
        "cand_dist",
        "cand_HIso",
        "cand_XYPar0",
        "cand_XYPar1",
        "cand_XYPar2",
        "cand_RZPar0",
        "cand_RZPar1",
        "cand_RZPar2",
        "cand_SatSubHits",
        "cand_SubHits",
        "cand_e55",
        "cand_f51",
        "eventNumber",
        "run",
        "lumiBlock",
        "PFMET_pt",
    ]

    data = read_multiple_root_files(file_paths, branches, tree_name=tree or "monopoles")
    LOGGER.info("Loaded rows=%d from %d files", len(data), len(file_paths))

    if trigger in ["HLT_PFMET300", "notPhoton175_PFMET300", "notPhoton200_PFMET250"]:
        filtered = do_analysis_PFMET(data, year)
    elif trigger in ["HLT_Photon200", "HLT_Photon175"]:
        filtered = do_analysis(data, year)
    else:
        raise ValueError(f"Unsupported trigger={trigger}")

    significant = select_highest_significance_rows(filtered)
    final = ABCD_plot(significant, year, unblind, regions_scheme, flavour, trigger, outdir=outdir)
    return data, final


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standalone ABCD analysis runner.")
    p.add_argument("files", nargs="*", help="Input ROOT files (or use --filelist)")
    p.add_argument("--filelist", default=None, help="Text file with one ROOT path per line")
    p.add_argument("--year", required=True)
    p.add_argument("--unblind", type=int, required=True, help="0,1,2,21,22,...")
    p.add_argument("--regions-scheme", default="default", help="default|spike_out|spike_only")
    p.add_argument("--flavour", required=True, choices=["MET", "EGamma"])
    p.add_argument("--trigger", required=True)
    p.add_argument("--tree", default="monopoles", help="ROOT tree name (default: monopoles)")
    p.add_argument("--outdir", default="./out")
    p.add_argument("--loglevel", default="INFO")
    return p

def load_filelist(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--filelist not found: {p}")

    files: list[str] = []
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip().strip('"').strip("'")
        if not line or line.startswith("#"):
            continue
        files.append(line)

    return files


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))
    outdir = Path(args.outdir)

    files: list[str] = list(args.files)

    if args.filelist:
        files.extend(load_filelist(args.filelist))

    # De-dup while preserving order
    seen = set()
    files = [f for f in files if not (f in seen or seen.add(f))]

    if not files:
        raise SystemExit("No input ROOT files resolved. Pass files or use --filelist inputs.txt")

    # Validate paths (EOS should be visible as a regular filesystem path)
    missing = [f for f in files if not Path(f).exists()]
    if missing:
        LOGGER.error("Some input files do not exist (first 10 shown):\n%s", "\n".join(missing[:10]))
        raise SystemExit(f"{len(missing)} input files do not exist. Fix inputs.txt or paths.")

    LOGGER.info("Resolved %d input files. First 5:\n%s", len(files), "\n".join(files[:5]))

    runABCD(
        unblind=args.unblind,
        year=args.year,
        file_paths=files,
        regions_scheme=args.regions_scheme,
        flavour=args.flavour,
        trigger=args.trigger,
        tree=args.tree,
        outdir=outdir,
    )
    return 0




if __name__ == "__main__":
    raise SystemExit(main())
