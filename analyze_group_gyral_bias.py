# --- analyze_group_gyral_bias.py (excerpt) ---
import argparse, os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, spearmanr
import numpy as np
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from utils import nature_style_plot, save_figure
from pathlib import Path
import numpy as np

def plot_curvature_conditions(df, out_dir, title_suffix="", species_map=None):
    """
    Generate bar plots of mean curvature by condition, replicating Figure 3 style.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: ["subject", "meridian", "eccentricity_bin", "area", "mean_curvature"].
    out_dir : Path
        Where to save the plots.
    title_suffix : str
        Extra label for titles and filenames.
    species_map : dict, optional
        Mapping from subject → species (e.g., {"100206": "Human", "M01": "Monkey"}).
        If None, assumes all subjects are "Human".
    """

    df = df.copy()

    # Assign species
    if species_map is not None:
        df["species"] = df["subject"].map(species_map).fillna("Unknown")
    else:
        df["species"] = "Human"  # default
    
    # --- Define conditions ---
    # Panel A: HM, VM, Fov, Per
    def ecc_to_condition(row):
        if row["meridian"] in ["HM", "VM"]:
            return row["meridian"]
        elif row["eccentricity_bin"] in ["0to2", "2to4"]:  # fovea bins
            return "Fov"
        elif row["eccentricity_bin"] in ["8to90"]:  # periphery
            return "Per"
        return None

    df["condition_A"] = df.apply(ecc_to_condition, axis=1)

    # Panel B: HM/VM × area
    df["condition_B"] = df["meridian"].astype(str)  + " (" + df["area"].astype(str)  + ")"

    # --- Aggregation: subject means ---
    subj_means_A = (
        df.dropna(subset=["condition_A"])
        .groupby(["subject", "species", "condition_A"])["mean_curvature"]
        .mean()
        .reset_index()
    )
    subj_means_B = (
        df.dropna(subset=["condition_B"])
        .groupby(["subject", "species", "condition_B"])["mean_curvature"]
        .mean()
        .reset_index()
    )

    # --- Plot helper ---
    def barplot_with_sem(data, condition_col, title, filename):
        summary = (
            data.groupby(["species", condition_col])["mean_curvature"]
            .agg(["mean", "sem"])
            .reset_index()
        )

        plt.figure(figsize=(7,5))
        sns.barplot(
            data=summary,
            y=condition_col,
            x="mean",
            hue="species",
            palette={"Human":"limegreen", "Monkey":"red"},
            orient="h",
            errorbar=None
        )

        # Add error bars manually
        for i, row in summary.iterrows():
            plt.errorbar(
                x=row["mean"],
                y=list(summary[condition_col].unique()).index(row[condition_col]),
                xerr=row["sem"],
                fmt="none",
                color="white",
                capsize=3
            )

        plt.axvline(0, color="white", linewidth=1)
        plt.xlabel("Mean curvature")
        plt.ylabel("")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=300, facecolor="black")
        plt.close()

    # --- Panel A ---
    barplot_with_sem(
        subj_means_A,
        "condition_A",
        f"Mean curvature: HM, VM, Fov, Per {title_suffix}",
        "barplot_curvature_panelA.png"
    )

    # --- Panel B ---
    barplot_with_sem(
        subj_means_B,
        "condition_B",
        f"Mean curvature: HM/VM × Areas {title_suffix}",
        "barplot_curvature_panelB.png"
    )


def safe_groupby_keys(df, keys):
    for key in keys:
        if key in df.columns and pd.api.types.is_categorical_dtype(df[key]):
            df[key] = df[key].astype(str)
    return df


def make_gradient(base_color, n=5, min_lum=0.4, max_lum=1.0):
    """
    Generate a gradient from base_color (hex or name) from light to base.
    """
    r, g, b = to_rgb(base_color)
    steps = np.linspace(max_lum, min_lum, n)
    return [ (r*lum, g*lum, b*lum) for lum in steps ]

def transform_meridians(df, mode, out_dir):
    """
    Transform 'meridian' column and potentially aggregate rows based on the chosen mode.
    Returns transformed df, palette, valid_order, and output subdirectory path.
    """


    # Always make a copy to avoid modifying input df
    df = df.copy()
    collapse = False  # Whether to apply aggregation after transformation

    if mode == "default":
        # Keep all four meridians
        valid_order = ["LHM", "RHM", "LVM", "UVM"]
        palette = {
            "LHM": "#a1d99b", "RHM": "#31a354",
            "LVM": "#fb1209", "UVM": "#3182bd"
        }
        out_subdir = out_dir / "plots_all"

    elif mode == "hm_vm":
        df["meridian"] = df["meridian"].replace({
            "LHM": "HM", "RHM": "HM", "LVM": "VM", "UVM": "VM"
        })
        valid_order = ["HM", "VM"]
        palette = {"HM": "#2ca02c", "VM": "#9467bd"}
        out_subdir = out_dir / "plots_HM_VM"
        collapse = True
        df["streamline_density"] = df["streamline_count"] / df["voxel_count"]

    elif mode == "hm_lvm_uvm":
        # Collapse RHM and LHM into HM
        df["meridian"] = df["meridian"].replace({
            "RHM": "HM",
            "LHM": "HM"
        })

        # Define order and colors
        valid_order = ["HM", "LVM", "UVM"]
        palette = {
            "HM": "#2ca02c",    # Green
            "LVM": "#d62728",   # Red
            "UVM": "#1f77b4"    # Blue
        }

        out_subdir = out_dir / "plots_HM_LVM_UVM"
        collapse = True  # aggregate RHM+LHM → HM
        df["streamline_density"] = df["streamline_count"] / df["voxel_count"]        

    elif mode == "hm_vm_lvm_uvm":
        # Collapse RHM and LHM into HM
        df["meridian"] = df["meridian"].replace({"LHM": "HM", "RHM": "HM"})

        # Compute HM aggregates
        hm_df = df[df["meridian"] == "HM"].copy()
        if "total_curvature" not in hm_df.columns:
            hm_df["total_curvature"] = hm_df["mean_curvature"] * hm_df["voxel_count"]
        hm_df = (
            hm_df.groupby(["subject", "eccentricity_bin"], as_index=False)
            .agg({
                "voxel_count": "sum",
                "streamline_count": "sum",
                "total_curvature": "sum"
            })
        )
        hm_df["mean_curvature"] = hm_df["total_curvature"] / hm_df["voxel_count"]
        hm_df["streamline_density"] = hm_df["streamline_count"] / hm_df["voxel_count"]
        hm_df["meridian"] = "HM"

        # Keep LVM and UVM as-is
        lvm_uvm = df[df["meridian"].isin(["LVM", "UVM"])].copy()

        # --- FIXED VM aggregation ---
        vm_input = df[df["meridian"].isin(["LVM", "UVM"])].copy()
        vm_input = vm_input.dropna(subset=["mean_curvature", "voxel_count", "streamline_count"])
        vm_input["total_curvature"] = vm_input["mean_curvature"] * vm_input["voxel_count"]

        # First group separately by meridian
        vm_split = (
            vm_input.groupby(["subject", "meridian", "eccentricity_bin"], as_index=False)
            .agg({
                "voxel_count": "sum",
                "streamline_count": "sum",
                "total_curvature": "sum"
            })
        )

        # Then collapse LVM + UVM → VM
        vm_df = (
            vm_split.groupby(["subject", "eccentricity_bin"], as_index=False)
            .agg({
                "voxel_count": "sum",
                "streamline_count": "sum",
                "total_curvature": "sum"
            })
        )
        vm_df["mean_curvature"] = vm_df["total_curvature"] / vm_df["voxel_count"]
        vm_df["streamline_density"] = vm_df["streamline_count"] / vm_df["voxel_count"]
        vm_df["meridian"] = "VM"

        # Combine all
        df = pd.concat([hm_df, lvm_uvm, vm_df], ignore_index=True)

        valid_order = ["HM", "VM", "LVM", "UVM"]
        palette = {
            "HM": "#2ca02c",     # green
            "VM": "#9467bd",     # purple
            "LVM": "#d62728",    # red
            "UVM": "#1f77b4",    # blue
        }

        out_subdir = out_dir / "plots_HM_VM_LVM_UVM"
        collapse = False  # Already aggregated above

    elif mode == "hm_rhm_lhm_vm_lvm_uvm":
        # Keep LHM, RHM, LVM, UVM
        # Also synthesize HM = LHM + RHM, VM = LVM + UVM

        # --- Compute HM ---
        lhm_rhm_df = df[df["meridian"].isin(["LHM", "RHM"])].dropna(
            subset=["mean_curvature", "voxel_count", "streamline_count"]
        )
        lhm_rhm_df["total_curvature"] = lhm_rhm_df["mean_curvature"] * lhm_rhm_df["voxel_count"]

        # First group separately by meridian
        hm_split = (
            lhm_rhm_df.groupby(["subject", "meridian", "eccentricity_bin"], as_index=False)
            .agg({
                "voxel_count": "sum",
                "streamline_count": "sum",
                "total_curvature": "sum"
            })
        )

        # Then sum across LHM + RHM → HM
        hm_df = (
            hm_split.groupby(["subject", "eccentricity_bin"], as_index=False)
            .agg({
                "voxel_count": "sum",
                "streamline_count": "sum",
                "total_curvature": "sum"
            })
        )
        hm_df["mean_curvature"] = hm_df["total_curvature"] / hm_df["voxel_count"]
        hm_df["streamline_density"] = hm_df["streamline_count"] / hm_df["voxel_count"]
        hm_df["meridian"] = "HM"

        # --- Compute VM ---
        lvm_uvm_df = df[df["meridian"].isin(["LVM", "UVM"])].dropna(
            subset=["mean_curvature", "voxel_count", "streamline_count"]
        )
        lvm_uvm_df["total_curvature"] = lvm_uvm_df["mean_curvature"] * lvm_uvm_df["voxel_count"]

        # First group by meridian
        vm_split = (
            lvm_uvm_df.groupby(["subject", "meridian", "eccentricity_bin"], as_index=False)
            .agg({
                "voxel_count": "sum",
                "streamline_count": "sum",
                "total_curvature": "sum"
            })
        )

        # Then sum LVM + UVM → VM
        vm_df = (
            vm_split.groupby(["subject", "eccentricity_bin"], as_index=False)
            .agg({
                "voxel_count": "sum",
                "streamline_count": "sum",
                "total_curvature": "sum"
            })
        )
        vm_df["mean_curvature"] = vm_df["total_curvature"] / vm_df["voxel_count"]
        vm_df["streamline_density"] = vm_df["streamline_count"] / vm_df["voxel_count"]
        vm_df["meridian"] = "VM"

        # --- Keep original LHM, RHM, LVM, UVM ---
        df = df[df["meridian"].isin(["LHM", "RHM", "LVM", "UVM"])].copy()

        # --- Combine everything ---
        df = pd.concat([df, hm_df, vm_df], ignore_index=True)

        valid_order = ["LHM", "RHM", "HM", "LVM", "UVM", "VM"]
        palette = {
            "LHM": "#a1d99b",  # light green
            "RHM": "#31a354",  # green
            "HM":  "#2ca02c",  # combined HM
            "LVM": "#fb1209",  # red
            "UVM": "#3182bd",  # blue
            "VM":  "#9467bd",  # purple (match other mode)
        }

        out_subdir = out_dir / "plots_HM_RHM_LHM_VM_LVM_UVM"
        collapse = False  # already aggregated where needed


    elif mode == "hm_vm_uro_ulo_lro_llo":
        # Combine classic horizontal & vertical meridians, add obliques (URO, ULO, LRO, LLO)
        # Collapse LHM+RHM → HM, LVM+UVM → VM
        df["meridian"] = df["meridian"].replace({
            "LHM": "HM", "RHM": "HM",
            "LVM": "VM", "UVM": "VM"
        })

        valid_order = ["HM", "VM", "URO", "ULO", "LRO", "LLO"]
        palette = {
            "HM":  "#2ca02c",   # green
            "VM":  "#9467bd",   # purple
            "URO": "#ff9933",   # orange (upper right)
            "ULO": "#ffcc66",   # light orange (upper left)
            "LRO": "#FF007F",   # fuchsia
            "LLO": "#b64b75",   # dark blue (lower left)
        }

        out_subdir = out_dir / "plots_HM_VM_URO_ULO_LRO_LLO"
        collapse = True  # to average HM/VM across hemispheres
        df["streamline_density"] = df["streamline_count"] / df["voxel_count"]


    elif mode == "hm_vm_lo_uo":
        # Collapse horizontal and vertical meridians
        df["meridian"] = df["meridian"].replace({
            "LHM": "HM", "RHM": "HM",
            "LVM": "VM", "UVM": "VM",
            "URO": "UO", "ULO": "UO",
            "LRO": "LO", "LLO": "LO"
        })

        # Define the order and colors
        valid_order = ["HM", "VM", "UO", "LO"]
        palette = {
            "HM": "#2ca02c",     # green
            "VM": "#9467bd",     # purple
            "UO": "#ffcc66",     # orange (upper oblique combined)
            "LO": "#FF007F",     # darker pink (lower oblique combined)
        }

        out_subdir = out_dir / "plots_HM_VM_LO_UO"
        collapse = True  # Collapse now makes sense: URO+ULO and LRO+LLO merged

        df["streamline_density"] = df["streamline_count"] / df["voxel_count"]

    elif mode == "rhm_lhm_lvm_uvm_uro_ulo_lro_llo":
        # Keep all hemispheric meridians and obliques separate
        valid_order = ["RHM", "LHM", "LVM", "UVM", "URO", "ULO", "LRO", "LLO"]

        palette = {
            "RHM": "#31a354",   # dark green
            "LHM": "#a1d99b",   # light green
            "LVM": "#fb1209",   # red
            "UVM": "#3182bd",   # blue
            "URO": "#ff9933",   # orange
            "ULO": "#ffcc66",   # light orange
            "LRO": "#FF007F",   # fuchsia
            "LLO": "#b64b75"    # darker pink
        }

        out_subdir = out_dir / "plots_RHM_LHM_LVM_UVM_URO_ULO_LRO_LLO"
        collapse = False
        df["streamline_density"] = df["streamline_count"] / df["voxel_count"]

    elif mode == "hm_vm_om":
        # Collapse horizontal (HM), vertical (VM), and oblique (OM) meridians
        # OM merges URO, ULO, LRO, LLO

        # Replace original labels with HM, VM, OM
        df["meridian"] = df["meridian"].replace({
            "LHM": "HM", "RHM": "HM",
            "LVM": "VM", "UVM": "VM",
            "URO": "OM", "ULO": "OM", "LRO": "OM", "LLO": "OM"
        })

        # Aggregate per subject and meridian
        df["total_curvature"] = df["mean_curvature"] * df["voxel_count"]

        agg = (
            df.groupby(["subject", "meridian", "eccentricity_bin"], as_index=False)
            .agg({
                "voxel_count": "sum",
                "streamline_count": "sum",
                "total_curvature": "sum"
            })
        )

        agg["mean_curvature"] = agg["total_curvature"] / agg["voxel_count"]
        agg["streamline_density"] = agg["streamline_count"] / agg["voxel_count"]
        df = agg.copy()

        valid_order = ["HM", "VM", "OM"]
        palette = {
            "HM": "#2ca02c",  # green
            "VM": "#9467bd",  # purple
            "OM": "#ff7f0e"   # orange for obliques
        }

        out_subdir = out_dir / "plots_HM_VM_OM"
        collapse = True

    elif mode == "hm_lvm_uvm_lom_uom":
        # Collapse LHM + RHM → HM
        # Merge lower obliques (LRO + LLO) → LOM
        # Merge upper obliques (URO + ULO) → UOM

        df["meridian"] = df["meridian"].replace({
            "LHM": "HM", "RHM": "HM",
            "LRO": "LOM", "LLO": "LOM",
            "URO": "UOM", "ULO": "UOM"
        })

        # Compute weighted total curvature
        df["total_curvature"] = df["mean_curvature"] * df["voxel_count"]

        # Aggregate across subject, meridian, eccentricity bin
        agg = (
            df.groupby(["subject", "meridian", "eccentricity_bin"], as_index=False)
            .agg({
                "voxel_count": "sum",
                "streamline_count": "sum",
                "total_curvature": "sum"
            })
        )
        agg["mean_curvature"] = agg["total_curvature"] / agg["voxel_count"]
        agg["streamline_density"] = agg["streamline_count"] / agg["voxel_count"]
        df = agg.copy()

        valid_order = ["HM", "LVM", "UVM", "LOM", "UOM"]
        palette = {
            "HM":  "#2ca02c",   # green
            "LVM": "#d62728",   # red
            "UVM": "#1f77b4",   # blue
            "LOM": "#ff7f0e",   # orange (lower oblique)
            "UOM": "#ffbb78",   # light orange (upper oblique)
        }

        out_subdir = out_dir / "plots_HM_LVM_UVM_LOM_UOM"
        collapse = True




    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Keep only desired meridians in valid_order
    df = df[df["meridian"].isin(valid_order)].copy()
    df["meridian"] = pd.Categorical(df["meridian"], categories=valid_order, ordered=True)

    # Drop any rows with NaNs in essential columns
    df = df.dropna(subset=["mean_curvature", "streamline_density", "meridian"])

    return df, palette, valid_order, out_subdir


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _ensure_palette(palette):
    # Default colors if none provided
    default = {
        "RHM": "#31a354",   # green
        "LHM": "#a1d99b",   # light green
        "UVM": "#3182bd",   # blue
        "LVM": "#fb1209",   # red
    }
    if palette is None:
        return default
    # fill any missing keys with defaults
    out = default.copy()
    out.update(palette)
    return out

def _axis_limits(series, pad=0.05):
    s = np.asarray(series.dropna().values)
    lo, hi = s.min(), s.max()
    span = hi - lo if hi > lo else (abs(hi) + 1e-6)
    return lo - pad * span, hi + pad * span


def get_meridian_map():
    """parcel_id -> meridian label"""
    m = {}

    # --- Vertical meridians ---
    for i in range(1, 11):  # 0–15°
        m[i] = "UVM"
    for i in range(41, 51):  # 165–180°
        m[i] = "LVM"

    # --- Horizontal meridians ---
    for i in range(21, 30, 2):  # 75–105°, left hemisphere (odd)
        m[i] = "LHM"
    for i in range(22, 31, 2):  # 75–105°, right hemisphere (even)
        m[i] = "RHM"

    # --- Oblique meridians ---
    # Upper Obliques: 30–60°
    for i in range(11, 20, 2):  # left hemisphere (odd)
        m[i] = "ULO"
    for i in range(12, 21, 2):  # right hemisphere (even)
        m[i] = "URO"

    # Lower Obliques: 120–150°
    for i in range(31, 40, 2):  # left hemisphere (odd)
        m[i] = "LLO"
    for i in range(32, 41, 2):  # right hemisphere (even)
        m[i] = "LRO"

    return m


def get_ecc_map():
    """parcel_id -> eccentricity bin {0to2,2to4,4to6,6to8,8to90}
    Matches your JSON scheme where eccentricity alternates within each polar-angle block.
    """
    ecc_bins = {
        1: "0to2",  2: "0to2",
        3: "2to4",  4: "2to4",
        5: "4to6",  6: "4to6",
        7: "6to8",  8: "6to8",
        9: "8to90", 0: "8to90"
    }
    return {pid: ecc_bins[pid % 10] for pid in range(1, 51)}


def prepare_subject_means(df,meridians=None):
    """
    Expects df with columns: parcel_id, subject, mean_curvature, streamline_density
    If 'streamline_density' is missing but ('streamline_count','voxel_count') exist,
    it will compute density = streamline_count / voxel_count.
    """
    if meridians is None:
        meridians = ["LHM","RHM","UVM","LVM"]
    ecc_order = ["0to2","2to4","4to6","6to8","8to90"]

    if "streamline_density" not in df.columns and {"streamline_count","voxel_count"} <= set(df.columns):
        df = df.copy()
        df["streamline_density"] = df["streamline_count"] / df["voxel_count"]

    df = df.copy()
    if "meridian" not in df.columns:
        df["meridian"] = df["parcel_id"].map(get_meridian_map())
    if "eccentricity_bin" not in df.columns:
        df["eccentricity_bin"] = df["parcel_id"].map(get_ecc_map())
    df = df[df["meridian"].isin(meridians)]
    df["eccentricity_bin"] = pd.Categorical(df["eccentricity_bin"], categories=ecc_order, ordered=True)

    # Subject-level means inside each (subject, meridian, ecc) cell (avoids unequal parcel weighting)
    df["total_curvature"] = df["mean_curvature"] * df["voxel_count"]
    
    group_cols = ["subject", "meridian", "eccentricity_bin"]
    df = safe_groupby_keys(df, group_cols)
    subj_means = df.groupby(group_cols, as_index=False).agg({
            "voxel_count": "sum",
            "streamline_count": "sum",
            "total_curvature": "sum"
        })
    
    subj_means["mean_curvature"] = subj_means["total_curvature"] / subj_means["voxel_count"]
    subj_means["streamline_density"] = subj_means["streamline_count"] / subj_means["voxel_count"]

    return subj_means

from matplotlib.patches import Patch

from matplotlib.patches import Patch
import numpy as np
import matplotlib.pyplot as plt

def plot_comprehensive_box_curvature(subj_means, out_path, meridians=None, palette=None):
    ecc_order = ["0to2", "2to4", "4to6", "6to8", "8to90"]
    ecc_labels = {
        "0to2": "0–2°",
        "2to4": "2–4°",
        "4to6": "4–6°",
        "6to8": "6–8°",
        "8to90": "8–90°"
    }

    if meridians is None:
        meridians = ["LHM", "RHM", "LVM", "UVM"]
    if palette is None:
        palette = {
            "LHM": "#a1d99b",
            "RHM": "#31a354",
            "LVM": "#fb1209",
            "UVM": "#3182bd"
        }

    gradients = {m: make_gradient(palette[m], len(ecc_order)) for m in meridians}

    fig, ax = plt.subplots(figsize=(12, 7))
    group_positions = np.arange(len(meridians)) * 6.0
    box_width = 0.6

    for m_idx, mer in enumerate(meridians):
        sub = subj_means[subj_means["meridian"] == mer]
        positions = group_positions[m_idx] + np.arange(len(ecc_order))

        for e_idx, ecc in enumerate(ecc_order):
            vals = sub[sub["eccentricity_bin"] == ecc]["mean_curvature"].dropna().values
            if len(vals) == 0:
                continue

            bp = ax.boxplot(
                [vals],
                positions=[positions[e_idx]],
                widths=box_width,
                patch_artist=True,
                showfliers=False
            )

            for patch in bp["boxes"]:
                patch.set_facecolor(gradients[mer][e_idx])
                patch.set_edgecolor("black")

            for element in ["whiskers", "caps", "medians"]:
                for line in bp[element]:
                    line.set_color("black")

        mer_handles = [
            Patch(
                facecolor=gradients[mer][i],
                edgecolor="black",
                label=ecc_labels[ecc_order[i]]
            )
            for i in range(len(ecc_order))
        ]

        q25 = sub["mean_curvature"].quantile(0.25)
        q75 = sub["mean_curvature"].quantile(0.75)
        center = (q25 + q75) / 2 if not (np.isnan(q25) or np.isnan(q75)) else 0

        if center > 0:
            y_anchor = -0.20
            loc = "upper center"
        else:
            y_anchor = 1.18
            loc = "lower center"

        denom = (group_positions[-1] + 4.0) if len(group_positions) > 1 else 4.0
        x_anchor = (group_positions[m_idx] + 2.0) / denom

        leg = ax.legend(
            handles=mer_handles,
            title=mer,
            fontsize=8,
            title_fontsize=9,
            frameon=True,
            loc=loc,
            bbox_to_anchor=(x_anchor, y_anchor),
            ncol=1
        )
        ax.add_artist(leg)

    ax.set_xticks(group_positions + 2.0)
    ax.set_xticklabels(meridians)
    ax.axhline(0, linewidth=1, color="black")
    ax.set_ylabel("Mean curvature", fontsize=16)

    # Title matched to Nature-style font size
    ax.set_title(
        "Mean curvature distributions by meridian with eccentricity gradients",
        fontsize=16,
        pad=14
    )

    nature_style_plot(
        ax,
        ymin=-0.6,
        ymax=0.35,
        fontsize=16,
        y_decimals=2,
        yticks=[-0.6, 0, 0.35],
        n_yticks=None,
        format_xticklabels=False,
        add_origin_padding=False,
    )

    # Add semantic labels for sign of curvature
    ax.text(
        -0.08, 0.18, "Gyrus",
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=14
    )
    ax.text(
        -0.08, 0.82, "Sulcus",
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=14
    )

    plt.tight_layout()
    save_figure(out_path, dpi=300)
    plt.close()

from matplotlib.patches import Patch

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Ellipse
import numpy as np
import pandas as pd

def _axis_limits(series, method="minmax", pad=0.05, q=0.01):
    """
    Compute axis limits from data with optional padding.

    Parameters
    ----------
    series : pandas.Series
        Data values to base limits on.
    method : str, {"minmax","quantile","iqr"}
        How to compute limits.
    pad : float
        Fractional padding to add around the range.
    q : float
        Quantile cutoff when method="quantile".
    """
    if method == "quantile":
        lo, hi = series.quantile([q, 1-q]).values
    elif method == "iqr":
        q1, q3 = series.quantile([0.25, 0.75]).values
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    else:  # default minmax
        lo, hi = series.min(), series.max()

    rng = hi - lo
    return lo - pad * rng, hi + pad * rng

def plot_comprehensive_centroid_scatter(
    subj_means,
    out_path,
    meridians=None,
    palette=None,
    title_suffix="",
    save_pdf=False,
    y_lim=None,
    y_from_data=True,
    y_method="minmax",
    y_q=0.01,
    y_pad=0.05,
    x_lim=None,
    x_from_data=False,
    x_method="minmax",
    x_q=0.01,
    x_pad=0.05
):
    """
    Plot centroids of (density vs curvature) per meridian and eccentricity bin,
    with SEM ellipses, legend ordering, and optional PDF export.

    Parameters
    ----------
    subj_means : pd.DataFrame
        Must contain ['meridian','eccentricity_bin','mean_curvature','streamline_density','subject'].
    out_path : Path
        Path to save PNG figure.
    meridians : list, optional
        Order of meridians to plot. Defaults to ["LHM","RHM","LVM","UVM"].
    palette : dict, optional
        Dict mapping meridian→base color. Defaults to green/red/blue families.
    title_suffix : str
        Extra text appended to plot title.
    save_pdf : bool
        If True, also saves a PDF version alongside the PNG.
    y_lim, x_lim : tuple, optional
        Axis limits. If None, can be inferred from data.
    *_from_data : bool
        Whether to derive axis limits from data.
    *_method : str
        How to compute limits ("minmax" or "quantile").
    *_q : float
        Quantile cutoff when method="quantile".
    *_pad : float
        Padding fraction around limits.
    """
    ecc_order = ["0to2", "2to4", "4to6", "6to8", "8to90"]
    ecc_labels = {
        "0to2": "0–2°", "2to4": "2–4°", "4to6": "4–6°",
        "6to8": "6–8°", "8to90": "8–90°"
    }

    if meridians is None:
        meridians = ["LHM", "RHM", "LVM", "UVM"]
    if palette is None:
        palette = {"LHM": "#a1d99b", "RHM": "#31a354",
                   "LVM": "#fb1209", "UVM": "#3182bd"}

    # Generate gradients for each meridian
    gradients = {m: make_gradient(palette[m], len(ecc_order)) for m in meridians}

    # Compute subject-level centroids
    subj_centroids = (
        subj_means.groupby(["subject","meridian","eccentricity_bin"], as_index=False)
        .agg(cx=("mean_curvature","mean"),
             cy=("streamline_density","mean"))
    )

    # Group-level centroid and SEM
    centroids = (
        subj_centroids.groupby(["meridian","eccentricity_bin"])
        .agg(cx=("cx","mean"), cy=("cy","mean"),
             cx_sem=("cx","sem"), cy_sem=("cy","sem"))
        .reset_index()
    )

    plt.figure(figsize=(8,6))
    idx_map = {b:i for i,b in enumerate(ecc_order)}

    for mer in meridians:
        sub = centroids[centroids["meridian"]==mer].sort_values("eccentricity_bin")
        for _, row in sub.iterrows():
            i = idx_map[str(row["eccentricity_bin"])]
            color = gradients[mer][i]

            # centroid point
            plt.scatter(row["cx"], row["cy"], color=color, s=90,
                        edgecolor="black", linewidth=0.6, zorder=3)

            # SEM ellipse
            if not (np.isnan(row["cx_sem"]) or np.isnan(row["cy_sem"])):
                ellipse = Ellipse(
                    (row["cx"], row["cy"]),
                    width=row["cx_sem"]*2,  # ±SEM along x
                    height=row["cy_sem"]*2, # ±SEM along y
                    facecolor=color, edgecolor="none",
                    alpha=0.2, zorder=2
                )
                plt.gca().add_patch(ellipse)

            # eccentricity label
            ecc_label = ecc_labels[str(row["eccentricity_bin"])]
            #plt.annotate(ecc_label, (row["cx"], row["cy"]),
            #             textcoords="offset points", xytext=(3,3), fontsize=8)


            # eccentricity label
            label_txt = f"${ecc_label}_{{{mer}}}$"

            plt.annotate(
                label_txt,
                (row["cx"], row["cy"]),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=5,
                ha="left",
                va="bottom"
                )   

    # Reference axes
    plt.axhline(0, linestyle="--", linewidth=1, color="gray")
    plt.axvline(0, linestyle="--", linewidth=1, color="gray")

    # Legend (ordered)
    label_map = {
        "RHM": "RHM (green)", "LHM": "LHM (light green)", "HM": "HM (merged)",
        "LVM": "LVM (red)", "UVM": "UVM (blue)", "VM": "VM (merged)"
    }
    legend_items = [
        Patch(facecolor=gradients[m][-1], edgecolor="black",
              label=label_map.get(m,m))
        for m in meridians if m in gradients
    ]
    plt.legend(handles=legend_items, title="Meridian (color family)", loc="best")

    # Axis scaling
    if y_lim is None and y_from_data:
        y_lim = _axis_limits(
            subj_means["streamline_density"], method=y_method, pad=y_pad, q=y_q
        )
    if x_lim is None and x_from_data:
        x_lim = _axis_limits(
            subj_means["mean_curvature"], method=x_method, pad=x_pad, q=x_q
        )
    if y_lim is not None:
        plt.ylim(y_lim)
    if x_lim is not None:
        plt.xlim(x_lim)

    plt.xlabel("Mean curvature")
    plt.ylabel("Streamline density ")
    plt.title("Centroids of (density vs curvature) by meridian with eccentricity gradients"
              + title_suffix)
    plt.tight_layout()

    # Save outputs
    plt.savefig(out_path, dpi=300)
    if save_pdf:
        pdf_path = out_path.with_suffix(".pdf")
        plt.savefig(pdf_path)
    plt.close()
    return x_lim, y_lim



def plot_meridian_centroids(subj_means, out_path, meridians=None, palette=None):
    """
    Plots one centroid per meridian, collapsed across all eccentricities and subjects.
    Expects input at (subject, meridian) level.
    Allows for external control of meridian list and color palette.
    """
    if meridians is None:
        meridians = ["LHM", "RHM", "LVM", "UVM"]
    if palette is None:
        palette = {
            "LHM": "#a1d99b",  # light green
            "RHM": "#31a354",  # green
            "LVM": "#e34a33",  # red
            "UVM": "#2b8cbe",  # blue
        }
    color_map = palette
    print(color_map)
    print(meridians)
    # Compute overall centroid per meridian (mean of all subjects)
    centroids = (
        subj_means.groupby("meridian")[["mean_curvature", "streamline_density"]]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 6))
    for _, row in centroids.iterrows():
        mer = row["meridian"]
        plt.scatter(row["mean_curvature"], row["streamline_density"],
                    color=color_map[mer], s=150, edgecolor='black', label=mer)

    plt.axhline(0, linestyle="--", linewidth=1, color="gray")
    plt.axvline(0, linestyle="--", linewidth=1, color="gray")
    plt.xlabel("Mean curvature")
    plt.ylabel("Streamline density ")
    plt.title("Centroids of (density vs curvature) by meridian (all eccentricities)")

    legend_handles = [
        Patch(facecolor=color_map[m], edgecolor='black', label=m)
        for m in meridians
    ]
    plt.legend(handles=legend_handles, title="Meridian", loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Ellipse
import matplotlib.transforms as transforms
from scipy.stats import chi2


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Ellipse
import matplotlib.transforms as transforms
from scipy.stats import chi2
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.mixture import GaussianMixture

def plot_meridian_modelling(subj_means, out_path, meridians=None, palette=None, 
                            spread_mode='kde', conf_level=0.95):
    """
    Advanced plotting with 7 modes: 
    Visual: 'kde', 'conf_ellipse', 'error_bar'
    Statistical: 'OLS', 'GLM', 'GMM', 'LME'
    """
    if meridians is None:
        meridians = ["LHM", "RHM", "LVM", "UVM"]
    if palette is None:
        palette = {"LHM": "#a1d99b", "RHM": "#31a354", "LVM": "#e34a33", "UVM": "#2b8cbe"}

    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    
    subj_col = "subject" if "subject" in subj_means.columns else "subject_id"

    # Grid for smooth line plotting (OLS, GLM, LME)
    x_range = np.linspace(subj_means["mean_curvature"].min(), 
                          subj_means["mean_curvature"].max(), 100)

    for mer in meridians:
        subset = subj_means[subj_means["meridian"] == mer].copy()
        if subset.empty: continue
        color = palette[mer]
        
        # --- 1. STATISTICAL MODELING MODES ---
        
        if spread_mode == 'OLS':
            # Linear Regression with 95% CI shading
            sns.regplot(data=subset, x="mean_curvature", y="streamline_density", 
                        scatter=False, color=color, ax=ax, label=f"{mer} OLS",
                        line_kws={'alpha': 0.8, 'linewidth': 2})

        elif spread_mode == 'GLM':
            # Gamma GLM with Log-link: strictly positive, handles heteroscedasticity
            # Adding small epsilon to density to avoid log(0) issues if any
            subset['density_eps'] = subset['streamline_density'] + 1e-6
            model = smf.glm("density_eps ~ mean_curvature", data=subset, 
                            family=sm.families.Gamma(link=sm.families.links.log())).fit()
            
            pred_df = pd.DataFrame({"mean_curvature": x_range})
            predictions = model.get_prediction(pred_df).summary_frame()
            
            plt.plot(x_range, predictions['mean'], color=color, linewidth=2)
            plt.fill_between(x_range, predictions['mean_ci_lower'], 
                             predictions['mean_ci_upper'], color=color, alpha=0.2)

        elif spread_mode == 'LME':
            # Linear Mixed-Effects: Accounts for subject-level repeated measures
            # Fixed effect: Curvature; Random effect: Subject-specific intercepts
            model = smf.mixedlm("streamline_density ~ mean_curvature", subset, 
                                groups=subset[subj_col]).fit()
            
            # Plot the Global (Fixed) Effect line
            y_pred = model.params['Intercept'] + model.params['mean_curvature'] * x_range
            plt.plot(x_range, y_pred, color=color, linewidth=3, label=f"{mer} Global")

        elif spread_mode == 'GMM':
            # Gaussian Mixture Model: Finds sub-clusters (e.g. 2 clusters per meridian)
            gmm = GaussianMixture(n_components=2, random_state=42).fit(
                subset[["mean_curvature", "streamline_density"]]
            )
            for i in range(gmm.n_components):
                draw_gmm_ellipse(gmm, i, ax, color=color, alpha=0.15)

        # --- 2. BASIC VISUAL MODES (from previous iterations) ---
        
        elif spread_mode == 'kde':
            sns.kdeplot(data=subset, x="mean_curvature", y="streamline_density", 
                        fill=True, color=color, alpha=0.2, levels=5, clip=((None,None),(0,None)))

        elif spread_mode == 'conf_ellipse':
            n_std = np.sqrt(chi2.ppf(conf_level, df=2))
            draw_simple_ellipse(subset["mean_curvature"], subset["streamline_density"], 
                                ax, n_std=n_std, facecolor=color, alpha=0.2)

    # --- CENTROID PLOTTING (The "anchors") ---
    centroids = subj_means.groupby("meridian")[["mean_curvature", "streamline_density"]].mean().reset_index()
    for _, row in centroids.iterrows():
        mer = row["meridian"]
        if mer in palette:
            plt.scatter(row["mean_curvature"], row["streamline_density"],
                        color=palette[mer], s=180, edgecolor='black', zorder=20)

    # Final Polish
    plt.axhline(0, linestyle="-", linewidth=1.5, color="black", alpha=0.7)
    plt.ylim(0, plt.ylim()[1] * 1.1) # Ensure Y starts at 0 and adds headroom
    plt.xlabel("Mean Curvature")
    plt.ylabel("Streamline Density")
    plt.title(f"Meridian Analysis: {spread_mode} Modelling")
    
    legend_handles = [Patch(facecolor=palette[m], edgecolor='black', label=m) for m in meridians]
    plt.legend(handles=legend_handles, title="Meridians", loc="upper right")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    #plt.show()

# --- HELPER FUNCTIONS ---

def draw_simple_ellipse(x, y, ax, n_std=2.0, **kwargs):
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs)
    scale_x, scale_y = np.sqrt(cov[0, 0]) * n_std, np.sqrt(cov[1, 1]) * n_std
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(np.mean(x), np.mean(y))
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def draw_gmm_ellipse(gmm, i, ax, color, alpha):
    """Draws an ellipse representing a GMM component."""
    covariances = gmm.covariances_[i][:2, :2]
    v, w = np.linalg.eigh(covariances)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    ell = Ellipse(gmm.means_[i, :2], v[0], v[1], 180 + angle, color=color, alpha=alpha)
    ax.add_artist(ell)


from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap, to_rgb

def eccentricity_bin_to_numeric(val):
    """
    Convert eccentricity bin labels to numeric midpoints for smoothing.
    """
    mapping = {
        "0to2": 1.0,
        "2to4": 3.0,
        "4to6": 5.0,
        "6to8": 7.0,
        "8to90": 49.0,   # broad outer bin; change if you prefer another representative value
    }
    if pd.isna(val):
        return np.nan
    return mapping.get(str(val), np.nan)


def make_white_to_color_cmap(base_color, name="custom_meridian_cmap"):
    """
    Continuous colormap from white to the given base color.
    """
    r, g, b = to_rgb(base_color)
    colors = [
        (1.0, 1.0, 1.0),
        (0.92 + 0.08*r, 0.92 + 0.08*g, 0.92 + 0.08*b),
        (0.75 + 0.25*r, 0.75 + 0.25*g, 0.75 + 0.25*b),
        (r, g, b),
    ]
    return LinearSegmentedColormap.from_list(name, colors)


def compute_kde_eccentricity_surface(
    x, y, ecc,
    xmin, xmax, ymin, ymax,
    gridsize=150,
    bw_method=None,
    density_quantile_mask=None
):
    """
    Compute:
      1) standard KDE density over (x, y)
      2) kernel-smoothed local mean eccentricity over the same grid

    Returns:
      xx, yy, density_grid, ecc_grid
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ecc = np.asarray(ecc, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(ecc)
    x, y, ecc = x[valid], y[valid], ecc[valid]

    if len(x) < 5:
        return None, None, None, None

    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, gridsize),
        np.linspace(ymin, ymax, gridsize)
    )
    grid = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])

    try:
        kde_den = gaussian_kde(values, bw_method=bw_method)
        density = kde_den(grid)

        # Weighted KDE for numerator of local mean eccentricity
        kde_num = gaussian_kde(values, weights=ecc, bw_method=bw_method)
        numerator = kde_num(grid)

    except np.linalg.LinAlgError:
        # Happens if covariance is singular / points too degenerate
        return None, None, None, None

    density_grid = density.reshape(xx.shape)
    numerator_grid = numerator.reshape(xx.shape)

    with np.errstate(divide='ignore', invalid='ignore'):
        ecc_grid = numerator_grid / density_grid

    # Mask invalid / extremely sparse regions
    mask = ~np.isfinite(ecc_grid) | ~np.isfinite(density_grid) | (density_grid <= 0)
    if density_quantile_mask is not None:
        thr = np.quantile(density_grid[np.isfinite(density_grid)], density_quantile_mask)
        mask |= density_grid < thr

    ecc_grid = np.ma.array(ecc_grid, mask=mask)
    density_grid = np.ma.array(density_grid, mask=mask)

    return xx, yy, density_grid, ecc_grid

def compute_local_mean_surface(x, y, z, xmin, xmax, ymin, ymax, gridsize=200, bw_method=None):
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, gridsize),
        np.linspace(ymin, ymax, gridsize)
    )
    grid = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])

    kde_den = gaussian_kde(values, bw_method=bw_method)
    den = kde_den(grid)

    kde_num = gaussian_kde(values, weights=z, bw_method=bw_method)
    num = kde_num(grid)

    zhat = num / den
    zhat = zhat.reshape(xx.shape)
    den = den.reshape(xx.shape)

    return xx, yy, zhat, den

from matplotlib.colors import LinearSegmentedColormap, to_rgb
import numpy as np

def make_meridian_eccentricity_cmap(base_color, name="meridian_ecc_cmap"):
    """
    Create a smooth light-to-base-color colormap for eccentricity.
    Low eccentricity = very light tint
    High eccentricity = full meridian color
    """
    r, g, b = to_rgb(base_color)

    colors = [
        (1.00, 1.00, 1.00),                                # near white
        (0.92 + 0.08*r, 0.92 + 0.08*g, 0.92 + 0.08*b),    # very light tint
        (0.75 + 0.25*r, 0.75 + 0.25*g, 0.75 + 0.25*b),    # medium tint
        (r, g, b),                                        # full meridian color
    ]
    return LinearSegmentedColormap.from_list(name, colors)

def plot_meridian_centroids_x(
    subj_means,
    out_path,
    meridians=None,
    palette=None,
    spread_mode='kde',
    kde_thr=0.1,
    conf_level=0.95,
    gridsize=180,
    bw_method=None,
    ecc_vmin=0.0,
    ecc_vmax=8.0,
    kde_ecc_alpha=0.55,
):
    if meridians is None:
        meridians = ["LHM", "RHM", "LVM", "UVM"]
    if palette is None:
        palette = {"LHM": "#a1d99b", "RHM": "#31a354", "LVM": "#e34a33", "UVM": "#2b8cbe"}

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Fixed plotting limits to match your current style
    xmin = -0.6
    xmax = 0.4
    ymin = 0.0
    ymax = 1.0

    # Scaling factor for ellipse
    if spread_mode == 'conf_ellipse':
        n_std = np.sqrt(chi2.ppf(conf_level, df=2))

    # Precompute numeric eccentricity if needed
    subj_means = subj_means.copy()
    if "eccentricity_bin" in subj_means.columns:
        subj_means["ecc_numeric"] = subj_means["eccentricity_bin"].map(eccentricity_bin_to_numeric)
    else:
        subj_means["ecc_numeric"] = np.nan

    for mer in meridians:
        subset = subj_means[subj_means["meridian"] == mer].copy()
        if subset.empty:
            continue

        color = palette[mer]
        x = subset["mean_curvature"].to_numpy(dtype=float)
        y = subset["streamline_density"].to_numpy(dtype=float)

        if spread_mode == 'kde':
            sns.kdeplot(
                x=x, y=y,
                fill=True,
                color=color,
                alpha=0.3,
                levels=5,
                thresh=kde_thr,
                clip=((xmin, xmax), (ymin, ymax))
            )

        elif spread_mode == 'kde_eccentricity':
            ecc = subset["ecc_numeric"].to_numpy(dtype=float)

            xx, yy, ecc_map, den_map = compute_local_mean_surface(
                x, y, ecc,
                xmin=xmin, xmax=xmax,
                ymin=ymin, ymax=ymax,
                gridsize=220,
                bw_method=0.3
            )

            if xx is not None:
                dmax = np.nanmax(den_map)
                abs_thr = kde_thr * dmax

                mask = ~np.isfinite(den_map) | ~np.isfinite(ecc_map) | (den_map < abs_thr)
                ecc_map = np.ma.array(ecc_map, mask=mask)
                den_map = np.ma.array(den_map, mask=mask)
                ecc_cmap = make_meridian_eccentricity_cmap(color, name=f"{mer}_ecc_cmap")
                cf = ax.contourf(
                    xx, yy, ecc_map,
                    levels=np.linspace(0, 10, 200),
                    cmap=ecc_cmap,
                    alpha=0.8,
                    zorder=1
                )

                contour_levels = np.linspace(abs_thr, dmax, 6)[1:]  # 5 contour levels
                ax.contour(
                    xx, yy, den_map,
                    levels=contour_levels,
                    colors=[color],
                    linewidths=1.2,
                    alpha=0.9,
                    zorder=2
                )
                    
        elif spread_mode == 'error_bar':
            y_mean = y.mean()
            y_err = y.std()
            lower_err = min(y_err, y_mean)

            plt.errorbar(
                x.mean(), y_mean,
                xerr=x.std(),
                yerr=[[lower_err], [y_err]],
                fmt='none',
                ecolor=color,
                elinewidth=2,
                capsize=3,
                alpha=0.6
            )

        elif spread_mode == 'conf_ellipse':
            draw_ellipse(x, y, ax, n_std=n_std, facecolor=color, alpha=0.2, zorder=1)
            draw_ellipse(
                x, y, ax,
                n_std=n_std,
                edgecolor=color,
                facecolor='none',
                linestyle='-',
                linewidth=1.2,
                alpha=0.6,
                zorder=2
            )

    if spread_mode == 'kde_eccentricity' and 'cf' in locals():
        cbar = plt.colorbar(cf, ax=ax, pad=0.02)
        cbar.set_label("Eccentricity (deg)")

    # --- CENTROID PLOTTING ---
    centroids = (
        subj_means
        .groupby("meridian")[["mean_curvature", "streamline_density"]]
        .mean()
        .reset_index()
    )

    for _, row in centroids.iterrows():
        mer = row["meridian"]
        if mer in palette:
            plt.scatter(
                row["mean_curvature"],
                row["streamline_density"],
                color=palette[mer],
                s=150,
                edgecolor='black',
                zorder=10
            )

    ax = nature_style_plot(
        ax,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        n_xticks=3,
        n_yticks=3,
        x_decimals=2,
        y_decimals=2,
        fontsize=16,
        add_origin_padding=True,
        pad_fraction=0.05,
        xticks=[xmin, 0, xmax],
        yticks=[ymin, ymax/2, ymax]
    )

    plt.axvline(0, linestyle="--", linewidth=1, color="gray", alpha=0.5)
    plt.ylim(0, plt.ylim()[1])

    plt.xlabel("Mean curvature")
    plt.ylabel("Streamline density")

    legend_handles = [
        Patch(facecolor=palette[m], edgecolor='none', label=m)
        for m in meridians if m in palette
    ]
    plt.legend(handles=legend_handles, title="Meridian", loc="best", frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def draw_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """Mathematical helper to project the covariance matrix as an ellipse."""
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(np.mean(x), np.mean(y))
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Helper function to calculate and draw a covariance-based confidence ellipse.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to get the eigenvalues of this 2-D dataset
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Standard deviation scale
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

# ----------------------------
# Axis limit utilities
# ----------------------------
def _axis_limits(series, method="minmax", pad=0.05, q=0.01):
    """
    Compute (lo, hi) for an axis from a pandas Series.
      method: "minmax" (use true min/max) or "quantile" (trim extremes)
      pad   : fractional padding added to each side
      q     : lower quantile for trimming when method="quantile" (upper=1-q)
    """
    s = np.asarray(series.dropna().values)
    if s.size == 0:
        return None
    if method == "quantile":
        lo, hi = np.quantile(s, [q, 1.0 - q])
    else:
        lo, hi = np.min(s), np.max(s)

    # handle degenerate case
    if np.isclose(lo, hi):
        span = 1.0 if hi == 0 else abs(hi) * 0.1
        lo, hi = lo - span, hi + span

    span = hi - lo
    lo -= pad * span
    hi += pad * span
    return (lo, hi)

def _ensure_palette(palette):
    # Defaults: RHM=green, LHM=light green, UVM=blue, LVM=red
    default = {"RHM": "#31a354", "LHM": "#a1d99b", "UVM": "#3182bd", "LVM": "#fb1209"}
    if palette is None:
        return default
    out = default.copy(); out.update(palette)
    return out

# --------------------------------------
# Boxplots: curvature by meridian × ecc
# --------------------------------------
def plot_box_curvature_by_meridian_per_ecc(
    subj_means,
    out_dir,
    palette=None,
    meridians=None,
    y_lim=None,
    y_from_data=True,
    y_method="minmax",  # or "quantile"
    y_q=0.01,
    y_pad=0.05,
):
    """
    One figure per eccentricity. If y_lim is None and y_from_data=True,
    uses global min/max (or quantiles) across *all* eccentricities to fix y-axis.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if meridians is None:
        meridians = ["LHM","RHM","LVM","UVM"]
    ecc_order = ["0to2","2to4","4to6","6to8","8to90"]
    if palette is None:
        palette = _ensure_palette(palette)

    # Compute a single y-limit from all data unless user provided y_lim
    if y_lim is None and y_from_data:
        y_lim = _axis_limits(
            subj_means["mean_curvature"], method=y_method, pad=y_pad, q=y_q
        )

    for ecc in ecc_order:
        sub = subj_means[subj_means["eccentricity_bin"] == ecc]
        data = [sub[sub["meridian"]==m]["mean_curvature"].dropna().values for m in meridians]

        plt.figure(figsize=(7,5))
        bp = plt.boxplot(data, labels=meridians, showfliers=False, patch_artist=True)

        for box, m in zip(bp['boxes'], meridians):
            box.set_facecolor(palette[m]); box.set_edgecolor("black")
        for elem in ['whiskers','caps','medians']:
            for line in bp[elem]:
                line.set_color("black")

        handles = [Patch(facecolor=palette[m], edgecolor="black", label=m) for m in meridians]
        plt.legend(handles=handles, title="Meridian", loc="best")

        plt.axhline(0, linewidth=1, color="black")
        plt.xlabel("Meridian"); plt.ylabel("Mean curvature")
        if y_lim is not None:
            plt.ylim(y_lim)
        plt.title(f"Mean curvature by meridian (eccentricity {ecc})")
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f"box_mean_curvature_by_meridian_ecc_{ecc}.png", dpi=300)
        plt.close()

# --------------------------------------------------------
# Scatter: density vs curvature per ecc (colored meridian)
# --------------------------------------------------------
def plot_scatter_density_vs_curvature_per_ecc(
    subj_means,
    out_dir,
    palette=None,
    meridians=None,
    y_lim=None,
    y_from_data=True,
    y_method="minmax",  # or "quantile"
    y_q=0.01,
    y_pad=0.05,
    x_lim=None,
    x_from_data=False,  # set True to also fix x-axis from data
    x_method="minmax",
    x_q=0.01,
    x_pad=0.05,
):
    """
    One scatter per eccentricity. Colors by meridian.
    If y_lim is None and y_from_data=True, uses global density min/max (or quantiles).
    Optionally set x_from_data=True to also lock curvature axis across panels.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if meridians is None:
        meridians = ["LHM","RHM","LVM","UVM"]
    ecc_order = ["0to2","2to4","4to6","6to8","8to90"]
    marker_map = {"LHM": "o", "RHM": "s", "LVM": "^", "UVM": "D", "HM": "p", "VM": "X"}
    if palette is None:
        palette = _ensure_palette(palette)

    if y_lim is None and y_from_data:
        y_lim = _axis_limits(
            subj_means["streamline_density"], method=y_method, pad=y_pad, q=y_q
        )
    if x_lim is None and x_from_data:
        x_lim = _axis_limits(
            subj_means["mean_curvature"], method=x_method, pad=x_pad, q=x_q
        )

    for ecc in ecc_order:
        sub = subj_means[subj_means["eccentricity_bin"] == ecc]
        plt.figure(figsize=(7,5))
        for mer in meridians:
            sm = sub[sub["meridian"] == mer]
            plt.scatter(
                sm["mean_curvature"], sm["streamline_density"],
                marker=marker_map.get(mer, "o"),
                color=palette[mer], alpha=0.8, label=mer
            )

        # Legend with unique handles
        handles = [Patch(facecolor=palette[m], edgecolor="black", label=m) for m in meridians]
        plt.legend(handles=handles, title="Meridian", loc="best")

        plt.axhline(0, linewidth=1, linestyle="--", color="gray")
        plt.axvline(0, linewidth=1, linestyle="--", color="gray")
        plt.xlabel("Mean curvature"); plt.ylabel("Streamline density ")
        if y_lim is not None:
            plt.ylim(y_lim)
        if x_lim is not None:
            plt.xlim(x_lim)
        plt.title(f"Streamline density vs curvature (eccentricity {ecc})")
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f"scatter_density_vs_curvature_by_ecc_{ecc}.png", dpi=300)
        plt.close()
    return x_lim, y_lim

def plot_scatter_density_vs_curvature(
    df_scatter, out_path, palette, title_suffix="",
    x_from_data=True, x_method="quantile", x_q=0.001, x_pad=0.01,
    y_from_data=True, y_method="quantile", y_q=0.001, y_pad=0.01,x_lim=None,y_lim=None
):
    plt.figure(figsize=(7,5))
    sns.scatterplot(
        data=df_scatter,
        x="mean_curvature",
        y="streamline_density",
        hue="meridian",
        palette=palette,
        alpha=0.6,
        edgecolor="none"
    )
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Mean curvature")
    plt.ylabel("Streamline density ")
    plt.title(f"Streamline density vs curvature{title_suffix}")

    if y_lim is None:

        # --- axis scaling ---
        if y_from_data:
            y_lim = _axis_limits(
                df_scatter["streamline_density"],
                method=y_method, pad=y_pad, q=y_q
            )
            plt.ylim(y_lim)
            
    if x_lim is None:
        if x_from_data:
            x_lim = _axis_limits(
                df_scatter["mean_curvature"],
                method=x_method, pad=x_pad, q=x_q
            )
            plt.xlim(x_lim)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group_csv", required=True)
    ap.add_argument("--output_dir", default="group_outputs")
    ap.add_argument("--areas", default="", help="Only keep rows computed with these areas (e.g., V1 or V1,V2). Empty=keep all.")
    args = ap.parse_args()
   
    

    modes=["hm_rhm_lhm_vm_lvm_uvm","hm_vm","hm_lvm_uvm", "default","hm_vm_lvm_uvm",'hm_vm_uro_ulo_lro_llo','hm_vm_lo_uo','rhm_lhm_lvm_uvm_uro_ulo_lro_llo','hm_vm_om','hm_lvm_uvm_lom_uom']
    modes=["hm_vm_lvm_uvm","hm_lvm_uvm"]
    #for mode in ["default", "hm_vm", "hm_vm_lvm_uvm"]:
    #mode=modes[1]

    for mode in modes:
        print('~~~~~~~~~~~~~')
        print("mode: "+str(mode))
        print('~~~~~~~~~~~~~')
        df = pd.read_csv(args.group_csv)

        #valid_order = ["LHM", "RHM", "LVM", "UVM"]
        #palette = {"LHM":"#2ca02c","RHM":"#2ca02c","UVM":"#1f77b4","LVM":"#d62728"}
        #alette = {"LHM":"#9ACD32","RHM":"#228B22","UVM":"#1f77b4","LVM":"#d62728"}


        # optional filter by areas used at subject stage
        if args.areas.strip():
            target = ",".join([a.strip() for a in args.areas.split(",") if a.strip()])
            # Keep only rows where areas_used matches exactly (so runs are comparable)
            if "areas_used" in df.columns:
                df = df[df["areas_used"].str.upper() == target.upper()]
            print(f"Filtered to areas: {target} → {len(df)} rows")

        # add meridian
        df["meridian"] = df["parcel_id"].map(get_meridian_map())
        df["eccentricity_bin"] = df["parcel_id"].map(get_ecc_map())
        # After assigning the meridian and before any further filtering or use
        df = df[df["meridian"].notna()].copy()
        out_dir = Path(args.output_dir)

        df_orig = df.copy()

        df, palette, valid_order, out_dir = transform_meridians(df, mode, out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print('df after transform')
        print(df)


        df = df[df["meridian"].isin(valid_order)].copy()
        print('df after valid order')
        print(df)

        # stats
        print("\nOne-sample t-tests per meridian (mean_curvature ≠ 0):")
        for mer in valid_order:
            sub = df.loc[df["meridian"]==mer, "mean_curvature"].dropna()
            if len(sub) < 2:
                print(f"{mer}: not enough data")
            else:
                t,p = ttest_1samp(sub, 0.0)
                print(f"{mer}: mean={sub.mean():.4f}, t={t:.3f}, p={p:.4f}")

        # plots

        df["meridian"] = pd.Categorical(df["meridian"], categories=valid_order, ordered=True)
        # per-subject mean for a cleaner boxplot
        # Compute weighted mean curvature per subject-meridian
        # Drop rows where any grouping or required column is missing
        # Drop rows where any grouping or required column is missing
        df = df.dropna(subset=["subject", "meridian", "voxel_count", "mean_curvature"]).copy()
        df["total_curvature"] = df["mean_curvature"] * df["voxel_count"]
        df = df.dropna(subset=["total_curvature"]).copy()
        print(df[["subject", "meridian", "voxel_count", "mean_curvature"]].isna().sum())
        print("DF shape before groupby:", df.shape)
        print("Unique subjects:", df["subject"].nunique())
        print("Unique meridians:", df["meridian"].unique())
        print("Rows going into aggregation:", df.shape[0])
        print("Any NaNs in total_curvature:", df["total_curvature"].isna().sum())



        # normalize the streamline counts per meridian
        subject_total_streamlines = df.groupby("subject")["streamline_count"].sum().to_dict()
        
        #streamline_density_norm
        df["streamline_density_norm"] = df.apply(
            lambda row: (row["streamline_count"] / row["voxel_count"]) / subject_total_streamlines[row["subject"]]
            if row["voxel_count"] > 0 and subject_total_streamlines[row["subject"]] > 0 else np.nan,
            axis=1
        )


        print("df.dtypes:")
        print(df.dtypes)
        print("df.head():")
        print(df.head())


        print("df.columns:", df.columns.tolist())

        print(pd.crosstab(df["subject"], df["meridian"]))


        # Now aggregate
        # Fully isolate the groupby columns from any possible naming conflict

        df["meridian"] = df["meridian"].astype(str)


        group_cols = ["subject", "meridian"]

        agg = (
            df.groupby(group_cols, as_index=False)
            .agg({
                "voxel_count": "sum",
                "total_curvature": "sum"
            })
)

        agg["mean_curvature"] = agg["total_curvature"] / agg["voxel_count"]
        agg = agg[["subject", "meridian", "mean_curvature"]]
        print(agg.describe())



        # Aggregated normalized streamline density per subject and meridian
        agg_density_norm = (
            df.groupby(["subject", "meridian"], as_index=False)
            .agg({
                "streamline_count": "sum",
                "voxel_count": "sum"
            })
        )

        # Re-apply normalization: (streamline_count / voxel_count) / total streamlines
        agg_density_norm["streamline_density_norm"] = (
            (agg_density_norm["streamline_count"] / agg_density_norm["voxel_count"]) /
            agg_density_norm["subject"].map(subject_total_streamlines)
        )



        plt.figure(figsize=(7,5))
        sns.boxplot(data=agg, x="meridian", y="mean_curvature", palette=palette, showfliers=False)
        plt.axhline(0, color="black", linewidth=1)
        plt.xlabel("Meridian")
        plt.ylabel("Mean curvature")
        title_suffix = f" (areas: {args.areas})" if args.areas.strip() else ""
        plt.title(f"Group mean curvature by meridian{title_suffix}")
        plt.tight_layout()
        plt.savefig(out_dir / "group_mean_curvature_boxplot.png", dpi=300)

        # total curvature (voxel-count weighted) — optional
        df["total_curvature"] = df["mean_curvature"] * df["voxel_count"]
        agg_tot = df.groupby(["subject","meridian"], as_index=False)["total_curvature"].sum()

        plt.figure(figsize=(7,5))
        sns.boxplot(data=agg_tot, x="meridian", y="total_curvature", palette=palette, showfliers=False)
        plt.axhline(0, color="black", linewidth=1)
        plt.xlabel("Meridian")
        plt.ylabel("Total curvature (∑ mean_curv × voxels)")
        plt.title(f"Group total curvature by meridian{title_suffix}")
        plt.tight_layout()
        plt.savefig(out_dir / "group_total_curvature_boxplot.png", dpi=300)


        # Recompute streamline_density if not present
        if "streamline_density" not in df.columns and "streamline_count" in df.columns and "voxel_count" in df.columns:
            df["streamline_density"] = df["streamline_count"] / df["voxel_count"]
            print("Added streamline_density column (streamline_count / voxel_count)")

        # Per-subject mean streamline density (boxplot)
        #agg_density = df.groupby(["subject", "meridian"], as_index=False)["streamline_density"].mean()
    
        agg_density = (
        df.groupby(["subject", "meridian"], as_index=False).agg({
            "voxel_count": "sum",
            "streamline_count": "sum"
        })
        )
        agg_density["streamline_density"] = agg_density["streamline_count"] / agg_density["voxel_count"]


        plt.figure(figsize=(7,5))
        sns.boxplot(data=agg_density, x="meridian", y="streamline_density", palette=palette, showfliers=False)
        plt.axhline(0, color="black", linewidth=1)
        plt.xlabel("Meridian")
        plt.ylabel("Streamline density (count / voxel)")
        plt.title(f"Group streamline density by meridian{title_suffix}")
        plt.tight_layout()
        plt.savefig(out_dir / "group_streamline_density_boxplot.png", dpi=300)
        plt.close()


        # Streamline density normalized 
        # === BOXPLOT: Group normalized streamline density by meridian ===
        # plt.figure(figsize=(7,5))
        # sns.boxplot(data=agg_density_norm,x="meridian", y="streamline_density_norm", palette=palette, showfliers=False)
        # plt.axhline(0, color="black", linewidth=1)
        # plt.xlabel("Meridian")
        # plt.ylabel("Normalized streamline density (count / voxel / total streamlines)")
        # plt.title(f"Group normalized streamline density by meridian{title_suffix}")
        # plt.tight_layout()
        # plt.savefig(out_dir / "group_streamline_density_norm_boxplot.png", dpi=300)
        # plt.close()

        # === STRIPPLOT: Scatter points on top of boxplot ===
        # plt.figure(figsize=(7,5))
        # sns.stripplot(
        #             data=agg_density_norm,
        #             x="meridian",
        #             y="streamline_density_norm",
        #             hue="meridian",
        #             palette=palette,
        #             dodge=False,
        #             jitter=True,
        #             alpha=0.6,
        #             legend=False,
        #         )
        # plt.axhline(0, color="black", linewidth=1)
        # plt.xlabel("Meridian")
        # plt.ylabel("Normalized streamline density (count / voxel / total streamlines)")
        # plt.title(f"Group normalized streamline density with scatter{title_suffix}")
        # plt.tight_layout()
        # plt.savefig(out_dir / "group_streamline_density_norm_scatter.png", dpi=300)
        # plt.close()


        # New: Scatter plot of streamline_density colored by meridian
        plt.figure(figsize=(7,5))
        sns.stripplot(
            data=agg_density,
            x="meridian",
            y="streamline_density",
            hue="meridian",
            palette=palette,
            dodge=False,
            jitter=True,
            alpha=0.6,
            legend=False,
        )
        plt.axhline(0, color="black", linewidth=1)
        plt.xlabel("Meridian")
        plt.ylabel("Streamline density ")
        plt.title(f"Streamline density scatter plot by meridian{title_suffix}")
        plt.tight_layout()
        plt.savefig(out_dir / "group_streamline_density_scatter.png", dpi=300)
        plt.close()

        # New: Scatter plot of streamline_count colored by meridian
        plt.figure(figsize=(7,5))
        sns.stripplot(data=agg_density, x="meridian", y="streamline_count", palette=palette, alpha=0.6, jitter=True)
        plt.axhline(0, color="black", linewidth=1)
        plt.xlabel("Meridian")
        plt.ylabel("Streamline Count")
        plt.title(f"Streamline Count scatter plot by meridian{title_suffix}")
        plt.tight_layout()
        plt.savefig(out_dir / "group_streamline_count_scatter.png", dpi=300)
        plt.close()


        # Drop rows with missing values for plotting
        df_scatter = df.dropna(subset=["mean_curvature", "streamline_density", "meridian"]).copy()

        # ----------------------------
        # Extra: HM vs VM difference 
        # Only for hm_vm mode
        # ----------------------------
        if mode == "hm_vm":
            print("Creating HM vs VM difference plot (streamline density vs curvature)...")

            # Pivot to align HM and VM per subject+parcel
            df_hm_vm = (
                df[df["meridian"].isin(["HM", "VM"])]
                .pivot_table(
                    index=["subject", "parcel_id"],
                    columns="meridian",
                    values="streamline_density",
                    dropna=False
                )
                .reset_index()
            )

            # Replace missing with 0 to avoid losing rows
            df_hm_vm["HM"] = df_hm_vm["HM"].fillna(0)
            df_hm_vm["VM"] = df_hm_vm["VM"].fillna(0)

            # Compute difference
            df_hm_vm["density_diff"] = df_hm_vm["HM"] - df_hm_vm["VM"]

            # Merge back curvature values
            df_curv = df[["subject", "parcel_id", "mean_curvature"]].drop_duplicates()
            df_hm_vm = pd.merge(df_hm_vm, df_curv, on=["subject", "parcel_id"], how="left")

            # Drop rows with missing curvature
            df_hm_vm = df_hm_vm.dropna(subset=["mean_curvature"])

            # Label which side is stronger
            df_hm_vm["sign"] = np.where(df_hm_vm["density_diff"] > 0, "HM > VM", "VM > HM")
            color_map = {"HM > VM": "yellow", "VM > HM": "cyan"}

            # Scatter plot
            plt.figure(figsize=(8,6))
            sns.scatterplot(
                data=df_hm_vm,
                x="mean_curvature",
                y="density_diff",
                hue="sign",
                palette=color_map,
                alpha=0.6,
                edgecolor="none",
                marker='+'
            )
            plt.axhline(0, color="gray", linestyle="--", linewidth=1)
            plt.axvline(0, color="gray", linestyle="--", linewidth=1)
            plt.xlabel("Curvature index (gyri < 0, sulci > 0)")
            #plt.ylabel("Streamline density difference (HM − VM)")
            plt.ylabel("HM vs VM")
            plt.title(f"HM vs VM streamline density difference vs curvature{title_suffix}")
            plt.tight_layout()
            plt.savefig(out_dir / "scatter_HM_vs_VM_density_diff_vs_curvature.png", dpi=300)
            plt.close()



        # add eccentricity
        #@df["eccentricity_bin"] = df["parcel_id"].map(get_ecc_map())
        out_csv = out_dir / "full_csv.csv"
        df.to_csv(out_csv, index=False)
        #df = pd.read_csv(args.group_csv)
        df = df_orig
        print('df origin')
        print(df)
        #df["eccentricity_bin"] = df["parcel_id"].map(get_ecc_map())
        df, _, _, _ = transform_meridians(df, mode, out_dir)
        print('df transformed again..')
        print("\nColumns in df:", df.columns.tolist())
        print("Meridians present:", df["meridian"].unique())
        print("Ecc bins present:", df["eccentricity_bin"].unique())
        print("Non-null count (mean_curvature):", df["mean_curvature"].notnull().sum())
        print("Non-null count (streamline_density):", df["streamline_density"].notnull().sum())
        print("Sample rows:\n", df.head(10))

        subj_means = prepare_subject_means(df,meridians=valid_order)
        print('df prepare_subject_means..')
        print(subj_means)
        #############
        # (A) comprehensive grouped boxplot
        plot_comprehensive_box_curvature(
            subj_means,
            out_path=out_dir / "comprehensive_boxplot_curvature_by_meridian_with_ecc_gradients.png", meridians=valid_order, palette=palette
        )

        y_pad=0.01
        x_pad=0.01
        y_q=0.001
        x_q=0.001

        # (B) comprehensive centroid scatter
        plot_comprehensive_centroid_scatter(
            subj_means,
            out_path=out_dir / "comprehensive_scatter_centroids_by_meridian_with_ecc_gradients.png", meridians=valid_order, palette=palette,
            y_lim=None, y_from_data=True, y_method="minmax", y_q=y_q, y_pad=y_pad,
            x_lim=None, x_from_data=True, x_method="minmax", x_q=x_q, x_pad=x_pad)

        plot_box_curvature_by_meridian_per_ecc(subj_means, out_dir,palette=palette,meridians=valid_order,
        y_lim=None, y_from_data=True, y_method="minmax", y_q=y_q, y_pad=y_pad)
        
        x_lim, y_lim = plot_scatter_density_vs_curvature_per_ecc(subj_means, out_dir,palette=palette,meridians=valid_order,
        y_lim=None, y_from_data=True, y_method="minmax", y_q=y_q, y_pad=y_pad,
        x_lim=None, x_from_data=True, x_method="minmax", x_q=x_q, x_pad=x_pad)

        print(palette)
        print(valid_order)

        plot_meridian_centroids(subj_means, out_dir / "centroid_scatter_all_ecc.png",palette=palette,meridians=valid_order)
        plot_meridian_centroids_x(subj_means, out_dir / "centroid_scatter_all_ecc_kde.png",palette=palette,meridians=valid_order, spread_mode='kde')
        plot_meridian_centroids_x(subj_means, out_dir / "centroid_scatter_all_ecc_kde_thr_0.05.png",palette=palette,meridians=valid_order, spread_mode='kde',kde_thr=0.05)
        plot_meridian_centroids_x(subj_means, out_dir / "centroid_scatter_all_ecc_conf_ellipse.png",palette=palette,meridians=valid_order, spread_mode='conf_ellipse', conf_level=0.68)
        plot_meridian_centroids_x(subj_means, out_dir / "centroid_scatter_all_ecc_conf_ellipse_95.png",palette=palette,meridians=valid_order, spread_mode='conf_ellipse', conf_level=0.95)
        plot_meridian_centroids_x(subj_means, out_dir / "centroid_scatter_all_ecc_error_barellipse.png",palette=palette,meridians=valid_order, spread_mode='error_bar')
        plot_meridian_centroids_x(
            subj_means,
            out_dir / "centroid_scatter_all_ecc_kde_eccentricity.png",
            palette=palette,
            meridians=valid_order,
            spread_mode='kde_eccentricity',
            kde_thr=0.08,
            kde_ecc_alpha=0.45,
            ecc_vmax=10.0
        )
        # --- Statistical Modeling Suite ---

        # OLS: Ordinary Least Squares (Linear trend with 95% Confidence Interval)
        #plot_meridian_modelling(subj_means, out_dir / "all_ecc_OLS.png", 
        #                        palette=palette, meridians=valid_order, spread_mode='OLS')

        # GLM: Generalized Linear Model (Gamma/Log-link to enforce positive density)
        #plot_meridian_modelling(subj_means, out_dir / "all_ecc_GLM.png", 
        #                        palette=palette, meridians=valid_order, spread_mode='GLM')

        # LME: Linear Mixed-Effects (Accounts for subject-level repeated measures)
        # Ensure 'subject_id' column exists in subj_means!
        #plot_meridian_modelling(subj_means, out_dir / "all_ecc_LME.png", 
        #                        palette=palette, meridians=valid_order, spread_mode='LME')

        # GMM: Gaussian Mixture Model (Detects sub-clusters/sub-populations)
        plot_meridian_modelling(subj_means, out_dir / "all_ecc_GMM.png", 
                                palette=palette, meridians=valid_order, spread_mode='GMM')        
        #if mode == "hm_vm":
        #    plot_curvature_conditions(df, out_dir, title_suffix=title_suffix, species_map=None)



        plot_scatter_density_vs_curvature(
            df_scatter,
            out_dir / "scatter_streamline_density_vs_curvature.png",
            palette=palette,
            title_suffix=title_suffix,y_lim=y_lim,x_lim=x_lim)



if __name__ == "__main__":
    main()
