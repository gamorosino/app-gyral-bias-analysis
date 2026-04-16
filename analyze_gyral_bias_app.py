import argparse
import json
import os
import re
import subprocess
import nibabel as nib
from pathlib import Path   
import nibabel as derive_tcks_from_whole_tractogram
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to
from dipy.io.streamline import load_tck
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.streamline import save_tck
from matplotlib import pyplot as plt
import seaborn as sns
from utils import save_figure
from typing import Optional, List
from analyze_group_gyral_bias import (
    get_meridian_map,
    get_ecc_map,
    prepare_subject_means,
    transform_meridians,
    plot_comprehensive_box_curvature,
    plot_comprehensive_centroid_scatter,
    plot_box_curvature_by_meridian_per_ecc,
    plot_scatter_density_vs_curvature_per_ecc,
    plot_meridian_centroids,
    plot_meridian_centroids_x,
    plot_meridian_modelling,
)

from retino_split import (
    AREA_LABELS,
    make_subject_patch_mask,
    extract_visual_area_mask,
    intersect_masks,
    run_tckedit_endpoints_in_mask,
)

VAREA_MAP = {
    "V1": 1, "V2": 2, "V3": 3, "hV4": 4, "VO1": 5, "VO2": 6,
    "LO1": 7, "LO2": 8, "TO1": 9, "TO2": 10, "V3b": 11, "V3a": 12
}


def normalize_meridian_mode(value):
    if value is None:
        return "hm_vm_lvm_uvm"

    v = str(value).strip()

    # direct named modes
    allowed = {
        "default",
        "hm_vm",
        "hm_lvm_uvm",
        "hm_vm_lvm_uvm",
        "hm_rhm_lhm_vm_lvm_uvm",
        "hm_vm_uro_ulo_lro_llo",
        "hm_vm_lo_uo",
        "rhm_lhm_lvm_uvm_uro_ulo_lro_llo",
        "hm_vm_om",
        "hm_lvm_uvm_lom_uom",
    }
    if v in allowed:
        return v

    # tolerate comma-separated legacy input
    parts = [p.strip().upper() for p in v.split(",") if p.strip()]
    parts_key = ",".join(parts)

    mapping = {
        "LHM,RHM,LVM,UVM": "default",
        "HM,VM": "hm_vm",
        "HM,LVM,UVM": "hm_lvm_uvm",
        "HM,VM,LVM,UVM": "hm_vm_lvm_uvm",
    }

    if parts_key in mapping:
        return mapping[parts_key]

    raise ValueError(f"Unknown mode: {value}")

def parse_meridian_mode(value: Optional[str]) -> List[str]:
    default = ["HM", "VM", "LVM", "UVM"]
    if value is None:
        return default
    vals = [v.strip().upper() for v in str(value).split(",") if v.strip()]
    return vals or default

def add_merged_meridians_for_plotting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add merged HM and VM rows using the same logic as the group script:
    sum voxel_count, streamline_count, total_curvature, then recompute
    mean_curvature and streamline_density.
    """
    df = df.copy()

    required = {"parcel_id", "meridian", "eccentricity_bin", "voxel_count", "mean_curvature", "streamline_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for merged meridian plotting: {sorted(missing)}")

    df["total_curvature"] = df["mean_curvature"] * df["voxel_count"]

    merged_rows = []

    # HM = LHM + RHM
    hm_input = df[df["meridian"].isin(["LHM", "RHM"])].copy()
    if not hm_input.empty:
        hm = (
            hm_input.groupby(["eccentricity_bin"], as_index=False)
            .agg({
                "voxel_count": "sum",
                "streamline_count": "sum",
                "total_curvature": "sum"
            })
        )
        hm["mean_curvature"] = hm["total_curvature"] / hm["voxel_count"]
        hm["streamline_density"] = hm["streamline_count"] / hm["voxel_count"]
        hm["meridian"] = "HM"
        hm["parcel_id"] = np.nan
        merged_rows.append(hm)

    # VM = LVM + UVM
    vm_input = df[df["meridian"].isin(["LVM", "UVM"])].copy()
    if not vm_input.empty:
        vm = (
            vm_input.groupby(["eccentricity_bin"], as_index=False)
            .agg({
                "voxel_count": "sum",
                "streamline_count": "sum",
                "total_curvature": "sum"
            })
        )
        vm["mean_curvature"] = vm["total_curvature"] / vm["voxel_count"]
        vm["streamline_density"] = vm["streamline_count"] / vm["voxel_count"]
        vm["meridian"] = "VM"
        vm["parcel_id"] = np.nan
        merged_rows.append(vm)

    if merged_rows:
        merged_df = pd.concat(merged_rows, ignore_index=True)
        keep_cols = [c for c in df.columns if c in merged_df.columns]
        merged_df = merged_df.reindex(columns=df.columns, fill_value=np.nan)
        df = pd.concat([df, merged_df], ignore_index=True)

    return df


def make_single_subject_plots(df, plots_dir, meridian_mode="hm_vm_lvm_uvm", only_kde=False):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["streamline_count"] = df["streamline_count_filtered"]
    df["streamline_density"] = df["streamline_density_filtered"]

    df["meridian"] = df["parcel_id"].map(get_meridian_map())
    df["eccentricity_bin"] = df["parcel_id"].map(get_ecc_map())

    df = df.dropna(subset=["mean_curvature", "streamline_density", "meridian"]).copy()

    # dummy subject column for compatibility with transform_meridians()
    if "subject" not in df.columns:
        df["subject"] = "single_subject"

    df, palette, valid_order, _ = transform_meridians(
        df,
        mode=meridian_mode,
        out_dir=plots_dir
    )

    if df.empty:
        print("[WARNING] No rows available for plotting after meridian transformation")
        return

    # Main KDE summary plot
    plot_meridian_centroids_x(
        df,
        plots_dir / "centroid_scatter_all_ecc_kde_thr_0.05.png",
        palette=palette,
        meridians=valid_order,
        spread_mode="kde",
        kde_thr=0.05
    )

    if only_kde:
        return

    # Optional parcel-level scatter
    plot_scatter_density_vs_curvature(
        df,
        plots_dir / "scatter_streamline_density_vs_curvature.png",
        palette=palette,
        title_suffix=""
    )

    # Optional per-eccentricity scatter
    x_lim, y_lim = plot_scatter_density_vs_curvature_per_ecc(
        df,
        plots_dir,
        palette=palette,
        meridians=valid_order,
        y_lim=None,
        y_from_data=True,
        y_method="minmax",
        y_q=0.001,
        y_pad=0.01,
        x_lim=None,
        x_from_data=True,
        x_method="minmax",
        x_q=0.001,
        x_pad=0.01,
    )

    # Optional per-eccentricity boxplot
    plot_box_curvature_by_meridian_per_ecc(
        df,
        plots_dir,
        palette=palette,
        meridians=valid_order,
        y_lim=None,
        y_from_data=True,
        y_method="minmax",
        y_q=0.001,
        y_pad=0.01
    )

def load_streamline_count(tck_file: Path) -> int:
    result = subprocess.run(["tckinfo", str(tck_file)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"tckinfo failed for {tck_file}: {result.stderr}")
    m = re.search(r"count:\s+(\d+)", result.stdout)
    return int(m.group(1)) if m else 0


def run_surf2vol(freesurfer_dir: Path, hemi: str, output_dir: Path) -> Path:
    """
    freesurfer_dir is the FreeSurfer SUBJECT dir containing surf/ and mri/
    """
    subject_id = freesurfer_dir.name
    output_file = output_dir / f"{hemi}.curv.mgz"

    surf_path = freesurfer_dir / "surf" / f"{hemi}.white"
    curv_path = freesurfer_dir / "surf" / f"{hemi}.curv"
    template_path = freesurfer_dir / "mri" / "aparc+aseg.mgz"

    if not surf_path.exists():
        raise FileNotFoundError(f"Missing {surf_path}")
    if not curv_path.exists():
        raise FileNotFoundError(f"Missing {curv_path}")
    if not template_path.exists():
        raise FileNotFoundError(f"Missing {template_path}")

    env = os.environ.copy()
    env["SUBJECTS_DIR"] = str(freesurfer_dir.parent)

    cmd = [
        "mri_surf2vol",
        "--so", str(surf_path), str(curv_path),
        "--subject", subject_id,
        "--identity", subject_id,
        "--template", str(template_path),
        "--o", str(output_file),
    ]
    subprocess.run(cmd, check=True, env=env)
    return output_file


def unpack_segmentation(segmentation_file: Path, output_folder: Path, base_name: str = "parc"):
    output_folder.mkdir(parents=True, exist_ok=True)
    img = nib.load(str(segmentation_file))
    data = np.squeeze(np.round(img.get_fdata()).astype(int))
    labels = np.unique(data)
    for lab in labels:
        if lab == 0:
            continue
        mask = (data == lab).astype(np.uint8)
        out = nib.Nifti1Image(mask, img.affine, img.header)
        nib.save(out, str(output_folder / f"{base_name}_{lab}.nii.gz"))


def compute_mean_curvature(mask_file: Path, curvature_file: Path):
    mask_img = nib.load(str(mask_file))
    mask_data = mask_img.get_fdata().astype(bool)
    if mask_data.ndim == 4 and mask_data.shape[-1] == 1:
        mask_data = mask_data.squeeze(-1)

    curv_img = nib.load(str(curvature_file))

    # resample curvature to mask grid if needed
    if mask_img.shape != curv_img.shape or (not np.allclose(mask_img.affine, curv_img.affine)):
        curv_img = resample_from_to(curv_img, mask_img, order=1)

    curv_data = curv_img.get_fdata()
    if curv_data.ndim == 4 and curv_data.shape[-1] == 1:
        curv_data = curv_data.squeeze(-1)

    vals = curv_data[mask_data]
    vals = vals[~np.isnan(vals)]

    mean_curv = float(np.mean(vals)) if vals.size else np.nan
    voxel_count = int(np.count_nonzero(mask_data))

    voxel_volume = float(np.prod(mask_img.header.get_zooms()[:3]))
    parcel_volume = float(voxel_count * voxel_volume)

    return mean_curv, voxel_count, parcel_volume


def build_varea_union_mask(parc_vareas_path: Path, visual_areas: list[str], out_path: Path) -> Path:
    img = nib.load(str(parc_vareas_path))
    data = np.squeeze(np.round(img.get_fdata()).astype(int))

    ids = []
    for a in visual_areas:
        if a not in VAREA_MAP:
            raise ValueError(f"Unknown visual area '{a}'. Allowed: {sorted(VAREA_MAP.keys())}")
        ids.append(VAREA_MAP[a])

    mask = np.isin(data, ids).astype(np.uint8)
    out = nib.Nifti1Image(mask, img.affine, img.header)
    nib.save(out, str(out_path))
    return out_path


def filter_tck_unordered(in_tck: Path, a_mask: Path, b_mask: Path, out_tck: Path) -> Path:
    tmp = out_tck.with_suffix(".tmp.tck")
    subprocess.run(["tckedit", str(in_tck), str(tmp), "-include", str(a_mask), "-ends_only", "-force"], check=True)
    subprocess.run(["tckedit", str(tmp), str(out_tck), "-include", str(b_mask), "-ends_only", "-force"], check=True)
    tmp.unlink(missing_ok=True)
    return out_tck


def filter_tck_a_or_b(in_tck: Path, union_mask: Path, out_tck: Path) -> Path:
    subprocess.run(["tckedit", str(in_tck), str(out_tck), "-include", str(union_mask), "-ends_only", "-force"], check=True)
    return out_tck


def voxel_of_world(affine, xyz):
    ijk = np.linalg.inv(affine) @ np.array([xyz[0], xyz[1], xyz[2], 1.0])
    return tuple(np.round(ijk[:3]).astype(int))


def filter_tck_ordered_union_python(
    in_tck: Path,
    parc_vareas_path: Path,
    a_id: int,
    b_id: int,
    out_tck: Path
) -> Path:
    """
    Keep streamlines that are A->B OR B->A (union), based on first/last point.
    """
    parc_img = nib.load(str(parc_vareas_path))
    parc_data = np.round(parc_img.get_fdata()).astype(int)
    affine = parc_img.affine
    shape = parc_data.shape[:3]

    sft = load_tck(str(in_tck), reference=str(parc_vareas_path))
    sl = sft.streamlines

    keep_idx = []
    for i, s in enumerate(sl):
        if len(s) < 2:
            continue
        p0 = s[0]
        p1 = s[-1]

        i0, j0, k0 = voxel_of_world(affine, p0)
        i1, j1, k1 = voxel_of_world(affine, p1)

        # bounds check
        if not (0 <= i0 < shape[0] and 0 <= j0 < shape[1] and 0 <= k0 < shape[2]):
            continue
        if not (0 <= i1 < shape[0] and 0 <= j1 < shape[1] and 0 <= k1 < shape[2]):
            continue

        v0 = parc_data[i0, j0, k0]
        v1 = parc_data[i1, j1, k1]

        if (v0 == a_id and v1 == b_id) or (v0 == b_id and v1 == a_id):
            keep_idx.append(i)

    filtered = [sl[i] for i in keep_idx]
    new_sft = StatefulTractogram(filtered, parc_img, Space.RASMM)
    save_tck(new_sft, str(out_tck), bbox_valid_check=False)
    return out_tck

def derive_tcks_from_whole_tractogram(
    tractogram: Path,
    ecc_map: Path,
    polar_map: Optional[Path],
    varea_map: Path,
    ecc_bins: list[str],
    polar_bins: list[str],
    out_tcks_dir: Path,
    out_parc_path: Path,
    out_label_json: Path,
):
    out_tcks_dir.mkdir(parents=True, exist_ok=True)
    roi_dir = out_tcks_dir.parent / "derived_rois"
    roi_dir.mkdir(parents=True, exist_ok=True)

    base_img = nib.load(str(varea_map))
    base_data = np.squeeze(base_img.get_fdata())
    parc_data = np.zeros(base_data.shape, dtype=np.int32)

    labels = []
    parcel_id = 1

    for area in AREA_LABELS:
        area_mask = extract_visual_area_mask(varea_map, area, roi_dir / f"area_{area}.nii.gz")

        for ecc_bin in ecc_bins:
            for polar_bin in polar_bins:
                use_polar = str(polar_bin).lower() != "all"

                patch_mask = make_subject_patch_mask(
                    ecc_map=ecc_map,
                    ang_map=polar_map if use_polar else None,
                    ecc_range=ecc_bin,
                    ang_range=polar_bin if use_polar else None,
                    out_dir=roi_dir / "patches"
                )

                suffix = f"{area}_ecc{ecc_bin}" + (f"_polar{polar_bin}" if use_polar else "")
                roi_mask = intersect_masks(
                    area_mask,
                    patch_mask,
                    roi_dir / f"roi_{parcel_id:03d}_{suffix}.nii.gz"
                )

                roi_img = nib.load(str(roi_mask))
                roi_data = np.squeeze(roi_img.get_fdata()) > 0

                if not np.any(roi_data):
                    continue

                parc_data[roi_data] = parcel_id

                labels.append({
                    "label": parcel_id,
                    "name": suffix
                })

                out_tck = out_tcks_dir / f"track{parcel_id}.tck"
                run_tckedit_endpoints_in_mask(tractogram, roi_mask, out_tck)

                parcel_id += 1

    nib.save(
        nib.Nifti1Image(parc_data, base_img.affine, base_img.header),
        str(out_parc_path)
    )

    out_label_json.write_text(json.dumps(labels, indent=2))

def parse_bins_arg(spec: str) -> list[str]:
    vals = [v.strip() for v in str(spec).split(",") if v.strip()]
    return vals if vals else ["all"]
    
def load_label_map(label_json: Path) -> dict[int, str]:
    if not label_json:
        return {}
    items = json.loads(Path(label_json).read_text())
    out = {}
    for it in items:
        # your ecc/polar labels use either "label" or "voxel_value"
        if "label" in it:
            pid = int(it["label"])
        else:
            pid = int(it["voxel_value"])
        out[pid] = it.get("name", str(pid))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parc", default="")
    ap.add_argument("--label_json", default="")
    ap.add_argument("--input_mode", default="precomputed_tcks",
                choices=["precomputed_tcks", "whole_tractogram"])

    ap.add_argument("--tcks_dir", default="")
    
    ap.add_argument("--tractogram", default="")
    ap.add_argument("--ecc", default="")
    ap.add_argument("--polar", default="")
    ap.add_argument("--varea", default="")
    ap.add_argument("--ecc_bins", default="0_2,2_4,4_6,6_8,8_90")
    ap.add_argument("--polar_bins", default="all")
    
    # Either provide these (recommended for stand-alone FS step)...
    ap.add_argument("--lh_curv", default="")
    ap.add_argument("--rh_curv", default="")
    
    # ...or provide freesurfer_dir so python can run mri_surf2vol itself (fallback)
    ap.add_argument("--freesurfer_dir", default="")
    
    ap.add_argument("--output_csv", required=True)

    ap.add_argument("--parc_vareas", default="")
    ap.add_argument("--visual_area_a", default="")
    ap.add_argument("--visual_area_b", default="")
    ap.add_argument("--roi_order", action="store_true")
    ap.add_argument("--make_plots", action="store_true",
                        help="Generate plots from the computed metrics")
    
    ap.add_argument("--subject_id", type=str, default="sub-unknown",
                        help="Subject identifier for plotting")
    
    ap.add_argument("--plots_dir", type=str, default="plots",
                        help="Directory to save plots")
    ap.add_argument(
                    "--meridian_mode",
                    default="HM,VM,LVM,UVM",
                    help="Comma-separated meridians to show in plots, e.g. HM,VM,LVM,UVM")
    ap.add_argument("--only_kde", action="store_true")
    args = ap.parse_args()

    parc = Path(args.parc).resolve()
    label_json = Path(args.label_json).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.input_mode == "whole_tractogram":
        varea_for_filtering = Path(args.varea).resolve()
    else:
        varea_for_filtering = Path(args.parc_vareas).resolve()
    
    if args.input_mode == "precomputed_tcks":
        if not args.parc.strip():
            raise SystemExit("[ERROR] input_mode=precomputed_tcks requires --parc")
        if not args.label_json.strip():
            raise SystemExit("[ERROR] input_mode=precomputed_tcks requires --label_json")
    
        parc = Path(args.parc).resolve()
        label_json = Path(args.label_json).resolve()
    
        tcks_dir = Path(args.tcks_dir).resolve()
    
    elif args.input_mode == "whole_tractogram":
        required = {
            "tractogram": args.tractogram,
            "ecc": args.ecc,
            "varea": args.varea,
        }
        missing = [k for k, v in required.items() if not str(v).strip()]
        if missing:
            raise SystemExit(f"[ERROR] input_mode=whole_tractogram missing required args: {missing}")
    
        tcks_dir = Path("/tmp/work_gyral_bias/derived_tcks").resolve()
        tcks_dir.mkdir(parents=True, exist_ok=True)
    
        parc = None
        label_json = None
    
    else:
        raise SystemExit(f"[ERROR] Unknown input_mode: {args.input_mode}")
    
    label_map = load_label_map(label_json)

    filtering_requested = bool(args.visual_area_a.strip()) or bool(args.visual_area_b.strip()) or bool(args.roi_order)
    
    if args.input_mode == "precomputed_tcks":
        parc_vareas = Path(args.parc_vareas).resolve() if args.parc_vareas else None
        if filtering_requested and not parc_vareas:
            raise SystemExit("[ERROR] Filtering requested but --parc_vareas not provided")
    else:
        parc_vareas = None  # not used in this mode

    # prepare curvature
    work = Path("/tmp/work_gyral_bias")
    work.mkdir(parents=True, exist_ok=True)

    if args.input_mode == "whole_tractogram":
        derived_parc = work / "derived_parc.nii.gz"
        derived_label_json = work / "derived_label.json"
    
        ecc_bins = parse_bins_arg(args.ecc_bins)
        polar_bins = parse_bins_arg(args.polar_bins)
    
        derive_tcks_from_whole_tractogram(
            tractogram=Path(args.tractogram).resolve(),
            ecc_map=Path(args.ecc).resolve(),
            polar_map=Path(args.polar).resolve() if args.polar.strip() else None,
            varea_map=Path(args.varea).resolve(),
            ecc_bins=ecc_bins,
            polar_bins=polar_bins,
            out_tcks_dir=tcks_dir,
            out_parc_path=derived_parc,
            out_label_json=derived_label_json,
        )
    
        parc = derived_parc
        label_json = derived_label_json
    
    lh_arg = args.lh_curv.strip()
    rh_arg = args.rh_curv.strip()
    
    if lh_arg or rh_arg:
        if not (lh_arg and rh_arg):
            raise SystemExit("[ERROR] If providing curvature files, provide BOTH --lh_curv and --rh_curv")
        lh_curv = Path(lh_arg).resolve()
        rh_curv = Path(rh_arg).resolve()
    else:
        if not args.freesurfer_dir.strip():
            raise SystemExit("[ERROR] Provide either --lh_curv/--rh_curv or --freesurfer_dir")
        freesurfer_dir = Path(args.freesurfer_dir).resolve()
        surf2vol_dir = work / "surf2vol"
        surf2vol_dir.mkdir(exist_ok=True, parents=True)
        lh_curv = run_surf2vol(freesurfer_dir, "lh", surf2vol_dir)
        rh_curv = run_surf2vol(freesurfer_dir, "rh", surf2vol_dir)


    if not lh_curv.exists(): raise FileNotFoundError(lh_curv)
    if not rh_curv.exists(): raise FileNotFoundError(rh_curv)
        
    # unpack parc into masks
    ecc_masks_dir = work / "ecc_polar"
    unpack_segmentation(parc, ecc_masks_dir, "parc")

    # build area masks if needed
    areas = []
    if args.visual_area_a.strip():
        areas.append(args.visual_area_a.strip())
    if args.visual_area_b.strip():
        areas.append(args.visual_area_b.strip())

    a_id = VAREA_MAP.get(args.visual_area_a.strip(), None) if args.visual_area_a.strip() else None
    b_id = VAREA_MAP.get(args.visual_area_b.strip(), None) if args.visual_area_b.strip() else None

    a_mask = b_mask = union_mask = None
    if filtering_requested:
        a_mask = work / f"mask_{args.visual_area_a}.nii.gz"
        build_varea_union_mask(varea_for_filtering, [args.visual_area_a.strip()], a_mask)

        if args.visual_area_b.strip():
            b_mask = work / f"mask_{args.visual_area_b}.nii.gz"
            build_varea_union_mask(varea_for_filtering, [args.visual_area_b.strip()], b_mask)

            union_mask = work / f"mask_union_{args.visual_area_a}_{args.visual_area_b}.nii.gz"
            build_varea_union_mask(varea_for_filtering, [args.visual_area_a.strip(), args.visual_area_b.strip()], union_mask)
        else:
            union_mask = a_mask

    rows = []
    for parc_file in sorted(ecc_masks_dir.glob("parc_*.nii.gz")):
        m = re.search(r"parc_(\d+)", parc_file.name)
        if not m:
            raise RuntimeError(f"[ERROR] Cannot parse parcel_id from {parc_file}")
        parcel_id = int(m.group(1))
        hemi = "lh" if parcel_id % 2 == 1 else "rh"
        curv_file = lh_curv if hemi == "lh" else rh_curv

        mean_curv, voxel_count, parcel_volume = compute_mean_curvature(parc_file, curv_file)

        tck_file = tcks_dir / f"track{parcel_id}.tck"
        if not tck_file.exists():
            # don’t silently skip: this means input is inconsistent with parc
            raise FileNotFoundError(f"Missing expected track file: {tck_file}")

        streamline_count_total = load_streamline_count(tck_file)
        streamline_count_filtered = streamline_count_total

        if filtering_requested:
            filtered_dir = Path("/tmp/work_gyral_bias/tcks_filtered")
            filtered_dir.mkdir(parents=True, exist_ok=True)
            out_tck = filtered_dir / f"track{parcel_id}_areas.tck"

            # If B missing => A only
            if not args.visual_area_b.strip():
                filter_tck_a_or_b(tck_file, union_mask, out_tck)

            else:
                # if roi_order true => ordered A->B + B->A union (python)
                if args.roi_order:
                    if a_id is None or b_id is None:
                        raise SystemExit("[ERROR] roi_order requires both visual_area_a and visual_area_b")
                    filter_tck_ordered_union_python(tck_file, varea_for_filtering, a_id, b_id, out_tck)
                else:
                    # unordered A<->B, endpoints must hit both masks
                    filter_tck_unordered(tck_file, a_mask, b_mask, out_tck)

            streamline_count_filtered = load_streamline_count(out_tck)

        streamline_density_filtered = (streamline_count_filtered / parcel_volume) if parcel_volume > 0 else np.nan

        rows.append({
            "subject": args.subject_id,
            "parcel_id": parcel_id,
            "parcel_name": label_map.get(parcel_id, f"parc_{parcel_id}"),
            "hemi": hemi,
            "voxel_count": voxel_count,
            "parcel_volume": parcel_volume,
            "mean_curvature": mean_curv,
            "streamline_count_total": streamline_count_total,
            "streamline_count_filtered": streamline_count_filtered,
            "streamline_density_filtered": streamline_density_filtered,
            "visual_area_a": args.visual_area_a.strip(),
            "visual_area_b": args.visual_area_b.strip(),
            "roi_order": bool(args.roi_order),
        })

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"[INFO] Wrote {output_csv}")
    
    if args.make_plots:
        print("[INFO] Generating plots...")
    
        df_plot = pd.DataFrame(rows).copy()
        df_plot["subject"] = args.subject_id
    
        try:
            make_single_subject_plots(
                df_plot,
                args.plots_dir,
                meridian_mode=normalize_meridian_mode(args.meridian_mode),
                only_kde=args.only_kde
            )
            print(f"[INFO] Plots saved to: {args.plots_dir}")
        except Exception as e:
            print(f"[WARNING] Plot generation failed: {e}")
                
if __name__ == "__main__":
    main()
