import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to
from dipy.io.streamline import load_tck
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.streamline import save_tck


VAREA_MAP = {
    "V1": 1, "V2": 2, "V3": 3, "hV4": 4, "VO1": 5, "VO2": 6,
    "LO1": 7, "LO2": 8, "TO1": 9, "TO2": 10, "V3b": 11, "V3a": 12
}


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
    data = np.round(img.get_fdata()).astype(int)
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
    data = np.round(img.get_fdata()).astype(int)

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
    ap.add_argument("--parc", required=True)
    ap.add_argument("--label_json", required=True)
    ap.add_argument("--tcks_dir", required=True)
    
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

    args = ap.parse_args()

    parc = Path(args.parc).resolve()
    label_json = Path(args.label_json).resolve()
    tcks_dir = Path(args.tcks_dir).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    label_map = load_label_map(label_json)

    filtering_requested = bool(args.visual_area_a.strip()) or bool(args.visual_area_b.strip()) or bool(args.roi_order)
    parc_vareas = Path(args.parc_vareas).resolve() if args.parc_vareas else None
    if filtering_requested and not parc_vareas:
        raise SystemExit("[ERROR] Filtering requested but --parc_vareas not provided")

    # prepare curvature
    work = Path("/tmp/work_gyral_bias")
    work.mkdir(parents=True, exist_ok=True)
    
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
        build_varea_union_mask(parc_vareas, [args.visual_area_a.strip()], a_mask)

        if args.visual_area_b.strip():
            b_mask = work / f"mask_{args.visual_area_b}.nii.gz"
            build_varea_union_mask(parc_vareas, [args.visual_area_b.strip()], b_mask)

            union_mask = work / f"mask_union_{args.visual_area_a}_{args.visual_area_b}.nii.gz"
            build_varea_union_mask(parc_vareas, [args.visual_area_a.strip(), args.visual_area_b.strip()], union_mask)
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
                    filter_tck_ordered_union_python(tck_file, parc_vareas, a_id, b_id, out_tck)
                else:
                    # unordered A<->B, endpoints must hit both masks
                    filter_tck_unordered(tck_file, a_mask, b_mask, out_tck)

            streamline_count_filtered = load_streamline_count(out_tck)

        streamline_density_filtered = (streamline_count_filtered / parcel_volume) if parcel_volume > 0 else np.nan

        rows.append({
            "subject": subject_id,
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


if __name__ == "__main__":
    main()
