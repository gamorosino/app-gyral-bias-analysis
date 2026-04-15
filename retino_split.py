from pathlib import Path
from typing import Optional
import nibabel as nib
import numpy as np
import subprocess

AREA_LABELS = ["V1","V2","V3","hV4","VO1","VO2","LO1","LO2","TO1","TO2","V3b","V3a"]

def subject_threshold_map(base_map: Path, low: float, high: float, var_type=None):
    img = nib.load(str(base_map))
    data = img.get_fdata()
    if var_type == "angle":
        data = np.abs(data)
    mask = (data >= low) & (data <= high)
    return mask.astype(np.uint8), img

def make_subject_patch_mask(
    ecc_map: Path,
    ang_map: Optional[Path],
    ecc_range: Optional[str],
    ang_range: Optional[str],
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    ecc_is_all = (ecc_range is None) or (str(ecc_range).lower() == "all")
    ang_is_all = (ang_range is None) or (str(ang_range).lower() == "all")

    if ecc_is_all and ang_is_all:
        out = out_dir / "full.nii.gz"
    elif ecc_is_all:
        out = out_dir / f"polar_{ang_range}.nii.gz"
    elif ang_is_all:
        out = out_dir / f"ecc_{ecc_range}.nii.gz"
    else:
        out = out_dir / f"ecc_{ecc_range}_polar_{ang_range}.nii.gz"

    if out.exists():
        return out

    ecc_img = nib.load(str(ecc_map))
    ecc_data = np.squeeze(ecc_img.get_fdata())

    if ecc_is_all and ang_is_all:
        roi = np.ones_like(ecc_data, dtype=np.uint8)
        nib.save(nib.Nifti1Image(roi, ecc_img.affine, ecc_img.header), str(out))
        return out

    if ecc_is_all:
        ecc_mask = np.ones_like(ecc_data, dtype=bool)
    else:
        ecc_low, ecc_high = map(float, str(ecc_range).split("_"))
        ecc_mask, _ = subject_threshold_map(ecc_map, ecc_low, ecc_high)
        ecc_mask = ecc_mask > 0

    if ang_is_all:
        ang_mask = np.ones_like(ecc_data, dtype=bool)
    else:
        if ang_map is None:
            raise ValueError("ang_map is required when ang_range is not None/'all'")
        ang_low, ang_high = map(float, str(ang_range).split("_"))
        ang_mask, _ = subject_threshold_map(ang_map, ang_low, ang_high, var_type="angle")
        ang_mask = ang_mask > 0

    roi = (ecc_mask & ang_mask).astype(np.uint8)
    nib.save(nib.Nifti1Image(roi, ecc_img.affine, ecc_img.header), str(out))
    return out

def extract_visual_area_mask(varea_img: Path, area_name: str, out_path: Path) -> Path:
    if out_path.exists():
        return out_path

    if area_name not in AREA_LABELS:
        raise ValueError(f"Unknown area '{area_name}'. Valid: {AREA_LABELS}")

    val = AREA_LABELS.index(area_name) + 1
    img = nib.load(str(varea_img))
    data = np.round(img.get_fdata()).astype(int)
    mask = (data == val).astype(np.uint8)
    nib.save(nib.Nifti1Image(mask, img.affine, img.header), str(out_path))
    return out_path

def intersect_masks(mask_a: Path, mask_b: Path, out_path: Path) -> Path:
    img_a = nib.load(str(mask_a))
    img_b = nib.load(str(mask_b))
    a = np.squeeze(img_a.get_fdata()) > 0
    b = np.squeeze(img_b.get_fdata()) > 0
    out = (a & b).astype(np.uint8)
    nib.save(nib.Nifti1Image(out, img_a.affine, img_a.header), str(out_path))
    return out_path

def run_tckedit_endpoints_in_mask(in_tck: Path, mask: Path, out_tck: Path):
    subprocess.run([
        "tckedit", str(in_tck), str(out_tck),
        "-include", str(mask),
        "-ends_only",
        "-force",
    ], check=True)
    return out_tck
