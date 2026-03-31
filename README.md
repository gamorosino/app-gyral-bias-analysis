# Gyral Bias Analysis (app-gyral-bias-analysis)

This app computes parcel-wise gyral bias metrics by combining cortical curvature (from FreeSurfer) with tractography-derived streamline counts and densities. Optional filtering allows analysis restricted to specific visual areas and ROI ordering.

---

## Summary

For each parcel, the app computes:

* Mean cortical curvature
* Parcel volume and voxel count
* Total streamline count
* Filtered streamline count (optional)
* Streamline density (filtered / volume)
* 
---
## Author

Gabriele amorosino (g.amorosino@gmail.com)

---

## Inputs

| Name     | Type                 | Description                                |
| -------- | -------------------- | ------------------------------------------ |
| `parc`   | `neuro/parcellation` | Parcel segmentation (NIfTI)                |
| `label`  | `json`               | Label mapping (parcel ID → name)           |
| `tcks`   | `track/tck` (dir)    | Directory of tractograms (`track<ID>.tck`) |
| `output` | `neuro/freesurfer`   | FreeSurfer subject directory               |

---

## Optional Inputs

| Name          | Type                 | Description                           |
| ------------- | -------------------- | ------------------------------------- |
| `parc_vareas` | `neuro/parcellation` | Visual area segmentation              |
| `lh_curv`     | `neuro/mgz`          | Left hemisphere curvature (override)  |
| `rh_curv`     | `neuro/mgz`          | Right hemisphere curvature (override) |

If curvature files are not provided, they are generated from FreeSurfer using `mri_surf2vol`.

---

## Parameters

| Parameter       | Type   | Description                           |
| --------------- | ------ | ------------------------------------- |
| `visual_area_a` | string | First visual area (e.g., `V1`)        |
| `visual_area_b` | string | Second visual area                    |
| `roi_order`     | bool   | Enforce A→B / B→A streamline ordering |

---

## Outputs

| Name                        | Type        | Description                                  |
| --------------------------- | ----------- | -------------------------------------------- |
| `curvature_streamlines.csv` | `table/csv` | Parcel-wise curvature and streamline metrics |

---

## Output Fields

* `parcel_id`
* `parcel_name`
* `hemi` (lh/rh)
* `voxel_count`
* `parcel_volume`
* `mean_curvature`
* `streamline_count_total`
* `streamline_count_filtered`
* `streamline_density_filtered`
* `visual_area_a`
* `visual_area_b`
* `roi_order`

---

## Processing Steps

1. Convert FreeSurfer surface curvature to volume (`mri_surf2vol`)
2. Split parcel segmentation into binary masks
3. Compute mean curvature per parcel
4. Load corresponding tractograms (`track<ID>.tck`)
5. Optionally filter streamlines:

   * Area A only
   * A ↔ B (unordered)
   * A → B / B → A (ordered endpoints)
6. Compute streamline density:

   ```
   density = filtered_count / parcel_volume
   ```
7. Write results to CSV

---

## Filtering Logic

| Mode                | Behavior                            |
| ------------------- | ----------------------------------- |
| None                | All streamlines used                |
| A only              | Keep streamlines touching area A    |
| A + B               | Keep streamlines connecting A and B |
| A + B + `roi_order` | Keep A→B or B→A based on endpoints  |

---

## Assumptions

* Each parcel has a corresponding tractogram:

  ```
  track<ID>.tck
  ```
* Parcel IDs match tractogram filenames
* FreeSurfer directory contains:

  * `surf/`
  * `mri/aparc+aseg.mgz`

---

## Example `config.json`

```json
{
  "parc": "parc.nii.gz",
  "label": "labels.json",
  "tcks": "tcks/",
  "output": "freesurfer_subject",

  "visual_area_a": "V1",
  "visual_area_b": "V2",
  "roi_order": true,
  "parc_vareas": "parc_vareas.nii.gz"
}
```

---

## Dependencies

* FreeSurfer
* MRtrix3 (`tckinfo`, `tckedit`)
* Dipy
