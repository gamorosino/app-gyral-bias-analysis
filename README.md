<h1 align="center">Gyral Bias Analysis</h1>

<p align="center">
  Computes parcel-wise gyral bias metrics by combining FreeSurfer cortical curvature with tractography streamline counts and densities, optionally restricted to visual areas and polar-angle meridians.
</p>

<p align="center">
  <a href="https://github.com/brainlife/abcd-spec"><img src="https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg" alt="ABCD Spec v1.1"></a>
  <a href="https://doi.org/10.25663/brainlife.app.914"><img src="https://img.shields.io/badge/Brainlife-brainlife.app.914-blue.svg" alt="Run on brainlife.io"></a>
</p>

## At a glance

<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

> **Computes parcel-wise gyral bias metrics by combining FreeSurfer cortical curvature with tractography streamline counts and densities, optionally restricted to visual areas and polar-angle meridians.**

Projects FreeSurfer surface curvature to the volume with mri_surf2vol, computes mean curvature and volume per parcel, counts the streamlines of each parcel's tractogram (track<ID>.tck), optionally keeps only streamlines connecting visual_area_a and visual_area_b, and reports streamline density per parcel with summary plots by meridian. Use it to test whether streamline terminations are biased towards gyral crowns. Needs parc (with label), tcks and the FreeSurfer directory; parc_vareas is required when visual area filtering is requested. Note: input_mode and normalize, and the tractogram and prf inputs, are not read by main.

| | |
|---|---|
| **Inputs** | `neuro/freesurfer` · `neuro/tcks` (optional) · `neuro/parcellation/volume` (optional) ×2 · `neuro/track/tck` (optional) · `neuro/prf` (optional) |
| **Outputs** | `neuro/tractmeasures` · `generic/images` |
| **Container** | `gamorosino/tract_align:v1.0` |
| **Resources** | 1 CPUs · 3 h walltime |
| **Requires** | `FREESURFER_LICENSE` |
| **Run it** | [doi.org/10.25663/brainlife.app.914](https://doi.org/10.25663/brainlife.app.914) |

#### Inputs

| Input | Datatype | Required | Description |
|---|---|---|---|
| `freesurfer` | `neuro/freesurfer` | yes | FreeSurfer recon-all subject directory (curvature and aparc+aseg are used). |
| `tcks` | `neuro/tcks` | no | Collection of per-parcel tractograms named track<ID>.tck. |
| `parc_vareas` | `neuro/parcellation/volume` | no | Visual area parcellation (for example Benson14 varea) used to filter streamlines by visual area. |
| `parc` | `neuro/parcellation/volume` | no | Parcellation whose parcels the metrics are computed for, with its label file. |
| `tractogram` | `neuro/track/tck` | no |  |
| `prf` | `neuro/prf` | no |  |

#### Outputs

| Output | Datatype | Description |
|---|---|---|
| `output` | `neuro/tractmeasures` | Table (tractmeasures.csv) of per-parcel curvature, volume, streamline counts and densities. |
| `plots` | `generic/images` | Summary plots of curvature and streamline density by meridian and eccentricity. |

<details>
<summary><b>Parameters</b> (5)</summary>

| Parameter | Type | Default | Description |
|---|---|---|---|
| `visual_area_a` | string | `V1` | First visual area name (for example V1) used to filter streamlines. |
| `visual_area_b` | string |  | Second visual area name; when set, only streamlines connecting area A and area B are kept. |
| `meridian_mode` | string | `HM, VM, LVM, UVM` | Which polar-angle meridian groups to compare in the plots (for example HM, VM, LVM, UVM). |
| `input_mode` | enum | `precomputed_tcks` | Options: `precomputed_tcks`, `whole_tractogram`. |
| `normalize` | boolean | `false` |  |

</details>

<!-- brainlife-app:end -->

## Running the App

### On brainlife.io

Open the App at [doi.org/10.25663/brainlife.app.914](https://doi.org/10.25663/brainlife.app.914), choose `neuro/freesurfer`, `neuro/tcks`, `neuro/parcellation/volume`, `neuro/parcellation/volume`, `neuro/track/tck`, `neuro/prf` from your project, set the parameters and press **Execute**. The results are saved to your project as `neuro/tractmeasures`, `generic/images`.

### Locally

You need [Singularity/Apptainer](https://apptainer.org/) (the App runs in `gamorosino/tract_align:v1.0`), [`jq`](https://jqlang.github.io/jq/), `FREESURFER_LICENSE`.

1. Clone this repository.
2. Create `config.json` with the paths to your inputs and the parameters:

   ```json
   {
       "output": "/path/to/freesurfer/output",
       "tcks": "/path/to/tcks/tcks",
       "tcks_json": "/path/to/tcks/tcks_json",
       "parc_vareas": "/path/to/parc_vareas/parc_vareas",
       "key_vareas": "/path/to/parc_vareas/key_vareas",
       "parc_vareas_label ": "/path/to/parc_vareas/parc_vareas_label ",
       "parc": "/path/to/parc/parc",
       "key": "/path/to/parc/key",
       "label": "/path/to/parc/label",
       "track": "/path/to/tractogram/track",
       "r2": "/path/to/prf/r2",
       "polarAngle": "/path/to/prf/polarAngle",
       "eccentricity": "/path/to/prf/eccentricity",
       "rfWidth": "/path/to/prf/rfWidth",
       "varea": "/path/to/prf/varea",
       "surfaces": "/path/to/prf/surfaces",
       "prf_surfaces": "/path/to/prf/prf_surfaces",
       "visual_area_a": "V1",
       "visual_area_b": "",
       "meridian_mode": "HM, VM, LVM, UVM",
       "input_mode": "precomputed_tcks",
       "normalize": false
   }
   ```

3. Run `./main`.

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

## Citing

If you use this App, please cite it, the software it builds on and brainlife.io:

- Gyral Bias Analysis. brainlife.io App. [doi:10.25663/brainlife.app.914](https://doi.org/10.25663/brainlife.app.914)
- Fischl. FreeSurfer. *NeuroImage* 62:774-781 (2012). [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)
- Garyfallidis et al. Dipy, a library for the analysis of diffusion MRI data. *Frontiers in Neuroinformatics* 8 (2014). [doi:10.3389/fninf.2014.00008](https://doi.org/10.3389/fninf.2014.00008)
- Hayashi S, Caron BA, Heinsfeld AS, et al. brainlife.io: a decentralized and open-source cloud platform to support neuroscience research. *Nature Methods* 21:809–813 (2024). [doi:10.1038/s41592-024-02237-2](https://doi.org/10.1038/s41592-024-02237-2)
- Avesani P, McPherson B, Hayashi S, et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. *Scientific Data* 6:69 (2019). [doi:10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)

## Authors

Gabriele Amorosino

---
