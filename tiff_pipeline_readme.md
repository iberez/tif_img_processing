# TIFF Processing Pipeline

This repo contains all the code needed to turn raw microscope slide images into processed, **autoaligned single-section TIFF images**: one image per tissue section, rotated and centered.

**Contents:** [Pipeline overview](#pipeline-overview) · [Key files](#key-files) · [How to run](#how-to-run) · [Output](#output)

---

## Pipeline overview

The pipeline takes a folder named by a brain ID (e.g. `137_02`) that contains the raw microscope slide images in TIFF format, and writes all of its outputs to subfolders inside that same folder. The slides are processed in three main steps:

1. **Slide preprocessing** — works on a 10× downsampled copy of each slide: removes bright artifacts and background, repairs cracks, separates the individual tissue sections, cleans up debris, and numbers the sections. *Output:* `preprocessed/`
2. **Extraction of single sections & metadata configuration** — maps the section labels from step 1 back onto the full-resolution slide and crops each section into its own image. *Output:* `single_section_autoraw/`
3. **Single-section processing** — autoaligns each section: finds its midline, rotates the section so the midline is vertical, and centers it. *Output:* `single_section_autoalign/`

Once processing is done, all sections can be reviewed and manually edited (e.g. keep/reject, rotate, flip, remove unwanted fragments).

---

## Key files

| File | Role |
|---|---|
| `tiff_pipeline_pp_functions.py` | Step 1 — slide preprocessing functions |
| `tiff_pipeline_bridge_functions.py` | Step 2 — functions for extracting single sections and configuring metadata |
| `tiff_pipeline_ss_functions.py` | Step 3 — single-section processing, including autoalignment (rotation and centering) |
| `tiff_pipeline_mega_loop.py` | Wrapper around the three modules above that processes all slides |
| `GUI_tiff_pipeline_cajal.py` | marimo app (GUI) for running the pipeline — see [How to run](#how-to-run) |
| **TODO** | Module(s) behind the review & manual-edit step, e.g. `tiff_pipeline_review_functions.py` |

---

## How to run

The pipeline is designed to run as a [marimo](https://marimo.io) notebook and assumes marimo is already installed (see the [marimo installation guide](https://docs.marimo.io/getting_started/installation/)).

Launch the GUI in app view:

```bash
marimo run GUI_tiff_pipeline_cajal.py
```

> **TODO:** Other Python dependencies (or link a `requirements.txt` / environment file).
>
> **TODO:** Short walkthrough of the GUI — choosing the `<brain_id>` folder, running the pipeline, reviewing and editing sections.

---

## Output

All outputs are written into the `<brain_id>` folder, next to the raw slide images.

### Naming key

| Placeholder | Meaning | Example |
|---|---|---|
| `<brain_id>` | Name of the brain folder | `137_02` |
| `<slide_id>` | Raw slide file name without its extension (written as `slice_<slide_id>` or `slide_<slide_id>` in some output names) | `S001` |
| `<n>` | Section number on the slide, as shown in that slide's `*_preprocessed_overview.png` | `1`, `2`, … |

Files named with `<slide_id>` exist once per slide; files that also carry `<n>` exist once per section.

### Folder structure

```text
<brain_id>/                       # e.g. 137_02/
├── S001.tif, S002.tif, ...       # raw slide images (input)
│
├── preprocessed/                 # step 1 · slide preprocessing
│   ├── <slide_id>_preprocessed_overview.png
│   ├── <slide_id>_preprocessed_all_steps.png
│   └── bridge_inputs/            # step 1 → step 2 hand-off
│       ├── <slide_id>_section_labels.tiff
│       └── <slide_id>_bridge_meta.json
│
├── single_section_autoraw/       # step 2 · extracted sections, not yet aligned
│   └── <slide_id>.tif_section_<n>.tiff
│
└── single_section_autoalign/     # step 3 · single-section processing
    ├── imgs/                     # ★ main output: autoaligned section images
    │   └── <brain_id>_img_slice_<slide_id>_section_<n>.tiff
    ├── masks/                    # section masks
    │   └── <brain_id>_mask_slice_<slide_id>_section_<n>.npy
    ├── plots/                    # per-section QC plots
    │   ├── combined/             # TODO
    │   └── <brain_id>_autorotated_centered_slice_<slide_id>_section_<n>.pdf
    └── transforms/               # per-slide transform tables
        └── <brain_id>_slide_<slide_id>_transforms.csv
```

> **Disk space:** in sample brain `137_02`, each raw slide is ~1.9 GB and every section adds ~0.4 GB of output (a 184 MB TIFF in both `single_section_autoraw/` and `single_section_autoalign/imgs/`, plus a 46 MB mask), so a 15-section slide produces ~6 GB.

### Folder contents

#### `<brain_id>/` — input

The folder the pipeline is run on, named by brain ID. It holds the raw slide images, one per microscope slide (`S001.tif`, `S002.tif`, …); in `137_02` these are 14 slides of ~1.9 GB each, stored as 2-channel uint16 OME-TIFFs. The pipeline adds the three output folders below.

> **TODO:** Required naming/format for the raw slide files, if any.

#### `preprocessed/` — step 1: slide preprocessing

Two images per slide, plus the `bridge_inputs/` subfolder:

- `<slide_id>_preprocessed_overview.png` — the **labeled output slide**: every detected tissue section, marked with its section number (assigned in reading order by default). These are the `<n>` numbers used in all downstream file names, so start here to check that the sections were found and separated correctly, and to look up which section is which.
- `<slide_id>_preprocessed_all_steps.png` — all the steps of the preprocessing flow for the slide; useful for tracking down which step went wrong when an overview looks off.

#### `preprocessed/bridge_inputs/` — hand-off from step 1 to step 2

Written at the end of preprocessing and read by step 2 to locate each section on the full-resolution slide:

- `<slide_id>_section_labels.tiff` — section label image at the 10× downsampled resolution (uint16): each pixel holds the number of the section it belongs to, with `0` = background.
- `<slide_id>_bridge_meta.json` — slide metadata for step 2: per-axis scale factors between the label image and the raw slide (`scale_y`, `scale_x`), the raw slide's dimensions, and the geometry of each section (number, centroid, area and bounding box, in label-image pixels).

#### `single_section_autoraw/` — step 2: extracted sections, not yet aligned

- `<slide_id>.tif_section_<n>.tiff` — section `<n>`, cropped from the raw slide at full resolution (no resampling) and placed, centered on its centroid, on a square canvas that is the same size for every section. Pixels belonging to neighboring sections are suppressed. Same channels and bit depth as the raw slide. The prefix is the raw slide's full file name, extension included (e.g. `S001.tif_section_1.tiff`).

#### `single_section_autoalign/` — step 3: single-section processing

The processed, autoaligned single-section images, plus the masks, plots and transforms that go with them, in four subfolders.

#### `single_section_autoalign/imgs/`

- `<brain_id>_img_slice_<slide_id>_section_<n>.tiff` — ★ **the main output of the pipeline**: each section after autoalignment, rotated so its midline is vertical and centered, on the same canvas size as in step 2 (e.g. `137_02_img_slice_S001_section_1.tiff`).

#### `single_section_autoalign/masks/`

- `<brain_id>_mask_slice_<slide_id>_section_<n>.npy` — binary mask of each section, stored as a NumPy array (load with `np.load`). **TODO:** what the mask covers, and whether it is saved in the aligned frame (i.e. overlays the matching image in `imgs/`).

#### `single_section_autoalign/plots/`

- `<brain_id>_autorotated_centered_slice_<slide_id>_section_<n>.pdf` — per-section figure of the autorotated, centered section, for a quick visual check of the alignment. **TODO:** describe what the figure shows.
- `combined/` — **TODO:** describe contents.

#### `single_section_autoalign/transforms/`

- `<brain_id>_slide_<slide_id>_transforms.csv` — one table per slide with the transforms of its sections. **TODO:** list the columns and units, and note whether the manual review also writes to this file.

> **Transform convention:** flip left–right first, then rotate counter-clockwise (CCW).

#### Review & manual-edit outputs

> **TODO:** Files or folders written by the review / manual-edit step (e.g. saved decisions, edited full-resolution images, cached thumbnails), if not already covered above.
