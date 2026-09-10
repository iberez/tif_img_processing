import marimo as mo
import numpy as np
import polars as pl
import pandas as pd
import psutil
import tifffile
import zarr
import dask.array as da
import openslide
import cv2
from skimage.transform import downscale_local_mean
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_triangle, threshold_li
import os
import time
import re
import glob
from io import BytesIO
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

# Headless remote (no DISPLAY): select the non-interactive backend
# BEFORE pyplot is imported, or figure creation can fail.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
import json
from skimage import exposure, filters, morphology, measure
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from skimage.segmentation import watershed
from skimage.morphology import convex_hull_image

from datetime import timedelta

'''bridge functions called in tiff_pipeline_cajal.py'''

def load_bridge_inputs(meta_path):
    """Read one slide's artifacts from annotate_sections(save_outputs=True)."""
    meta_path = Path(meta_path)
    meta = json.loads(meta_path.read_text())
    label_mask = tifffile.imread(meta_path.parent / meta["label_file"])
    if meta.get("scale_y") is None:
        raise ValueError(
            f"{meta_path.name} has no scale_y/scale_x -- re-run annotate_sections "
            f"with raw_slide_path set."
        )
    return label_mask, meta

def inventory_sections(label_mask, slide_name, scale_y, scale_x):
    """Per-section geometry from the section-numbered 10x label mask.

    r_max stays in mask px; need_raw converts using the larger of the two
    axis scales (they differ by <0.1%, so this is conservative by ~5 px).
    """
    labels = np.unique(label_mask)
    labels = labels[labels != 0]
    slices = ndi.find_objects(label_mask)
    s_max = max(scale_y, scale_x)

    rows = []
    for lab in labels:
        sl = slices[int(lab) - 1]
        if sl is None:
            continue
        sub = label_mask[sl] == lab
        coords = np.argwhere(sub).astype(np.float64)
        coords[:, 0] += sl[0].start
        coords[:, 1] += sl[1].start

        cy, cx = coords.mean(axis=0)
        r_max = float(np.hypot(coords[:, 0] - cy, coords[:, 1] - cx).max())
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start

        rows.append({
            "slide": slide_name,
            "label": int(lab),
            "area_px": int(sub.sum()),
            "centroid_y": cy, "centroid_x": cx,
            "bbox_h": h, "bbox_w": w,
            "bbox_diag": float(np.hypot(h, w)),
            "r_max": r_max,
            "need_raw": 2.0 * r_max * s_max,
            "touches_edge": bool(
                sl[0].start == 0 or sl[1].start == 0
                or sl[0].stop == label_mask.shape[0]
                or sl[1].stop == label_mask.shape[1]
            ),
        })
    return pl.DataFrame(rows).sort("need_raw", descending=True)

def plan_canvas(inv_dfs, mask_downsample=10, margin_frac=0.10, round_to=64):
    """Freeze ONE canvas size for the whole study.

    inv_dfs: list of per-slide DataFrames from inventory_sections(), one per
    slide. Must cover every slide you intend to process -- recomputing this
    per batch would give different-sized outputs across batches.

    Note the 10x mask was block-max downsampled, so it is a dilation of the
    true tissue mask; r_max * mask_downsample therefore OVERestimates the
    full-res extent. The bound errs safe.
    """
    all_inv = pl.concat(inv_dfs)
    limiting = all_inv.sort("r_max", descending=True).row(0, named=True)

    r_max_raw = all_inv["r_max"].max() * mask_downsample
    needed = 2.0 * r_max_raw * (1.0 + margin_frac)
    canvas = int(np.ceil(needed / round_to) * round_to)

    n_edge = int(all_inv["touches_edge"].sum())
    print(f"limiting section : label {limiting['label']}  r_max={limiting['r_max']:.1f} (10x)")
    print(f"required extent  : {2 * r_max_raw:.0f} raw px")
    print(f"canvas           : {canvas} x {canvas}  "
          f"(slack {canvas - 2 * r_max_raw:.0f} px)")
    print(f"per-section file : {canvas ** 2 * 2 * 2 / 1e6:.0f} MB uint16 2ch")
    if n_edge:
        print(f"WARNING: {n_edge} section(s) touch a slide edge -- tissue may "
              f"already be cut off in the raw slide, not just the canvas")
    return canvas

def extract_sections(
    slide_path,
    slide_name,
    label_mask,
    inv_df,
    canvas_size,
    out_dir,
    scale_y,
    scale_x,
    n_channels=2,
    background="mask",       # "mask" | "suppress_neighbors" | "raw"
    mask_dilate=0,           # extra halo, in MASK px (1 px ~= 10 raw px)
    compress=None,
    verbose=True,
):
    """Full-res raw crops, centroid-centred in a fixed square canvas.

    Writes <slide_name>.tiff_section_<n>.tiff, one per label in inv_df.

    background:
      "mask"                zero everything outside this section's mask. This is
                            the only path by which remove_debris / remove_whiskers /
                            remove_bright_artifact reach the full-res crop.
      "suppress_neighbors"  zero other sections only; slide background kept raw.
      "raw"                 no zeroing at all.

    scale_y / scale_x come from the bridge meta JSON and are applied per-axis;
    the slide is not an exact integer multiple of the mask on both axes.

    Returns a manifest mapping canvas coords back to raw slide coords.
    """
    if background not in ("mask", "suppress_neighbors", "raw"):
        raise ValueError(f"unknown background mode: {background!r}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tifffile.TiffFile(slide_path) as tif:
        raw = np.stack([tif.pages[c].asarray() for c in range(n_channels)])
    n_ch, H, W = raw.shape
    half = canvas_size // 2

    # Guard against a mask/slide mismatch (wrong slide, transposed mask).
    exp_h, exp_w = label_mask.shape[0] * scale_y, label_mask.shape[1] * scale_x
    if abs(exp_h - H) > scale_y or abs(exp_w - W) > scale_x:
        raise ValueError(
            f"{slide_name}: mask {label_mask.shape} x scale ({scale_y:.4f},"
            f"{scale_x:.4f}) implies {exp_h:.0f}x{exp_w:.0f} but slide is {H}x{W}"
        )

    rows = []
    for rec in inv_df.iter_rows(named=True):
        lab = rec["label"]

        if rec["need_raw"] > canvas_size:
            raise ValueError(
                f"{slide_name} section {lab} needs {rec['need_raw']:.0f} px but "
                f"canvas is {canvas_size} -- rotation will clip. Re-run plan_canvas."
            )

        # Mask pixel i spans raw rows [i*s, (i+1)*s); its centre is at
        # i*s + (s-1)/2, so a subpixel mask centroid maps as below.
        cy_raw = scale_y * rec["centroid_y"] + (scale_y - 1) / 2.0
        cx_raw = scale_x * rec["centroid_x"] + (scale_x - 1) / 2.0

        y0, x0 = int(round(cy_raw)) - half, int(round(cx_raw)) - half
        y1, x1 = y0 + canvas_size, x0 + canvas_size

        sy0, sy1 = max(0, y0), min(H, y1)      # source window, clipped to slide
        sx0, sx1 = max(0, x0), min(W, x1)
        dy0, dx0 = sy0 - y0, sx0 - x0          # destination offset in canvas
        dy1, dx1 = dy0 + (sy1 - sy0), dx0 + (sx1 - sx0)

        canvas_arr = np.zeros((n_ch, canvas_size, canvas_size), raw.dtype)
        canvas_arr[:, dy0:dy1, dx0:dx1] = raw[:, sy0:sy1, sx0:sx1]

        n_zeroed = 0
        if background != "raw":
            # Nearest-neighbour lookup into the mask: no interpolation, and
            # only one bool array at full res.
            r10 = np.clip((np.arange(sy0, sy1) / scale_y).astype(int),
                          0, label_mask.shape[0] - 1)
            c10 = np.clip((np.arange(sx0, sx1) / scale_x).astype(int),
                          0, label_mask.shape[1] - 1)

            if background == "mask":
                keep10 = label_mask == lab
                if mask_dilate:
                    keep10 = ndi.binary_dilation(keep10, iterations=int(mask_dilate))
                kill = ~keep10[np.ix_(r10, c10)]        # everything not this section
            else:
                other10 = (label_mask != 0) & (label_mask != lab)
                kill = other10[np.ix_(r10, c10)]        # other sections only

            n_zeroed = int(kill.sum())
            win = canvas_arr[:, dy0:dy1, dx0:dx1]
            win[:, kill] = 0

        fname = f"{slide_name}.tif_section_{lab}.tiff"
        tifffile.imwrite(
            out_dir / fname,
            canvas_arr,
            photometric="minisblack",
            compression=compress,
            metadata={"axes": "CYX"},
        )

        clipped = bool(dy0 or dx0 or dy1 < canvas_size or dx1 < canvas_size)
        rows.append({
            "slide": slide_name, "section": lab, "filename": fname,
            "canvas_size": canvas_size, "background": background,
            "crop_y0": y0, "crop_x0": x0,       # canvas origin in raw coords
            "clipped": clipped,
            "centroid_y_raw": cy_raw, "centroid_x_raw": cx_raw,
            "r_max_raw": rec["need_raw"] / 2.0,
            "scale_y": scale_y, "scale_x": scale_x,
            "n_zeroed": n_zeroed,
            "touches_edge": rec["touches_edge"],
        })
        if verbose:
            print(f"  {fname}: crop=({y0},{x0}) zeroed={n_zeroed} clipped={clipped}")

    return pl.DataFrame(rows)