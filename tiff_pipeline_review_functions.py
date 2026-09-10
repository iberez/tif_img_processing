
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

"""
tiff_pipeline_review_fuctions.py - manual review stage for autoaligned single-section TIFFs called in tiff_pipeline_cajal.py.

Consumes the padded square TIFFs emitted by the single-section autoalign
pipeline and lets a human confirm / reject each one and apply gross
orientation fixes (90-degree rotation, L/R mirror, small angular nudges).

Reactivity contract (Marimo):
    Cell 1  -> find_section_paths(), load_review_thumbs()      [disk, slow]
    Cell 2  -> build_review_controls()                          [UI creation]
    Cell 3  -> render_review_grid()                             [reads .value]
    Cell 4  -> collect_decisions()                              [manifest]
    Cell 5  -> apply_decisions() behind a mo.ui.run_button      [disk, slow]

Cells 2 and 3 MUST stay separate: a UI element's .value cannot be read in
the cell that defines it. Keeping the load in cell 1 means a button click
re-runs only the render, so previews update without touching disk.

Transform convention, used identically in preview and apply:
    FLIP LEFT/RIGHT FIRST, THEN ROTATE COUNTER-CLOCKWISE.
If these two paths ever disagree the bug is nearly invisible, because the
preview still looks plausible.
"""

# --------------------------------------------------------------------------
# Discovery and loading
# --------------------------------------------------------------------------

SECTION_RE = re.compile(r"_section_(\d+)\.tiff$")



def find_section_paths(brain_id, slide_num, root="/bigdata/isaac/rabies_img_processing"):
    """Aligned section TIFFs for one slide, ordered by integer section number.

    Sorts on the parsed integer, not the string, so slides with 10+ sections
    do not come back in lexicographic order (2 after 19).
    """
    aligned_dir = os.path.join(root, str(brain_id), "single_section_autoalign", "imgs")
    pattern = os.path.join(aligned_dir, f"{brain_id}_img_slice_{slide_num}_section_*.tiff")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"no sections matched {pattern}")
    return sorted(paths, key=lambda p: int(SECTION_RE.search(p).group(1)))


def section_number(path):
    """Integer section index parsed out of a filename."""
    return int(SECTION_RE.search(path).group(1))


def load_thumbs(section_paths, factor=0.1, verbose=True):
    """Load each section at `factor` scale as (C, Y, X) uint16.

    Full resolution is freed each iteration, so peak memory is one full-res
    section rather than the whole slide's worth.
    """
    thumbs = []
    for p in section_paths:
        with tifffile.TiffFile(p) as tif:
            full = tif.series[0].asarray()  # (C, Y, X) uint16
        down = np.stack(
            [
                cv2.resize(full[c], None, fx=factor, fy=factor,
                           interpolation=cv2.INTER_AREA)
                for c in range(full.shape[0])
            ],
            axis=0,
        )
        if verbose:
            print(f"{os.path.basename(p)}: {full.shape} -> {down.shape}")
        thumbs.append(down)
        del full
    return thumbs


def pad_to_common(thumbs):
    """Center-pad every thumbnail onto a shared canvas so grid cells align."""
    h = max(t.shape[1] for t in thumbs)
    w = max(t.shape[2] for t in thumbs)
    out = []
    for t in thumbs:
        dh, dw = h - t.shape[1], w - t.shape[2]
        top, left = dh // 2, dw // 2
        out.append(np.pad(t, ((0, 0), (top, dh - top), (left, dw - left))))
    return out


def load_review_thumbs(section_paths, factor=0.1, cache_path=None, verbose=True):
    """Load + pad, with an optional .npz cache so re-opening a slide is instant."""
    if cache_path and os.path.exists(cache_path):
        with np.load(cache_path) as z:
            cached = [z[k] for k in sorted(z.files, key=lambda k: int(k.split("_")[1]))]
        if len(cached) == len(section_paths):
            if verbose:
                print(f"loaded {len(cached)} thumbs from cache {cache_path}")
            return cached
        print(f"cache has {len(cached)} thumbs but {len(section_paths)} paths; reloading")

    thumbs = pad_to_common(load_thumbs(section_paths, factor=factor, verbose=verbose))

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, **{f"t_{i}": t for i, t in enumerate(thumbs)})
        if verbose:
            print(f"cached {len(thumbs)} thumbs -> {cache_path}")
    return thumbs


# --------------------------------------------------------------------------
# Display conversion (uint16 -> uint8, preview only, never written out)
# --------------------------------------------------------------------------


def _stretch(plane, p_low=0.5, p_high=99.5):
    """Percentile autoscale one 2-D plane to uint8.

    Percentiles are taken over NONZERO pixels only. zero_background() ran
    upstream, so padded background would otherwise dominate the low
    percentile and wash the tissue out.
    """
    if plane.ndim != 2:
        raise ValueError(f"_stretch expects a single 2-D plane, got {plane.shape}")
    fg = plane[plane > 0]
    if fg.size == 0:
        return np.zeros(plane.shape, np.uint8)
    lo, hi = np.percentile(fg, [p_low, p_high])
    hi = max(hi, lo + 1)
    scaled = (plane.astype(np.float32) - lo) / (hi - lo)
    return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)


def _stretch_u8(plane):
    return plane if plane.dtype == np.uint8 else _stretch(plane)


def to_pil_ready(arr):
    """Coerce to something PIL accepts: 2-D uint8, or (Y, X, 3) uint8.

    PIL reads the LAST axis as bands, so a (2, Y, X) array raises
    'Cannot handle this data type: (1, 1, N), |u1'. This catches
    channel-first input before it gets that far.
    """
    a = np.asarray(arr)
    if a.ndim == 3:
        if a.shape[0] <= 4 and a.shape[0] < a.shape[-1]:  # channel-first
            if a.shape[0] == 1:
                a = a[0]
            elif a.shape[0] == 2:  # two-channel -> green / magenta composite
                rgb = np.zeros(a.shape[1:] + (3,), np.uint8)
                rgb[..., 0] = _stretch_u8(a[1])
                rgb[..., 1] = _stretch_u8(a[0])
                rgb[..., 2] = _stretch_u8(a[1])
                return np.ascontiguousarray(rgb)
            else:
                a = np.moveaxis(a[:3], 0, -1)
        elif a.shape[-1] == 1:
            a = a[..., 0]
        elif a.shape[-1] not in (3, 4):
            raise ValueError(f"unexpected thumbnail shape {a.shape}")
    if a.dtype != np.uint8:
        a = _stretch(a)
    # np.flip returns a negative-stride view and PIL is inconsistent about those.
    return np.ascontiguousarray(a)


def _as_display(t, channel=0):
    """One thumbnail -> display-ready uint8. channel='composite' for both."""
    a = np.asarray(t)
    if (a.ndim == 3 and a.shape[0] <= 4 and a.shape[0] < a.shape[-1]
            and channel != "composite"):
        a = a[channel]
    return to_pil_ready(a)

def _pick_plane(t, channel=0):
    a = np.asarray(t)
    if a.ndim == 3 and a.shape[0] <= 4 and a.shape[0] < a.shape[-1]:
        return a[channel]
    return a
    
def to_display_stack(thumbs, channel=0, verbose=True):
    """Precompute display arrays once at load time."""
    out = []
    for i, t in enumerate(thumbs):
        d = _as_display(t, channel)
        assert d.ndim == 2 or d.shape[-1] == 3, f"section {i}: bad display shape {d.shape}"
        out.append(d)
    if verbose and out:
        print(f"{len(out)} display thumbs, {out[0].shape}, {out[0].dtype}")
    return out

def to_display_shared(thumbs, channel=0, p_low=0.5, p_high=99.5):
    """Autoscale several thumbnails onto ONE shared intensity range.

    to_display_stack() scales each image by its own percentiles, which is right
    for a review contact sheet but wrong for a before/after check: per-image
    autoscale masks real intensity changes and manufactures apparent ones.
    """
    planes = [_pick_plane(t, channel) for t in thumbs]
    pooled = np.concatenate([p[p > 0].ravel() for p in planes])
    if pooled.size == 0:
        return [np.zeros(p.shape, np.uint8) for p in planes]
    lo, hi = np.percentile(pooled, [p_low, p_high])
    hi = max(hi, lo + 1)
    out = []
    for p in planes:
        scaled = (p.astype(np.float32) - lo) / (hi - lo)
        out.append((np.clip(scaled, 0, 1) * 255).astype(np.uint8))
    return out


def plot_comparison(thumbs, labels, channel=0, shared_scale=True,
                    guides=True, figsize=(11, 6), cmap="gray"):
    """Side-by-side panels on a common intensity scale, with midline guides.

    Set shared_scale=False only if you want each panel autoscaled to itself,
    which makes the panels prettier and the comparison meaningless.
    """
    disp = (to_display_shared(thumbs, channel=channel) if shared_scale
            else [_as_display(t, channel) for t in thumbs])

    fig, axes = plt.subplots(1, len(disp), figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, d, lab in zip(axes, disp, labels):
        ax.imshow(d, cmap=cmap, vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(lab, fontsize=9)
        if guides:
            ax.axvline(d.shape[1] / 2, color="r", lw=0.7, alpha=0.5)
            ax.axhline(d.shape[0] / 2, color="r", lw=0.7, alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"shared intensity scale · channel {channel}" if shared_scale
        else "PER-IMAGE autoscale — contrast differences are not meaningful",
        fontsize=8,
    )
    return fig

# --------------------------------------------------------------------------
# Preview transforms
# --------------------------------------------------------------------------


def pending_angle(v):
    """Net CCW rotation in degrees from the button counters."""
    return (v["rot90"] + v["nudge_ccw"] - v["nudge_cw"]) % 360


def rotate_display(img, angle_deg):
    """Preview rotation. Exact (no interpolation) for multiples of 90."""
    a = angle_deg % 360
    if a == 0:
        return img
    if a % 90 == 0:
        return np.ascontiguousarray(np.rot90(img, k=int(a) // 90, axes=(0, 1)))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), a, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


def apply_pending_display(disp, angle_deg, flip_lr):
    """Flip L/R first, then rotate CCW. Mirrors apply_transform_full_res().

    Assumes Y is axis 0, so `disp` must already be display-ready. Handing this
    a (C, Y, X) array silently mirrors along Y instead of X.
    """
    out = np.flip(disp, axis=1) if flip_lr else disp
    return rotate_display(np.ascontiguousarray(out), angle_deg)


def add_reference_lines(img, thickness=3, color=(255, 90, 90), alpha=0.55):
    """Fixed crosshair, drawn AFTER rotation so it stays put as tissue moves.

    Without a stationary reference a 1-degree nudge is invisible: at r=349 px
    it displaces an edge pixel ~6 px, which is under 2 screen px once the
    thumbnail is scaled down to card width.
    """
    rgb = np.stack([img] * 3, axis=-1) if img.ndim == 2 else img.copy()
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    cx, cy = w // 2, h // 2
    t = max(1, thickness // 2)
    col = np.array(color, np.float32)
    for sl in (np.s_[:, cx - t:cx + t + 1], np.s_[cy - t:cy + t + 1, :]):
        rgb[sl] = (rgb[sl] * (1 - alpha) + col * alpha).astype(np.uint8)
    return rgb


# --------------------------------------------------------------------------
# UI: controls and grid
# --------------------------------------------------------------------------


def build_review_controls(n_sections, nudge_step=5):
    """One button group per section, held in a mo.ui.array so the DAG sees it.

    Every control is a button. Status is a single toggle rather than two
    buttons because two independent buttons cannot share one value without
    mo.state and on_change callbacks.

    Button labels are fixed at creation, so a toggle cannot relabel itself to
    show its own state. render_review_grid() carries the readout instead.
    """
    return mo.ui.array(
        [
            mo.ui.dictionary(
                {
                    "status": mo.ui.button(
                        value="keep",
                        on_click=lambda v: "reject" if v == "keep" else "keep",
                        label="keep / reject",
                    ),
                    "rot90": mo.ui.button(
                        value=0,
                        on_click=lambda v: (v + 90) % 360,
                        label="⟲ 90°",
                    ),
                    # s=nudge_step binds now; a bare closure would capture later values.
                    "nudge_ccw": mo.ui.button(
                        value=0,
                        on_click=lambda v, s=nudge_step: v + s,
                        label=f"↺ {nudge_step}°",
                    ),
                    "nudge_cw": mo.ui.button(
                        value=0,
                        on_click=lambda v, s=nudge_step: v + s,
                        label=f"↻ {nudge_step}°",
                    ),
                    "flip_lr": mo.ui.button(
                        value=False,
                        on_click=lambda v: not v,
                        label="flip L/R",
                    ),
                }
            )
            for _ in range(n_sections)
        ]
    )


def render_review_grid(thumbs_in, section_paths, controls, edits = None, channel=0, ncols=3,
                       width=300, show_guides=True, debug=False):
    """Contact sheet of live previews with per-section controls beneath each.

    Accepts raw (C, Y, X) uint16 thumbs or precomputed display arrays. Each
    preview is re-derived from the unmodified source every render, so repeated
    warpAffine interpolation never compounds across clicks.
    """
    cards = []
    for i, (t, p) in enumerate(zip(thumbs_in, section_paths)):
        d = _as_display(t, channel)  # must precede the transform: Y-first assumed

        v = controls.value[i]
        angle, flip = pending_angle(v), v["flip_lr"]
        e = edits.get(i, {"extra_rot": 0.0, "removals": []})
        prev = edited_preview(d, angle, flip, e)              # <- new
        prev = to_pil_ready(add_reference_lines(prev) if show_guides else prev)

        if debug and i == 0:
            print(f"in={np.asarray(t).shape} disp={d.shape} prev={prev.shape} "
                  f"{prev.dtype} contig={prev.flags['C_CONTIGUOUS']}")

        keep = v["status"] == "keep"
        cards.append(
            mo.vstack(
                [
                    mo.md(f"**{i + 1}** · `{os.path.basename(p)}`"),
                    mo.image(prev, width=width),
                    mo.md(
                        f"**{v['status'].upper()}** · rot **{angle}°** · "
                        f"flip **{'L/R' if flip else 'none'}**"
                    ),
                    controls[i]["status"],
                    mo.hstack(
                        [controls[i]["rot90"],
                         controls[i]["nudge_ccw"],
                         controls[i]["nudge_cw"]],
                        justify="start",
                        gap=0.25,
                    ),
                    controls[i]["flip_lr"],
                ],
                gap=0.25,
            ).style(
                {
                    "border": f"2px solid {'#2a7' if keep else '#c33'}",
                    "border-radius": "6px",
                    "padding": "0.5rem",
                    "opacity": "1.0" if keep else "0.55",
                }
            )
        )

    return mo.vstack(
        [mo.hstack(cards[r:r + ncols], widths="equal", gap=1)
         for r in range(0, len(cards), ncols)],
        gap=1,
    )

# --------------------------------------------------------------------------
# Interactive editing: sliver-to-midline rotation, lasso removal
# --------------------------------------------------------------------------

def selection_to_mask(sel, shape):
    """Boolean mask over image pixels from a mo.ui.matplotlib selection.

    Uses only get_mask(x, y), the documented interface. x is the column and y
    the row, which is the axes coordinate space imshow() sets up. Always call
    this at THUMBNAIL scale: a full-res section is ~48M points and get_mask
    would crawl.
    """
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    flat = sel.get_mask(xx.ravel().astype(float), yy.ravel().astype(float))
    return np.asarray(flat, dtype=bool).reshape(h, w)


def midline_angle_from_mask(mask, min_px=25, min_elongation=2.0):
    """CCW degrees that bring the selection's long axis to vertical.

    PCA on the selected pixel coordinates. Returns (angle, message); angle is
    None when the selection cannot define a line, and the message says why.

    A lasso closes itself, so a perfectly straight stroke encloses no area and
    selects nothing. Draw an elongated sliver along the midline instead.
    """
    ys, xs = np.nonzero(mask)
    n = int(ys.size)
    if n < min_px:
        return None, f"only {n} px selected — draw an elongated sliver, not a hairline"

    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    pts -= pts.mean(axis=0)
    _, s, vt = np.linalg.svd(pts, full_matrices=False)

    elong = float(s[0] / max(s[1], 1e-9))
    if elong < min_elongation:
        return None, f"selection too round (elongation {elong:.1f}) to define a line"

    vx, vy = float(vt[0, 0]), float(vt[0, 1])
    # Image y points DOWN, so negate it to get a standard math-coords angle.
    phi = np.degrees(np.arctan2(-vy, vx))
    # A line is invariant under 180 deg, so take the smallest rotation to vertical.
    theta = (90.0 - phi) % 180.0
    if theta > 90.0:
        theta -= 180.0
    return theta, f"{n} px, elongation {elong:.1f}, rotate {theta:+.2f}° CCW"


def upscale_mask(mask_small, target_hw):
    """Nearest-neighbour upscale to exact target dims.

    Passing explicit (w, h) rather than a scale factor keeps this correct when
    the full-res shape is not an integer multiple of the thumbnail shape.
    """
    h, w = int(target_hw[0]), int(target_hw[1])
    up = cv2.resize(mask_small.astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST)
    return up.astype(bool)


def apply_removals(arr, masks_small, verbose=False):
    """Zero every selected region on a (C, Y, X) or (Y, X) array.

    Masks are recorded at thumbnail scale in the frame the user was looking at,
    so this must run AFTER the rotation/flip, never before.
    """
    if not masks_small:
        return arr
    hw = arr.shape[-2:]
    combined = np.zeros(hw, bool)
    for m in masks_small:
        combined |= upscale_mask(m, hw)
    out = arr.copy()
    out[..., combined] = 0
    if verbose:
        print(f"removed {int(combined.sum())} px of {combined.size} "
              f"({100 * combined.mean():.2f}%)")
    return out


# --------------------------------------------------------------------------
# Edit state + canvas
# --------------------------------------------------------------------------


def empty_edits(n_sections):
    """Per-section interactive edits, held in mo.state.

    extra_rot accumulates sliver-derived corrections; removals are thumbnail-
    scale boolean masks recorded in the frame the user was looking at.
    """
    return {i: {"extra_rot": 0.0, "removals": [], "drawn_at": 0.0}
            for i in range(n_sections)}


def edited_preview(disp, angle_deg, flip_lr, edit):
    """Display array with buttons + interactive edits applied, in apply order.

    flip, then the button angle plus any sliver correction, then removals.
    Removals go last because the masks were drawn in the rotated frame.
    """
    total = (angle_deg + edit.get("extra_rot", 0.0)) % 360
    out = apply_pending_display(disp, total, flip_lr)
    return apply_removals(out, edit.get("removals", []))


def build_edit_canvas(disp, angle_deg, flip_lr, edit, title="", figsize=(7, 7)):
    """Figure + axes for mo.ui.matplotlib, at thumbnail pixel coordinates.

    Existing removals are shown as a translucent overlay rather than as zeroed
    pixels, so you can see what you already cut and undo it if it was wrong.
    Selections map to this axes' coordinate space, which imshow makes equal to
    pixel indices, so selection_to_mask() needs no rescaling.
    """
    total = (angle_deg + edit.get("extra_rot", 0.0)) % 360
    base = apply_pending_display(disp, total, flip_lr)
    base = base if base.ndim == 2 else base[..., 0]

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.imshow(base, cmap="gray", vmin=0, vmax=255, interpolation="nearest")

    removals = edit.get("removals", [])
    if removals:
        hw = base.shape[:2]
        combined = np.zeros(hw, bool)
        for m in removals:
            combined |= upscale_mask(m, hw)
        overlay = np.zeros(hw + (4,), np.float32)
        overlay[combined] = (1.0, 0.15, 0.15, 0.45)
        ax.imshow(overlay, interpolation="nearest")

    h, w = base.shape[:2]
    ax.axvline(w / 2, color="#4af", lw=1.2, alpha=0.9)
    ax.axhline(h / 2, color="#4af", lw=0.7, alpha=0.4)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


def save_removal_masks(edits, out_path):
    """Sidecar for the removal masks; the CSV manifest holds only scalars."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {}
    for i, e in edits.items():
        for j, m in enumerate(e.get("removals", [])):
            payload[f"s{i}_m{j}"] = np.packbits(m)
            payload[f"s{i}_m{j}_shape"] = np.array(m.shape)
    np.savez_compressed(out_path, **payload)
    print(f"wrote {len(payload) // 2} removal masks -> {out_path}")
    return out_path


def load_removal_masks(path, n_sections):
    edits = empty_edits(n_sections)
    if not os.path.exists(path):
        return edits
    with np.load(path) as z:
        for k in sorted(k for k in z.files if not k.endswith("_shape")):
            i = int(k.split("_")[0][1:])
            shape = tuple(z[f"{k}_shape"])
            bits = np.unpackbits(z[k])[: int(np.prod(shape))]
            edits[i]["removals"].append(bits.astype(bool).reshape(shape))
    return edits

# --------------------------------------------------------------------------
# Decisions manifest
# --------------------------------------------------------------------------


def collect_decisions(section_paths, controls, edits=None, brain_id=None,
                      slide_num=None):
    """Button state plus interactive edits, one row per section.
 
    rotate_ccw_deg is the buttons alone and extra_rot the sliver corrections;
    total_rot_ccw_deg is what apply_decisions actually uses. Keeping all three
    means a surprising result can be traced back to which control produced it.
    """
    rows = []
    for i, p in enumerate(section_paths):
        v = controls.value[i]
        e = (edits or {}).get(i, {})
        btn = float(pending_angle(v))
        extra = float(e.get("extra_rot", 0.0))
        rs = _norm_removals(e.get("removals", []))
        rows.append(
            {
                "idx": i,
                "brain_id": str(brain_id) if brain_id is not None else None,
                "slide_num": int(slide_num[-2:]) if slide_num is not None else None,
                "section": section_number(p),
                "filename": os.path.basename(p),
                "path": p,
                "status": v["status"],
                "flip_lr": bool(v["flip_lr"]),
                "rotate_ccw_deg": btn,
                "extra_rot": round(extra, 3),
                "total_rot_ccw_deg": round((btn + extra) % 360, 3),
                "n_removals": len(rs),
                "removal_px": int(sum(int(r["mask"].sum()) for r in rs)),
                "unanchored_removals": int(sum(1 for r in rs if not np.isfinite(r["at"]))),
            }
        )
    return pl.DataFrame(rows)
 



def save_decisions(decisions, out_path):
    """Persist the manifest so a review can be resumed and a rerun reproduced."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if out_path.endswith(".json"):
        with open(out_path, "w") as f:
            json.dump(decisions.to_dicts(), f, indent=2)
    else:
        decisions.write_csv(out_path)
    print(f"wrote {decisions.height} decisions -> {out_path}")
    return out_path


def load_decisions(path):
    if path.endswith(".json"):
        with open(path) as f:
            return pl.DataFrame(json.load(f))
    return pl.read_csv(path)


def _rotate_mask(mask, delta_deg):
    """Move a boolean mask between display frames. Nearest-neighbour only."""
    d = float(delta_deg) % 360
    if d == 0:
        return mask
    if d % 90 == 0:
        return np.ascontiguousarray(np.rot90(mask, k=int(d) // 90, axes=(0, 1)))
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), d, 1.0)
    return cv2.warpAffine(mask.astype(np.uint8), M, (w, h),
                          flags=cv2.INTER_NEAREST, borderValue=0).astype(bool)
 
 
def _norm_removals(removals):
    """Accept bare masks or {'mask', 'at'} dicts; always return dicts.
 
    'at' is the total display angle the lasso was drawn at. NaN means unknown,
    in which case the mask is used as-is and cannot be reprojected.
    """
    out = []
    for r in removals or []:
        if isinstance(r, dict):
            out.append({"mask": np.asarray(r["mask"], bool),
                        "at": float(r.get("at", np.nan))})
        else:
            out.append({"mask": np.asarray(r, bool), "at": float("nan")})
    return out
 
 
def _fit_to(mask, target_hw):
    """Center-crop or center-pad a mask to target dims.
 
    pad_to_common() puts every thumbnail on a shared canvas, so a mask drawn
    there is larger than the section's own downsampled extent. Cropping back is
    what keeps mask and image aligned at full resolution.
    """
    h, w = mask.shape
    th, tw = int(target_hw[0]), int(target_hw[1])
    if (h, w) == (th, tw):
        return mask
    out = mask
    if h > th or w > tw:
        top, left = max(0, (h - th) // 2), max(0, (w - tw) // 2)
        out = out[top:top + min(th, h), left:left + min(tw, w)]
    if out.shape != (th, tw):
        dh, dw = th - out.shape[0], tw - out.shape[1]
        out = np.pad(out, ((dh // 2, dh - dh // 2), (dw // 2, dw - dw // 2)))
    return out
 
 
def apply_removals(arr, removals, final_angle=None, verbose=False):
    """Zero every selected region on (C, Y, X), (Y, X), or (Y, X, 3).
 
    Masks are at thumbnail scale in the frame they were drawn in. Pass
    final_angle to reproject them onto the current frame; leave it None to use
    them as drawn. Always runs AFTER the rotation/flip, never before.
    """
    rs = _norm_removals(removals)
    if not rs:
        return arr
 
    a = np.asarray(arr)
    chan_last = a.ndim == 3 and a.shape[-1] in (3, 4)
    hw = a.shape[:2] if chan_last else a.shape[-2:]
 
    combined = np.zeros(hw, bool)
    for r in rs:
        m = r["mask"]
        if final_angle is not None and np.isfinite(r["at"]):
            m = _rotate_mask(m, final_angle - r["at"])
        combined |= upscale_mask(_fit_to(m, hw), hw) if m.shape != hw else m
 
    out = a.copy()
    if chan_last:
        out[combined, :] = 0
    else:
        out[..., combined] = 0
    if verbose:
        print(f"removed {int(combined.sum())} px ({100 * combined.mean():.2f}%)")
    return out
 
 
def removals_to_full_mask(removals, full_hw, final_angle=None, factor=0.1,
                          verbose=False):
    """Combine removal masks and upscale to the TRANSFORMED full-res frame.
 
    full_hw must be the shape after the warp, since 90 and 270 swap the axes.
    """
    rs = _norm_removals(removals)
    if not rs:
        return None
    H, W = int(full_hw[0]), int(full_hw[1])
    # the section's own thumbnail extent, before pad_to_common widened it
    own_hw = (int(H * factor + 0.5), int(W * factor + 0.5))
 
    combined, n_stale, n_cropped = None, 0, 0
    for r in rs:
        m = r["mask"]
        if final_angle is not None and np.isfinite(r["at"]):
            delta = final_angle - r["at"]
            if abs(delta) > 1e-6:
                m = _rotate_mask(m, delta)
                n_stale += 1
        if m.shape != own_hw:
            m = _fit_to(m, own_hw)
            n_cropped += 1
        combined = m if combined is None else (combined | m)
 
    if verbose and n_stale:
        print(f"       reprojected {n_stale} mask(s) onto the final angle")
    if verbose and n_cropped:
        print(f"       cropped {n_cropped} mask(s) off the pad_to_common canvas")
    return upscale_mask(combined, (H, W))



# --------------------------------------------------------------------------
# Apply at full resolution
# --------------------------------------------------------------------------


def _compose_flip_rotate(h, w, angle_deg, flip_lr):
    """Single 2x3 affine for flip-then-rotate, so warping happens once."""
    F = np.eye(3, dtype=np.float64)
    if flip_lr:
        F[0, 0] = -1.0
        F[0, 2] = w - 1.0
    R = np.eye(3, dtype=np.float64)
    R[:2] = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), angle_deg % 360, 1.0)
    return (R @ F)[:2]  # flip applied first, then rotation


def apply_transform_full_res(arr, angle_deg, flip_lr, verbose=False):
    """Apply the reviewed transform to a (C, Y, X) uint16 array.

    Multiples of 90 take an exact path with no interpolation at all. Other
    angles compose flip and rotation into one warpAffine, because two
    sequential warps would blur twice for no reason.
    """
    a = angle_deg % 360
    if a == 0 and not flip_lr:
        return arr

    if a % 90 == 0:
        out = np.flip(arr, axis=-1) if flip_lr else arr
        k = int(a) // 90
        if k:
            out = np.rot90(out, k=k, axes=(-2, -1))
        return np.ascontiguousarray(out)

    c, h, w = arr.shape
    M = _compose_flip_rotate(h, w, a, flip_lr)
    planes = []
    for ch in range(c):
        warped = cv2.warpAffine(
            arr[ch].astype(np.float32), M, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        # Defensive: keeps any interpolation undershoot from wrapping modulo 2**16.
        planes.append(np.clip(warped, 0, 65535).round().astype(np.uint16))
    out = np.stack(planes, axis=0)
    if verbose:
        print(f"warped {arr.shape} by {a}° flip={flip_lr}, "
              f"range {out.min()}-{out.max()}")
    return out


def apply_decisions(decisions, out_dir, edits=None, dry_run=True, overwrite=False,
                    factor=0.1, verbose=True):
    """Write corrected full-res TIFFs for every kept section.
 
    Order is flip -> rotate(total) -> zero removals, matching edited_preview()
    exactly. Removals come last because the masks were drawn in the rotated
    frame; applying them first would cut the wrong pixels.
 
    Existing outputs are left alone unless overwrite=True, so a re-fired
    reactive cell writes nothing rather than redoing ~200 MB per section.
    """
    os.makedirs(out_dir, exist_ok=True)
    written, skipped = [], []
    edits = edits or {}
 
    for row in decisions.iter_rows(named=True):
        if row["status"] != "keep":
            skipped.append(row["filename"])
            if verbose:
                print(f"SKIP   {row['filename']}  (status={row['status']})")
            continue
 
        stem, ext = os.path.splitext(row["filename"])
        out_path = os.path.join(out_dir, f"{stem}_reviewed{ext}")
        angle = float(row["total_rot_ccw_deg"])
        flip = bool(row["flip_lr"])
        e = edits.get(row["idx"], {})
        removals = e.get("removals", [])
 
        if os.path.exists(out_path) and not overwrite and not dry_run:
            skipped.append(row["filename"])
            if verbose:
                print(f"EXISTS {row['filename']}  (pass overwrite=True to replace)")
            continue
 
        if verbose:
            print(f"APPLY  {row['filename']}  rot={angle}° "
                  f"(btn {row['rotate_ccw_deg']} {row['extra_rot']:+}) "
                  f"flip={flip}  cuts={len(removals)}"
                  + ("  [dry run]" if dry_run else ""))
        if dry_run:
            written.append(out_path)
            continue
 
        with tifffile.TiffFile(row["path"]) as tif:
            full = tif.series[0].asarray()
        fixed = apply_transform_full_res(full, angle, flip, verbose=verbose)
 
        fm = removals_to_full_mask(removals, fixed.shape[-2:], final_angle=angle,
                                   factor=factor, verbose=verbose)
        if fm is not None:
            fixed = fixed.copy()          # may still alias `full` when angle==0
            fixed[..., fm] = 0
            if verbose:
                print(f"       zeroed {int(fm.sum())} px ({100 * fm.mean():.2f}%)")
 
        tifffile.imwrite(out_path, fixed, photometric="minisblack",
                         metadata={"axes": "CYX"})
        written.append(out_path)
        del full, fixed
 
    if verbose:
        print(f"\n{len(written)} written, {len(skipped)} skipped: {skipped}")
    return written, skipped
