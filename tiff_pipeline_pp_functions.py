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

'''preprocessing functions called in tiff_pipeline_cajal.py'''

def downsample_tiff(slide_path):
    with tifffile.TiffFile(slide_path) as _tif:
        full = _tif.series[0].asarray()  # (2, 15223, 30449), CYX, uint16

    down_area = np.stack(
        [cv2.resize(full[_c], None, fx=0.1, fy=0.1,
                    interpolation=cv2.INTER_AREA)
        for _c in range(full.shape[0])],
        axis=0,
    )
    return down_area

def remove_bright_artifact(
    img,
    detect_channel=None,
    threshold=None,
    min_area_frac=0.002,
    require_border=True,
    min_aspect=3.0, #dropped from 5.0 to handle 3.1 aspect of 129_02, S010 w/ only three slides
    max_thin_frac=0.15,
    min_span_frac=0.4,
    min_bright_ratio=1.15,
    halo_ratio=1.5,      # NEW: grow into pixels above halo_ratio * background level
    max_grow=30,         # NEW: growth limited to this many px from the core (prevents tissue leak)
    dilate=3,
    fill="zero",
    verbose=False,
    return_mask=False,
):
    def _disk(r):
        _l = np.arange(-r, r + 1)
        _x, _y = np.meshgrid(_l, _l)
        return (_x ** 2 + _y ** 2) <= r ** 2

    arr = np.asarray(img)
    is_stack = arr.ndim == 3
    det = (arr.max(axis=0) if detect_channel is None else arr[detect_channel]) if is_stack else arr
    det = det.astype(np.float32)
    H, W = det.shape

    thr = threshold_otsu(det) if threshold is None else float(threshold)
    lbl = label(det > thr)
    props = regionprops(lbl, intensity_image=det)
    mask = np.zeros(det.shape, bool)

    big = [p for p in props if p.area >= int(min_area_frac * det.size)]
    if not big:
        if verbose:
            print("no bright components; image unchanged")
        return (arr.copy(), mask) if return_mask else arr.copy()

    border_labels = set(lbl[0, :]) | set(lbl[-1, :]) | set(lbl[:, 0]) | set(lbl[:, -1])
    border_labels.discard(0)
    means = np.array([p.intensity_mean for p in big])

    best, best_score = None, -1.0
    for _i, p in enumerate(big):
        _minr, _minc, _maxr, _maxc = p.bbox
        _bh, _bw = (_maxr - _minr) / H, (_maxc - _minc) / W
        aspect = p.axis_major_length / max(p.axis_minor_length, 1e-6)
        span, thin = max(_bh, _bw), min(_bh, _bw)
        _others = np.delete(means, _i)
        _ref = np.median(_others) if _others.size else np.median(det[lbl != p.label])
        ratio = p.intensity_mean / max(_ref, 1e-6)

        _reasons = []
        if require_border and p.label not in border_labels:
            _reasons.append("not on border")
        if aspect < min_aspect:
            _reasons.append(f"aspect {aspect:.1f}<{min_aspect}")
        if thin > max_thin_frac:
            _reasons.append(f"too wide {thin:.2f}>{max_thin_frac}")
        if span < min_span_frac:
            _reasons.append(f"span {span:.2f}<{min_span_frac}")
        if ratio < min_bright_ratio:
            _reasons.append(f"not brighter {ratio:.2f}<{min_bright_ratio}")

        if _reasons:
            if verbose:
                print(f"reject {p.label}: " + ", ".join(_reasons))
            continue
        if verbose:
            print(f"candidate {p.label}: aspect={aspect:.1f} span={span:.2f} "
                  f"thin={thin:.2f} bright_ratio={ratio:.2f}")
        _score = p.area * float(p.intensity_mean)
        if _score > best_score:
            best_score, best = _score, p.label

    if best is None:
        if verbose:
            print("no artifact found; image unchanged")
        return (arr.copy(), mask) if return_mask else arr.copy()

    core = lbl == best

    # --- NEW: hysteresis growth to capture the soft halo around the bar ---
    if halo_ratio:
        _bg = np.median(det[det <= thr])              # dark background level
        _low = _bg * halo_ratio
        _reach = ndi.binary_dilation(core, structure=_disk(max_grow))
        mask = ndi.binary_propagation(core, mask=(det > _low) & _reach)
        if verbose:
            print(f"halo growth: bg={_bg:.0f} low_thr={_low:.0f} "
                  f"core={int(core.sum())} -> grown={int(mask.sum())} px")
    else:
        mask = core

    if dilate:
        mask = ndi.binary_dilation(mask, structure=_disk(dilate))
    mask = ndi.binary_fill_holes(mask)

    out = arr.astype(np.float32) if fill == "nan" else arr.copy()

    def _apply(plane):
            if fill == "zero":
                plane[mask] = 0
            elif fill == "nan":
                plane[mask] = np.nan
            else:
                plane[mask] = np.percentile(plane[~mask], bg_pct) if (~mask).any() else 0
            return plane

    if is_stack:
        for _c in range(out.shape[0]):
            out[_c] = _apply(out[_c])
    else:
        out = _apply(out)
    return (out, mask) if return_mask else out

def zero_background(
    img,
    detect_channel=0,
    method="li",
    smooth_sigma=2,
    open_radius=5,
    min_size_frac=0.0005,
    close_radius=8,
    dilate=1,
    exclude_mask=None,      # NEW: region to ignore when thresholding (e.g. artifact_mask)
    exclude_zeros=True,     # NEW: auto-ignore blanked-out (exactly zero) regions
    max_mask_frac=0.75,     # NEW: sanity check on the resulting mask
    clip_pct=99.5,
    return_mask=False,
):
    '''removes background noise in slide image'''
    def _disk(r):
        _l = np.arange(-r, r + 1)
        _x, _y = np.meshgrid(_l, _l)
        return (_x ** 2 + _y ** 2) <= r ** 2

    arr = np.asarray(img)
    is_stack = arr.ndim == 3
    _raw = (arr.max(axis=0) if detect_channel is None else arr[detect_channel]) if is_stack else arr

    # build the "valid" region BEFORE smoothing (smoothing blurs the zero edge)
    valid = np.ones(_raw.shape, bool)
    if exclude_mask is not None:
        valid &= ~exclude_mask
    if exclude_zeros:
        _z = _raw == 0
        if _z.any():
            _pad = int(3 * smooth_sigma) + 1
            valid &= ~ndi.binary_dilation(_z, structure=_disk(_pad))

    det = ndi.gaussian_filter(_raw.astype(np.float32), smooth_sigma)

    if isinstance(method, (int, float)):
        thr = float(method)
    else:
        _vals = det[valid] if 'valid' in dir() else det.ravel()
        _hi = np.percentile(_vals, clip_pct)      # clip_pct=99.5 as a new parameter
        _vals = np.minimum(_vals, _hi)
        _fn = {"otsu": threshold_otsu, "triangle": threshold_triangle,
               "li": threshold_li}[method]
        thr = _fn(_vals, tolerance=1.0) if method == "li" else _fn(_vals)
        print(f"threshold ({method}) = {thr:.1f} "
              f"[clipped at p{clip_pct}={_hi:.1f}], "
              f"mask frac = {(det > thr).mean():.3f}")

    mask = (det > thr) & valid                   # excluded region is never tissue

    if mask.mean() > max_mask_frac:
        print(f"WARNING: mask_frac={mask.mean():.3f} > {max_mask_frac}; "
              f"threshold {thr:.1f} may be below background")

    if open_radius:
        mask = ndi.binary_opening(mask, structure=_disk(open_radius))
    lbl, n = ndi.label(mask)
    if n > 0:
        _sizes = np.bincount(lbl.ravel())
        _keep = _sizes >= int(min_size_frac * det.size)
        _keep[0] = False
        mask = _keep[lbl]
    if close_radius:
        mask = ndi.binary_closing(mask, structure=_disk(close_radius))
    mask = ndi.binary_fill_holes(mask)
    if dilate:
        mask = ndi.binary_dilation(mask, structure=_disk(dilate))
    mask &= valid                                # re-apply after morphology

    print(f"threshold ({method}) = {thr:.1f}, mask frac = {mask.mean():.3f}")

    out = arr.copy()
    if is_stack:
        out[:, ~mask] = 0
    else:
        out[~mask] = 0
    return (out, mask) if return_mask else out

def repair_cracks(mask, max_gap=30, piece_max_frac=0.85,
                  combined_range=(0.70, 1.35), verbose=True, return_stats=False):
    def _disk(r):
        _l = np.arange(-r, r + 1)
        _x, _y = np.meshgrid(_l, _l)
        return (_x ** 2 + _y ** 2) <= r ** 2

    lbl = label(mask, connectivity=2)
    props = regionprops(lbl)
    areas = np.array([p.area for p in props], float)
    typical = np.median(areas)
    cands = [p for p in props if p.area < piece_max_frac * typical]
    if verbose:
        print(f"typical={int(typical)}  candidates={[(int(p.label), int(p.area)) for p in cands]}")
    if len(cands) < 2:
        print("  -> fewer than 2 undersized pieces; nothing to join")
        return (mask.copy(), []) if return_stats else mask.copy()

    pairs = []
    for _i in range(len(cands)):
        for _j in range(_i + 1, len(cands)):
            a, b = cands[_i], cands[_j]
            combined = a.area + b.area
            if not (combined_range[0] * typical <= combined <= combined_range[1] * typical):
                if verbose:
                    print(f"  {a.label}+{b.label}: REJECT combined={combined/typical:.2f}x")
                continue
            _r0 = max(min(a.bbox[0], b.bbox[0]) - (max_gap + 2), 0)
            _c0 = max(min(a.bbox[1], b.bbox[1]) - (max_gap + 2), 0)
            _r1 = min(max(a.bbox[2], b.bbox[2]) + (max_gap + 2), mask.shape[0])
            _c1 = min(max(a.bbox[3], b.bbox[3]) + (max_gap + 2), mask.shape[1])
            _sub = lbl[_r0:_r1, _c0:_c1]
            _A, _B = _sub == a.label, _sub == b.label
            gap = float(ndi.distance_transform_edt(~_A)[_B].min())
            if gap > max_gap:
                if verbose:
                    print(f"  {a.label}+{b.label}: REJECT gap={gap:.1f} > max_gap={max_gap}")
                continue
            pairs.append((abs(combined - typical) / typical, gap, a.label, b.label,
                          (_r0, _c0, _r1, _c1), combined))
    pairs.sort(key=lambda t: (t[0], t[1]))

    out = mask.copy()
    used, stats = set(), []
    for _score, gap, la, lb, (_r0, _c0, _r1, _c1), combined in pairs:
        if la in used or lb in used:
            continue
        _sub = lbl[_r0:_r1, _c0:_c1]
        _pair = (_sub == la) | (_sub == lb)
        _others = (_sub != 0) & ~_pair
        _forbidden = ndi.binary_dilation(_others, structure=_disk(1))
        _joined, _rad = None, None
        for _rad in range(int(np.ceil(gap / 2)) + 1, int(np.ceil(gap)) + 12):
            _bridged = ndi.binary_closing(_pair, structure=_disk(_rad))
            _cand = _pair | (_bridged & ~_pair & ~_forbidden)
            if label(_cand, connectivity=2).max() == 1:
                _joined = _cand
                break
        if _joined is None:
            print(f"  {la}+{lb}: REJECT could not bridge even at rad={_rad} (gap={gap:.1f})")
            continue
        out[_r0:_r1, _c0:_c1] |= _joined
        used.update([la, lb])
        stats.append((int(la), int(lb), float(gap), int(combined), round(combined / typical, 2)))
        print(f"  JOIN {la}+{lb}: gap={gap:.1f}px rad={_rad} combined={combined/typical:.2f}x")
    return (out, stats) if return_stats else out

def _split_component(comp, k):
    _dist = ndi.distance_transform_edt(comp)
    _best, _best_n = None, 0
    for _frac in np.linspace(0.30, 0.95, 40):
        _core = _dist > _frac * _dist.max()
        _m, _n = ndi.label(_core)
        if _n >= 2:
            _sizes = np.bincount(_m.ravel())
            _keep = np.where(_sizes >= max(20, 0.02 * comp.sum()))[0]
            _keep = _keep[_keep != 0]
            if _keep.size >= 2:
                _m2 = np.zeros_like(_m)
                for _i, _lab in enumerate(_keep, start=1):
                    _m2[_m == _lab] = _i
                if _keep.size >= _best_n:
                    _best, _best_n = _m2, _keep.size
                if _keep.size >= k:
                    break
    if _best is None or _best_n < 2:
        return None
    return watershed(-_dist, markers=_best, mask=comp, watershed_line=True)

def separate_tissues(mask, merge_ratio=1.5, cut_width=5, verbose=False, return_stats=False):
    def _disk(r):
        _l = np.arange(-r, r + 1)
        _x, _y = np.meshgrid(_l, _l)
        return (_x ** 2 + _y ** 2) <= r ** 2

    lbl = label(mask)
    props = regionprops(lbl)
    if len(props) < 2:
        return (mask.copy(), []) if return_stats else mask.copy()

    areas = np.array([p.area for p in props], float)
    typical = np.median(areas[areas <= np.median(areas) * merge_ratio])
    if not np.isfinite(typical) or typical <= 0:
        typical = np.median(areas)

    out = mask.copy()
    stats = []
    for p in props:
        k = int(round(p.area / typical))
        if p.area < merge_ratio * typical or k < 2:
            continue
        comp = lbl == p.label
        split = _split_component(comp, k)
        if split is None:
            if verbose:
                print(f"comp {p.label}: area={int(p.area)} k={k} -> could not split, left intact")
            continue
        out[comp] = False
        out[split > 0] = True

        # widen the watershed line so it survives 8-connected labeling
        _cut = comp & ~(split > 0)
        if _cut.any() and cut_width > 1:
            out &= ~ndi.binary_dilation(_cut, structure=_disk(max(1, cut_width // 2)))

        _n_after = label(out & comp, connectivity=2).max()   # verify with 8-connectivity
        stats.append((int(p.label), int(p.area), k, int(_n_after)))
        if verbose:
            print(f"comp {p.label}: area={int(p.area)} (~{p.area/typical:.1f}x typical) "
                  f"k={k} -> {_n_after} components after cut"
                  + ("" if _n_after >= 2 else "   <-- CUT FAILED, still connected"))
    return (out, stats) if return_stats else out

def remove_debris(
    img,
    mask,
    size_frac=0.15,          # large compact components are tissue outright
    max_aspect=4.0,          # main filter: elongated -> debris
    min_solidity=0.3,        # main filter: wispy -> debris
    rescue_dist=9,           # a fragment this close to a section is rejoined (0 disables)
    rescue_max_aspect=2.5,   # only ROUNDISH small blobs are eligible for rescue
    verbose=False,
    return_mask=False,
):
    '''removes any remaining artificats/debris'''
    def _disk(r):
        _l = np.arange(-r, r + 1)
        _x, _y = np.meshgrid(_l, _l)
        return (_x ** 2 + _y ** 2) <= r ** 2

    lbl = label(mask)
    props = regionprops(lbl)
    if not props:
        return (img.copy(), mask) if return_mask else img.copy()

    areas = np.array([p.area for p in props], dtype=float)
    ref = np.median(areas[areas >= np.percentile(areas, 50)])

    tissue, candidates = [], []
    for p in props:
        aspect = p.axis_major_length / max(p.axis_minor_length, 1e-6)
        compact = aspect <= max_aspect and p.solidity >= min_solidity
        if p.area >= size_frac * ref and compact:
            tissue.append(p.label)
        elif rescue_dist and aspect <= rescue_max_aspect and p.solidity >= min_solidity:
            candidates.append(p.label)        # small + roundish -> maybe a fragment
        elif verbose:
            print(f"drop {p.label}: area={int(p.area)} aspect={aspect:.1f} "
                  f"sol={p.solidity:.2f} (shape)")

    if rescue_dist and candidates and tissue:
        near = ndi.binary_dilation(np.isin(lbl, tissue), structure=_disk(rescue_dist))
        for _lab in candidates:
            if (near & (lbl == _lab)).any():
                tissue.append(_lab)
                if verbose:
                    print(f"rescue {_lab}: fragment adjacent to a section")
            elif verbose:
                print(f"drop {_lab}: small & isolated")

    clean_mask = np.isin(lbl, tissue) & mask
    out = img.copy()
    if img.ndim == 3:
        out[:, ~clean_mask] = 0
    else:
        out[~clean_mask] = 0
    return (out, clean_mask) if return_mask else out


def remove_whiskers(
        img,
        mask,
        width=5,                 # opening radius; severs protrusions thinner than ~2*width px
        min_whisker_area=50,     # ignore tiny boundary specks, 
        min_whisker_aspect=2.5,  # only trim ELONGATED protrusions, never compact bumps
        verbose=False,
        return_mask=False,
    ):
        def _disk(r):
            _l = np.arange(-r, r + 1)
            _x, _y = np.meshgrid(_l, _l)
            return (_x ** 2 + _y ** 2) <= r ** 2

        se = _disk(width)
        out_mask = mask.copy()
        lbl = label(mask)
        for p in regionprops(lbl):
            comp = lbl == p.label
            core = ndi.binary_opening(comp, structure=se)   # thin parts vanish, body stays
            removed = comp & ~core
            if not removed.any():
                continue
            rlbl = label(removed)
            for rp in regionprops(rlbl):
                aspect = rp.axis_major_length / max(rp.axis_minor_length, 1e-6)
                if rp.area >= min_whisker_area and aspect >= min_whisker_aspect:
                    out_mask &= ~(rlbl == rp.label)
                    if verbose:
                        print(f"trim whisker on section {p.label}: "
                              f"area={int(rp.area)} aspect={aspect:.1f}")

        out = img.copy()
        if img.ndim == 3:
            out[:, ~out_mask] = 0
        else:
            out[~out_mask] = 0
        return (out, out_mask) if return_mask else out

def plot_pp_steps(dewhiskered, dewhiskered_mask, tissue_mask, artifact_mask, final, final_mask,
                  down_area, cleaned, foreground, channel, slide_out_folder, slide_name, savefig=False):
    
    _vmax = np.percentile(dewhiskered[channel][dewhiskered_mask], 99)
    _debris = tissue_mask & ~final_mask
    _whiskers = final_mask & ~dewhiskered_mask

    _fig, _ax = plt.subplots(4, 3, figsize=(15, 16))

    # Row 0: bright artifact removal
    _ax[0, 0].imshow(down_area[channel], cmap="gray",
                     vmax=np.percentile(down_area[0], 99.5))
    _ax[0, 0].set_title(f"original ch{channel} (artifact)")
    _ax[0, 1].imshow(down_area[channel], cmap="gray", vmax=_vmax)
    _ax[0, 1].imshow(artifact_mask, cmap="Reds", alpha=0.4)
    _ax[0, 1].set_title("detected artifact mask")
    _ax[0, 2].imshow(cleaned[channel], cmap="gray",
                     vmin=np.percentile(cleaned[0], 1), vmax=_vmax)
    _ax[0, 2].set_title(f"cleaned ch{channel}")

    # Row 1: background removal
    _ax[1, 0].imshow(cleaned[channel], cmap="gray",
                     vmin=np.percentile(cleaned[channel], 1), vmax=_vmax)
    _ax[1, 0].set_title("cleaned ch0 (grid + noise)")
    _ax[1, 1].imshow(tissue_mask, cmap="gray")
    _ax[1, 1].set_title("tissue mask")
    _ax[1, 2].imshow(foreground[channel], cmap="gray", vmax=_vmax)
    _ax[1, 2].set_title("background zeroed")

    # Row 2: debris removal
    _ax[2, 0].imshow(foreground[channel], cmap="gray", vmax=_vmax)
    _hi = np.zeros((*_debris.shape, 4))
    _hi[ndi.binary_dilation(_debris, structure=np.ones((3, 3)))] = [1, 0, 0, 0.55]
    _ax[2, 0].imshow(_hi)
    for _p in regionprops(label(_debris)):
        _r0, _c0, _r1, _c1 = _p.bbox
        _pad = 5
        _ax[2, 0].plot(
            [_c0 - _pad, _c1 + _pad, _c1 + _pad, _c0 - _pad, _c0 - _pad],
            [_r0 - _pad, _r0 - _pad, _r1 + _pad, _r1 + _pad, _r0 - _pad],
            color="red", lw=1.0,
        )
    _ax[2, 0].set_title("before — dropped debris (red)")
    _ax[2, 1].imshow(final_mask, cmap="gray")
    _ax[2, 1].set_title("filtered mask")
    _ax[2, 2].imshow(final[channel], cmap="gray", vmax=_vmax)
    _ax[2, 2].set_title("debris removed")

    # Row 3: whisker removal
    _ax[3, 0].imshow(final[channel], cmap="gray", vmax=_vmax)
    _hiw = np.zeros((*_whiskers.shape, 4))
    _hiw[ndi.binary_dilation(_whiskers, structure=np.ones((3, 3)))] = [1, 0, 0, 0.55]
    _ax[3, 0].imshow(_hiw)
    for _p in regionprops(label(_whiskers)):
        _r0, _c0, _r1, _c1 = _p.bbox
        _pad = 5
        _ax[3, 0].plot(
            [_c0 - _pad, _c1 + _pad, _c1 + _pad, _c0 - _pad, _c0 - _pad],
            [_r0 - _pad, _r0 - _pad, _r1 + _pad, _r1 + _pad, _r0 - _pad],
            color="red", lw=1.0,
        )
    _ax[3, 0].set_title("before — dropped whiskers (red)")
    _ax[3, 1].imshow(dewhiskered_mask, cmap="gray")
    _ax[3, 1].set_title("dewhiskered mask")
    _ax[3, 2].imshow(dewhiskered[channel], cmap="gray", vmax=_vmax)
    _ax[3, 2].set_title("whiskers removed")

    for _a in _ax.ravel():
        _a.axis("off")
    plt.tight_layout()
    if savefig:
        plt.savefig(f'{slide_out_folder}{slide_name}_preprocessed_all_steps.png')
    plt.close(_fig)          # free the figure so it does not accumulate across slides
    return None

# Annotation/QC, with bridge inputs
def annotate_sections(
        img,
        mask,
        custom_order=None,       # None -> 1..n in reading order; else a list (one int per section, in reading order)
        channel=0,               # which channel to show as the grayscale base
        color=(255, 0, 0),       # RGB red
        font_scale=None,         # None -> auto-scaled to section size
        thickness=None,          # None -> auto
        return_order=False,
        return_labels=False,     # NEW: also return the section-numbered label mask
        savefig=False,           # True -> plot and save the annotated image
        slide_out_folder=None,   # output folder for the saved figure
        slide_name=None,         # slide name used in the saved figure filename
        save_outputs=False,      # NEW: write bridge-pipeline inputs to a subdirectory
        raw_slide_path=None,     # NEW: full-res source, used to record mask->raw scale
        subdir="bridge_inputs",  # NEW: subdirectory name inside slide_out_folder
    ):
        '''adds number label to each section, assumes top->bottom, left->right ordering'''
        def _reading_order(props):
            heights = np.array([p.bbox[2] - p.bbox[0] for p in props])
            row_tol = 0.5 * np.median(heights)          # a y-jump bigger than this starts a new row
            items = sorted(props, key=lambda p: p.centroid[0])
            rows, cur = [], [items[0]]
            for p in items[1:]:
                if p.centroid[0] - cur[-1].centroid[0] > row_tol:
                    rows.append(cur); cur = [p]
                else:
                    cur.append(p)
            rows.append(cur)
            ordered = []
            for row in rows:
                ordered.extend(sorted(row, key=lambda p: p.centroid[1]))
            return ordered

        out_root = Path(slide_out_folder) / subdir

        def _write_outputs(numbered_mask, sections):
            out_root.mkdir(parents=True, exist_ok=True)

            raw_shape = None
            if raw_slide_path is not None:
                with tifffile.TiffFile(raw_slide_path) as _tif:
                    raw_shape = list(_tif.pages[0].shape[-2:])

            meta = {
                "slide_name": slide_name,
                "raw_slide_path": str(raw_slide_path) if raw_slide_path else None,
                "raw_shape": raw_shape,
                "mask_shape": list(numbered_mask.shape),
                "n_sections": len(sections),
                "label_file": f"{slide_name}_section_labels.tiff",
                "sections": sections,
            }
            if raw_shape is not None:
                sy = raw_shape[0] / numbered_mask.shape[0]
                sx = raw_shape[1] / numbered_mask.shape[1]
                meta["scale_y"], meta["scale_x"] = sy, sx
                if abs(sy - sx) / max(sy, sx) > 0.02:
                    print(f"  WARNING: scale_y={sy:.4f} and scale_x={sx:.4f} differ by "
                          f">2% -- is the mask transposed relative to the raw slide?")
            else:
                print("  WARNING: raw_slide_path not given; mask->raw scale not recorded. "
                      "The bridge will have to assume a downsample factor.")

            tifffile.imwrite(
                out_root / meta["label_file"],
                numbered_mask.astype(np.uint16),
                photometric="minisblack",
            )
            (out_root / f"{slide_name}_bridge_meta.json").write_text(json.dumps(meta, indent=2))
            print(f"  wrote {len(sections)} section labels -> {out_root}")

        def _pack(rgb_out, order_out, labels_out):
            parts = [rgb_out]
            if return_order:
                parts.append(order_out)
            if return_labels:
                parts.append(labels_out)
            parts.append(out_root)
            return parts[0] if len(parts) == 1 else tuple(parts)

        base = (img[channel] if img.ndim == 3 else img).astype(np.float32)
        vmax = np.percentile(base[mask], 99) if mask.any() else max(base.max(), 1)
        norm = np.clip(base / max(vmax, 1e-6), 0, 1)
        rgb = np.repeat((norm * 255).astype(np.uint8)[..., None], 3, axis=2)

        lab_img = label(mask)                 # hoisted: props now carry a usable .label
        props = regionprops(lab_img)
        if not props:
            empty = np.zeros(lab_img.shape, np.uint16)
            if save_outputs:
                _write_outputs(empty, [])
            return _pack(rgb, [], empty)

        ordered = _reading_order(props)
        n = len(ordered)
        if custom_order is None:
            numbers = list(range(1, n + 1))
        else:
            if len(custom_order) != n:
                raise ValueError(f"custom_order has {len(custom_order)} entries but there are {n} sections")
            numbers = [int(v) for v in custom_order]
            # These become pixel VALUES in the label mask, so duplicates would merge
            # two sections into one and a 0 would erase one into the background.
            if len(set(numbers)) != n:
                raise ValueError(f"custom_order must be unique; got {numbers}")
            if min(numbers) < 1:
                raise ValueError(f"custom_order must be >= 1 (0 is background); got {numbers}")

        med_h = np.median([p.bbox[2] - p.bbox[0] for p in props])
        fs = font_scale if font_scale is not None else max(0.5, med_h / 120)
        th = thickness if thickness is not None else max(1, int(round(fs * 2)))
        font = cv2.FONT_HERSHEY_SIMPLEX

        # LUT maps raster-scan label -> reading-order section number in one pass.
        lut = np.zeros(lab_img.max() + 1, np.uint16)
        for prop, num in zip(ordered, numbers):
            lut[prop.label] = num
        numbered = lut[lab_img]

        order_map, sections = [], []
        for prop, num in zip(ordered, numbers):
            minr, minc = prop.bbox[0], prop.bbox[1]
            (_tw, _tht), _ = cv2.getTextSize(str(num), font, fs, th)
            org = (int(minc) + 3, int(minr) + _tht + 3)   # inside the top-left corner
            cv2.putText(rgb, str(num), org, font, fs, color, th, cv2.LINE_AA)
            order_map.append((num, (float(prop.centroid[0]), float(prop.centroid[1]))))
            sections.append({
                "number": int(num),
                "centroid_y": float(prop.centroid[0]),
                "centroid_x": float(prop.centroid[1]),
                "area_px": int(prop.area),
                "bbox": [int(v) for v in prop.bbox],
            })

        if savefig:
            _fig, _ax = plt.subplots(figsize=(12, 6))
            _ax.imshow(rgb)   # already RGB
            _ax.axis("off")
            _ax.set_title("section numbering")
            plt.savefig(Path(slide_out_folder) / f"{slide_name}_preprocessed_overview.png")
            plt.close(_fig)   # free the figure so it does not accumulate across slides

        if save_outputs:
            _write_outputs(numbered, sections)

        return _pack(rgb, order_map, numbered)

