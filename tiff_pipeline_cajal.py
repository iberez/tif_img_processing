import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
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

    return (
        BytesIO,
        Path,
        PdfReader,
        PdfWriter,
        canvas,
        convex_hull_image,
        glob,
        label,
        mo,
        ndi,
        np,
        os,
        pl,
        plt,
        psutil,
        re,
        regionprops,
        threshold_li,
        threshold_otsu,
        threshold_triangle,
        tifffile,
        time,
    )


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md("""
    # TIFF preprocessing pipeline
    **Workflow:** load slide and load 10X downsampled slide → sanity-check metadata → remove background artifact → remove background noise →
    remove any remaining debris/artifacts
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## preprocessing functions
    """)
    return


@app.cell
def _():
    import tiff_pipeline_pp_functions as tppf

    return (tppf,)


@app.cell
def _(psutil):
    # Shared machine: know what you are actually working with before you
    # allocate a 15 GB array.
    _vm = psutil.virtual_memory()
    host_state = {
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_percent": psutil.cpu_percent(interval=0.3),
        "ram_total_gb": round(_vm.total / 1e9, 1),
        "ram_available_gb": round(_vm.available / 1e9, 1),
    }
    host_state
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Slide preprocessing loop
    """)
    return


@app.cell
def _(Path, mo, os, re, time, tppf):
    _pp_loop_enable = False
    #single slide all preprocessing functions
    mo.stop(_pp_loop_enable is False, mo.md("*preprocessing loop disabled (`_pp_loop_enable` is False)*"))

    _brain_id = '129_02'
    _channel = 0
    _WORKDIR = Path(f"/bigdata/isaac/rabies_img_processing/{str(_brain_id)}")
    _process_all_toggle = True

    if _process_all_toggle is True:
        print ('processing all slices')
        _tiff_pattern = re.compile(r".*\.tif$")
        _all_slide_paths = [
            str(_WORKDIR / _f)
            for _f in os.listdir(_WORKDIR)
            if _tiff_pattern.match(_f)
        ]
    _num_slides = len(_all_slide_paths)
    print (f'found {_num_slides} slides to process')
    _start_time_all_slides = time.perf_counter()
    _counter  = 1
    #_slide_path = _all_slide_paths[0]
    for _slide_path in _all_slide_paths:
        _slide_name = str(_slide_path).split('/')[-1].split('.')[0]
        print (f'processing {_slide_name}')
        _slide_folder = str(_slide_path).rsplit("/", 1)[0] + "/"
        _slide_out_folder = _slide_folder + "preprocessed/"
        os.makedirs(_slide_out_folder, exist_ok=True)
        _raw_slide_path = str(Path(_slide_out_folder).parent) + '/' + _slide_name + '.tif'
        print (_slide_out_folder)
        _start_time = time.perf_counter()

        _down_area = tppf.downsample_tiff(_slide_path)

        _cleaned, _artifact_mask = tppf.remove_bright_artifact(_down_area, return_mask=True)

        _foreground, _tissue_mask = tppf.zero_background(_cleaned, return_mask=True)
        #handle severed tissues
        _repaired, _crack_stats = tppf.repair_cracks(_tissue_mask, max_gap=15, return_stats=True)
        #handle occasional case of overlapping tissues
        _separated, _split_stats = tppf.separate_tissues(_repaired, cut_width=5, return_stats=True)

        _final, _final_mask = tppf.remove_debris(_foreground, _separated, return_mask=True)

        _dewhiskered, _dewhiskered_mask = tppf.remove_whiskers(_final, _final_mask, return_mask=True)

        tppf.plot_pp_steps(_dewhiskered, _dewhiskered_mask, _tissue_mask, _artifact_mask, _final,_final_mask,
                      _down_area, _cleaned, _foreground, _channel, slide_out_folder=_slide_out_folder, slide_name=_slide_name, savefig=True)

        _annotated_output = tppf.annotate_sections(_dewhiskered, _dewhiskered_mask, return_order=True,
                                                        savefig=True,           # True -> plot and save the annotated image
                                                        slide_out_folder=_slide_out_folder,   # output folder for the saved figure
                                                        slide_name=_slide_name,                                                 
                                                        save_outputs=True,      # NEW: write bridge-pipeline inputs to a subdirectory
                                                        raw_slide_path=_raw_slide_path,     # NEW: full-res source 
                                                        subdir="bridge_inputs",  # NEW: subdirectory name inside slide_out_folder)
                                             )

        _end_time = time.perf_counter()
        _execution_time = _end_time - _start_time
        print (f'completed in {_execution_time:6f} seconds')

    _end_time_all_slides = time.perf_counter()
    _execution_time_all_slides = _end_time_all_slides - _start_time_all_slides

    print (f'completed preprocessing of brain {_brain_id}, containing {_num_slides} raw tiff slide img files, in {_execution_time_all_slides:6f} seconds')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preprocessing step-by-step (for debugging)
    """)
    return


@app.cell
def _(Path, mo):
    _pp_debug_enable = False
    #single slide all preprocessing functions
    mo.stop(_pp_debug_enable is False, mo.md("*preprocessing loop disabled (`_pp_debug_enable` is False)*"))


    WORKDIR = Path("/bigdata/isaac/rabies_img_processing/129_02")

    slide_browser = mo.ui.file_browser(
        initial_path=WORKDIR,
        filetypes=[".tif", ".tiff"],
        multiple=False,
        label="Slide",
    )

    process_all_toggle = mo.ui.switch(
        value=False,
        label="Process all .tif files in folder",
    )


    mo.vstack([slide_browser, process_all_toggle])
    return (slide_browser,)


@app.cell
def _(mo, os, slide_browser):
    mo.stop(not slide_browser.value, mo.md("*Select a slide to begin.*"))
    slide_path = slide_browser.path(index=0)

    slide_name = str(slide_path).split('/')[-1].split('.')[0]

    slide_folder = str(slide_path).rsplit("/", 1)[0] + "/"
    slide_out_folder = slide_folder + "preprocessed/"
    os.makedirs(slide_out_folder, exist_ok=True)


    print (slide_path)
    print (slide_name)
    print (slide_out_folder)
    return slide_name, slide_out_folder, slide_path


@app.cell
def _(slide_path, tifffile):
    #verify image is non pyramidal
    with tifffile.TiffFile(slide_path) as tif:
        s = tif.series[0]
        print(s.shape, s.axes, s.dtype, s.mpp)
        print("pyramidal:", s.is_pyramidal, [lv.shape for lv in s.levels])
        p = tif.pages.first
        print("tiled:", p.is_tiled, p.tile, "photometric:", p.photometric,
              "compression:", p.compression)
    return (tif,)


@app.cell
def _(tif):
    print (tif.pages.first.is_memmappable)
    return


@app.cell
def _(np, slide_path, tifffile):
    # Header-only read: metadata, not pixels. Cheap at any file size.
    with tifffile.TiffFile(slide_path) as _tif:
        _s = _tif.series[0]
        slide_info = {
            "shape": _s.shape,
            "dtype": str(_s.dtype),
            "axes": _s.axes,
            "n_levels": len(_s.levels),
            "n_pages": len(_tif.pages),
            "compression": str(_tif.pages[0].compression),
            "is_ome": _tif.is_ome,
            "decompressed_gb": round(
                np.prod(_s.shape) * _s.dtype.itemsize / 1e9, 2
            ),
        }
    slide_info
    return


@app.cell
def _(slide_path):
    slide_path
    return


@app.cell
def _():
    '''
    with tifffile.TiffFile(slide_path) as _tif:
        full = _tif.series[0].asarray()  # (2, 15223, 30449), CYX, uint16

    down_area = np.stack(
        [cv2.resize(full[_c], None, fx=0.1, fy=0.1,
                    interpolation=cv2.INTER_AREA)
         for _c in range(full.shape[0])],
        axis=0,
    )
    down_area.shape  # (2, 1522, 3045)
    '''
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _(slide_path, tppf):
    down_area = tppf.downsample_tiff(slide_path)
    return (down_area,)


@app.cell
def _(down_area, np, plt):
    _c0 = down_area[1]
    _lo, _hi = np.percentile(_c0, (1, 99.5))
    _h, _w = _c0.shape
    _scale = 8 / max(_h, _w)

    plt.figure(figsize=(_w * _scale, _h * _scale))
    plt.imshow(_c0, cmap="gray", vmin=_lo, vmax=max(_hi, _lo + 1))
    plt.title(f"down_area (10x) — channel 0 — {_c0.shape}")
    plt.axis("off")
    plt.gca()
    return


@app.cell
def _(down_area, tppf):
    cleaned, artifact_mask = tppf.remove_bright_artifact(down_area, fill="zero", min_aspect=3,return_mask=True, verbose=True)
    cleaned.shape
    return artifact_mask, cleaned


@app.cell
def _():
    channel = 0
    return (channel,)


@app.cell
def _(slide_name):
    slide_name
    return


@app.cell
def _(artifact_mask, channel, cleaned, down_area, np, plt):
    _vmax = np.percentile(cleaned[channel], 99.5)
    _fig, _ax = plt.subplots(1, 3, figsize=(15, 10))
    _ax[0].imshow(down_area[channel], cmap="gray",
                  vmax=np.percentile(down_area[0], 99.5))
    _ax[0].set_title(f"original ch{channel} (artifact)")
    _ax[1].imshow(down_area[channel], cmap="gray", vmax=_vmax)
    _ax[1].imshow(artifact_mask, cmap="Reds", alpha=0.4)
    _ax[1].set_title("detected artifact mask")
    _ax[2].imshow(cleaned[channel], cmap="gray", vmin=np.percentile(cleaned[0], 1), vmax=_vmax)
    _ax[2].set_title(f"cleaned ch{channel}")
    for _a in _ax:
        _a.axis("off")
    plt.tight_layout()
    plt.gcf()
    #plt.savefig(f'{slide_out_folder}{slide_name}_artifact_removed_comparison.png')
    return


@app.cell
def _(cleaned, ndi, np, threshold_li, threshold_otsu, threshold_triangle):
    _raw = cleaned[0]
    print(f"raw: shape={_raw.shape} dtype={_raw.dtype} "
          f"min={_raw.min()} max={_raw.max()}")
    _det = ndi.gaussian_filter(_raw.astype(np.float32), 2)
    print("det percentiles [1,25,50,75,90,99,99.9]:",
          np.round(np.percentile(_det, [1, 25, 50, 75, 90, 99, 99.9]), 1))

    # 1) does ANY threshold give a sane mask?
    for _name, _t in [("otsu", threshold_otsu(_det)),
                      ("li(no init)", threshold_li(_det, tolerance=1.0)),
                      ("li(init=otsu)", threshold_li(_det, tolerance=1.0,
                                                     initial_guess=threshold_otsu(_det))),
                      ("triangle", threshold_triangle(_det)),
                      ("pct 60", float(np.percentile(_det, 60)))]:
        print(f"  {_name:15s} thr={_t:10.1f}  mask_frac={float((_det > _t).mean()):.4f}")
    return


@app.cell
def _(cleaned, ndi, np, threshold_li, threshold_otsu):
    # 2) walk the morphology and find the step that zeroes it
    def _dsk(r):
        _l = np.arange(-r, r + 1); _x, _y = np.meshgrid(_l, _l)
        return (_x ** 2 + _y ** 2) <= r ** 2

    _det2 = ndi.gaussian_filter(cleaned[0].astype(np.float32), 2)
    _thr = threshold_li(_det2, tolerance=1.0, initial_guess=threshold_otsu(_det2))
    _m = _det2 > _thr
    print(f"threshold={_thr:.1f}")
    print(f"  after threshold   px={int(_m.sum()):9d} frac={_m.mean():.4f} "
          f"comps={ndi.label(_m)[1]}")
    _m = ndi.binary_opening(_m, structure=_dsk(10))
    print(f"  after opening(10) px={int(_m.sum()):9d} frac={_m.mean():.4f} "
          f"comps={ndi.label(_m)[1]}")
    _lbl, _n = ndi.label(_m)
    if _n:
        _sizes = np.bincount(_lbl.ravel())
        _cut = int(0.0005 * _det2.size)
        print(f"  size cutoff={_cut}, component sizes (top 5): "
              f"{sorted(_sizes[1:])[-5:]}")
        _keep = _sizes >= _cut; _keep[0] = False
        _m = _keep[_lbl]
    print(f"  after size filter px={int(_m.sum()):9d} frac={_m.mean():.4f}")
    return


@app.cell
def _(channel, cleaned, tppf):
    foreground, tissue_mask = tppf.zero_background(cleaned, 
                                                detect_channel=0,   # None -> detect on max-projection across channels
                                                method="li",     # "otsu" | "triangle" | "li" | a numeric threshold
                                                smooth_sigma=2,      # Gaussian blur before thresholding (noise suppression)
                                                open_radius=5,         # morphological opening: removes thin grid lines & speckle
                                                min_size_frac=0.0005,  # drop connected components smaller than this frac of image
                                                close_radius=8,        # close + fill holes so section interiors stay solid
                                                dilate=1,              # grow mask slightly to keep dim tissue edges
                                                return_mask=True)

    print ('foreground shape', foreground.shape)
    print ('tissue mask', foreground[channel][tissue_mask])
    return foreground, tissue_mask


@app.cell
def _(foreground, tissue_mask):
    foreground[0][tissue_mask]
    return


@app.cell
def _(channel, cleaned, foreground, np, plt, tissue_mask):
    _vmax = np.percentile(foreground[channel][tissue_mask], 99)
    _fig, _ax = plt.subplots(1, 3, figsize=(15, 4))
    _ax[0].imshow(cleaned[channel], cmap="gray", vmin=np.percentile(cleaned[channel], 1), vmax=_vmax)
    _ax[0].set_title("cleaned ch0 (grid + noise)")
    _ax[1].imshow(tissue_mask, cmap="gray")
    _ax[1].set_title("tissue mask")
    _ax[2].imshow(foreground[channel], cmap="gray", vmax=_vmax)
    _ax[2].set_title("background zeroed")
    for _a in _ax:
        _a.axis("off")
    plt.tight_layout()
    plt.gcf()
    #plt.savefig(f'{slide_out_folder}{slide_name}_artifact_background_removed_comparison.png')
    return


@app.cell
def _(tissue_mask, tppf):
    repaired, crack_stats = tppf.repair_cracks(tissue_mask, max_gap=30,verbose=True, return_stats=True)
    return crack_stats, repaired


@app.cell
def _(crack_stats):
    crack_stats
    return


@app.cell
def _(convex_hull_image, label, ndi, np, plt, regionprops, tissue_mask):
    #diagnostic for repair_cracks()
    _mask = tissue_mask                      # the mask you pass to repair_cracks
    _lbl = label(_mask, connectivity=2)
    _props = regionprops(_lbl)
    _areas = np.array([p.area for p in _props], float)
    _typical = np.median(_areas)
    _med_sol = float(np.median([p.solidity for p in _props]))
    print(f"components = {len(_props)}, typical area = {int(_typical)}, "
          f"median solidity = {_med_sol:.3f}\n")

    _under, _notched = [], []
    for _p in sorted(_props, key=lambda q: q.centroid[1]):
        _r = _p.area / _typical
        _flag = ""
        if _r < 0.85:
            _flag += "  <-- UNDERSIZED (crack half)"
            _under.append(_p)
        if _p.solidity < _med_sol - 0.012:
            _flag += "  <-- LOW SOLIDITY (notch / partial crack)"
            _notched.append(_p)
        print(f"  comp {_p.label:3d}: area={int(_p.area):7d} ({_r:4.2f}x) "
              f"sol={_p.solidity:.3f}{_flag}")

    print()
    if _under:
        print("CASE A: crack FULLY separates -> repair_cracks() applies. Pairwise gaps:")
        for _i in range(len(_under)):
            for _j in range(_i + 1, len(_under)):
                _a, _b = _under[_i], _under[_j]
                _r0 = max(min(_a.bbox[0], _b.bbox[0]) - 40, 0)
                _c0 = max(min(_a.bbox[1], _b.bbox[1]) - 40, 0)
                _r1 = min(max(_a.bbox[2], _b.bbox[2]) + 40, _mask.shape[0])
                _c1 = min(max(_a.bbox[3], _b.bbox[3]) + 40, _mask.shape[1])
                _sub = _lbl[_r0:_r1, _c0:_c1]
                _A, _B = _sub == _a.label, _sub == _b.label
                if not _A.any() or not _B.any():
                    continue
                _gap = float(ndi.distance_transform_edt(~_A)[_B].min())
                _comb = (_a.area + _b.area) / _typical
                print(f"    {_a.label}+{_b.label}: gap={_gap:.1f}px  "
                      f"combined={_comb:.2f}x typical"
                      + ("   <-- THE CRACK: set max_gap above this"
                         if 0.7 <= _comb <= 1.35 else ""))
    else:
        print("CASE A: no undersized components -> the crack does NOT fully separate.")
        print("        repair_cracks() can never find it (it joins TWO components).")

    #print()
    if _notched:
        print("CASE B: concavity geometry (notch = partial crack):")
        for _p in _notched:
            _r0, _c0, _r1, _c1 = _p.bbox
            _sub = (_lbl == _p.label)[_r0:_r1, _c0:_c1]
            _defic = convex_hull_image(_sub) & ~_sub
            for _dp in sorted(regionprops(label(_defic, connectivity=2)),
                              key=lambda q: -q.area)[:3]:
                print(f"    comp {_p.label}: notch area={int(_dp.area):6d} "
                      f"width~{_dp.axis_minor_length:5.1f}px "
                      f"depth~{_dp.axis_major_length:5.1f}px")

        _lbl2 = label(tissue_mask, connectivity=2)
        _suspects = _under + [p for p in _notched if p not in _under]

        _fig, _ax = plt.subplots(1, 1 + len(_suspects),
                                 figsize=(6 + 4 * len(_suspects), 5))
        _ax = np.atleast_1d(_ax)
        _rng = np.random.default_rng(0)
        _colors = np.vstack([[0, 0, 0], _rng.random((_lbl2.max(), 3)) * 0.7 + 0.3])
        _ax[0].imshow(_colors[_lbl2])
        _ax[0].set_title(f"components ({_lbl2.max()})")
        for _p in _suspects:
            _r0, _c0, _r1, _c1 = _p.bbox
            _ax[0].plot([_c0, _c1, _c1, _c0, _c0], [_r0, _r0, _r1, _r1, _r0],
                        color="red", lw=1.5)

        for _k, _p in enumerate(_suspects, start=1):
            _r0, _c0, _r1, _c1 = _p.bbox
            _pad = 30
            _rr0, _cc0 = max(_r0 - _pad, 0), max(_c0 - _pad, 0)
            _rr1 = min(_r1 + _pad, tissue_mask.shape[0])
            _cc1 = min(_c1 + _pad, tissue_mask.shape[1])
            _ax[_k].imshow(_colors[_lbl2[_rr0:_rr1, _cc0:_cc1]], interpolation="nearest")
            _ax[_k].set_title(f"comp {_p.label}: {_p.area/np.median([q.area for q in _props]):.2f}x, "
                              f"sol={_p.solidity:.3f}")
        for _a in _ax:
            _a.axis("off")
        plt.tight_layout()
        plt.gcf()
    return


@app.cell
def _(label, repaired, tissue_mask, tppf):
    separated, split_stats = tppf.separate_tissues(repaired, cut_width=5, verbose=True, return_stats=True)
    print("components:", label(tissue_mask, connectivity=2).max(),
          "->", label(separated, connectivity=2).max())
    return separated, split_stats


@app.cell
def _(separated):
    separated
    return


@app.cell
def _(split_stats):
    split_stats
    return


@app.cell
def _(foreground, separated, tppf):
    final, final_mask = tppf.remove_debris(
        foreground, 
        separated, 
        size_frac=0.15,          # large compact components are tissue outright
        max_aspect=4.0,          # main filter: elongated -> debris
        min_solidity=.3,        # main filter: wispy -> debris
        rescue_dist=9,           # a fragment this close to a section is rejoined (0 disables)
        rescue_max_aspect=2.5, 
        verbose=True, 
        return_mask=True
    )
    final.shape
    return final, final_mask


@app.cell
def _(
    channel,
    final,
    final_mask,
    foreground,
    label,
    ndi,
    np,
    plt,
    regionprops,
    tissue_mask,
):
    _vmax = np.percentile(final[channel][final_mask], 99)
    _debris = tissue_mask & ~final_mask  # pixels the filter dropped


    _fig, _ax = plt.subplots(1, 3, figsize=(15, 4))

    _ax[0].imshow(foreground[channel], cmap="gray", vmax=_vmax)
    _hi = np.zeros((*_debris.shape, 4))
    _hi[ndi.binary_dilation(_debris, structure=np.ones((3, 3)))] = [1, 0, 0, 0.55]
    _ax[0].imshow(_hi)  # red fill on the dropped components (dilated so thin/small ones show)
    for _p in regionprops(label(_debris)):
        _r0, _c0, _r1, _c1 = _p.bbox
        _pad = 5
        _ax[0].plot(
            [_c0 - _pad, _c1 + _pad, _c1 + _pad, _c0 - _pad, _c0 - _pad],
            [_r0 - _pad, _r0 - _pad, _r1 + _pad, _r1 + _pad, _r0 - _pad],
            color="red", lw=1.0,
        )
        _ax[0].text(
            _c1 + _pad + 3, (_r0 + _r1) / 2, str(_p.label),
            color="red", fontsize=8, ha="left", va="center",
        )
    _ax[0].set_title("before — dropped debris (red)")
    _ax[1].imshow(final_mask, cmap="gray")
    _ax[1].set_title("filtered mask")
    _ax[2].imshow(final[channel], cmap="gray", vmax=_vmax)
    _ax[2].set_title("debris removed")
    for _a in _ax:
        _a.axis("off")
    plt.tight_layout()
    plt.gcf()
    #plt.savefig(f'{slide_out_folder}{slide_name}_artifact_background_debris_removed_comparison.png')
    return


@app.cell
def _(label, np, regionprops, separated):
    #diagnostic for seperate_tissues()
    _lbl = label(separated)
    _props = regionprops(_lbl)
    _areas = np.array([p.area for p in _props], float)
    _ref = np.median(_areas[_areas >= np.percentile(_areas, 50)])
    print(f"n components = {len(_props)},  ref (typical area) = {int(_ref)},  "
          f"size cutoff = {int(0.15 * _ref)}")
    for _p in sorted(_props, key=lambda p: p.centroid[1]):
        _asp = _p.axis_major_length / max(_p.axis_minor_length, 1e-6)
        _big = _p.area >= 0.15 * _ref
        _ok = _asp <= 4.0 and _p.solidity >= 0.3
        print(f"  comp {_p.label:3d}: area={int(_p.area):7d} "
              f"asp={_asp:5.2f} sol={_p.solidity:.3f} "
              f"| size_ok={_big} shape_ok={_ok} -> "
              f"{'KEEP' if (_big and _ok) else 'DROP/rescue'}")
    return


@app.cell
def _(label, separated, tissue_mask):
    print("tissue_mask components:", label(tissue_mask).max())
    print("separated components:  ", label(separated).max())
    print("mask actually passed:  ", label(separated).max())
    print("pixels differing:      ", int((tissue_mask != separated).sum()))
    return


@app.cell
def _(final, final_mask, tppf):
    dewhiskered, dewhiskered_mask = tppf.remove_whiskers(
                                                    final, 
                                                    final_mask,         
                                                    width=5,                 # opening radius; severs protrusions thinner than ~2*width px
                                                    min_whisker_area=50,     # ignore tiny boundary specks, 
                                                    min_whisker_aspect=2.5, 
                                                    verbose=True, 
        return_mask=True
    )

    dewhiskered.shape
    return dewhiskered, dewhiskered_mask


@app.cell
def _(
    channel,
    dewhiskered,
    dewhiskered_mask,
    final,
    final_mask,
    label,
    ndi,
    np,
    plt,
    regionprops,
):
    _vmax = np.percentile(dewhiskered[channel][dewhiskered_mask], 99)
    _whiskers = final_mask & ~dewhiskered_mask  # pixels the whisker filter dropped

    _fig, _ax = plt.subplots(1, 3, figsize=(15, 4))

    _ax[0].imshow(final[channel], cmap="gray", vmax=_vmax)
    _hi = np.zeros((*_whiskers.shape, 4))
    _hi[ndi.binary_dilation(_whiskers, structure=np.ones((3, 3)))] = [1, 0, 0, 0.55]
    _ax[0].imshow(_hi)  # red fill on the dropped whiskers (dilated so thin ones show)
    for _p in regionprops(label(_whiskers)):
        _r0, _c0, _r1, _c1 = _p.bbox
        _pad = 5
        _ax[0].plot(
            [_c0 - _pad, _c1 + _pad, _c1 + _pad, _c0 - _pad, _c0 - _pad],
            [_r0 - _pad, _r0 - _pad, _r1 + _pad, _r1 + _pad, _r0 - _pad],
            color="red", lw=1.0,
        )
    _ax[0].set_title("before — dropped whiskers (red)")
    _ax[1].imshow(dewhiskered_mask, cmap="gray")
    _ax[1].set_title("dewhiskered mask")
    _ax[2].imshow(dewhiskered[channel], cmap="gray", vmax=_vmax)
    _ax[2].set_title("whiskers removed")
    for _a in _ax:
        _a.axis("off")
    plt.tight_layout()
    plt.gca()
    #plt.savefig(f'{slide_out_folder}{slide_name}_whiskers_removed.png')
    return


@app.cell
def _(slide_out_folder):
    slide_out_folder
    return


@app.cell
def _(
    artifact_mask,
    channel,
    cleaned,
    dewhiskered,
    dewhiskered_mask,
    down_area,
    final,
    final_mask,
    foreground,
    slide_name,
    slide_out_folder,
    tissue_mask,
    tppf,
):
    tppf.plot_pp_steps(dewhiskered, dewhiskered_mask, tissue_mask, artifact_mask, final, final_mask,
                  down_area, cleaned, foreground, channel, slide_out_folder=slide_out_folder, slide_name=slide_name, savefig=False)
    return


@app.cell
def _(Path, slide_out_folder):
    slide_out_folder
    _subdir = "bridge_inputs"
    _out_root = Path(slide_out_folder) / _subdir
    _out_root.mkdir(parents=True, exist_ok=True)
    return


@app.cell
def _(Path, slide_name, slide_out_folder):
    raw_slide_path = str(Path(slide_out_folder).parent) + '/' + slide_name + '.tif'
    raw_slide_path
    return (raw_slide_path,)


@app.cell
def _(
    dewhiskered,
    dewhiskered_mask,
    raw_slide_path,
    slide_name,
    slide_out_folder,
    tppf,
):
    annotated_output = tppf.annotate_sections(dewhiskered, dewhiskered_mask, return_order=True,
                                                    savefig=True,           # True -> plot and save the annotated image
                                                    slide_out_folder=slide_out_folder,   # output folder for the saved figure
                                                    slide_name=slide_name,
                                                    save_outputs=True,      # NEW: write bridge-pipeline inputs to a subdirectory
                                                    raw_slide_path=raw_slide_path,     # NEW: full-res source, used to record mask->raw 
                                                    subdir="bridge_inputs",  # NEW: subdirectory name inside slide_out_folder
                                                        )
    return (annotated_output,)


@app.cell
def _(annotated_output):
    annotated_output
    return


@app.cell
def _(annotated_output):
    str(annotated_output[2])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bridge - Preprocessed slide to single sections
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bridge functions
    """)
    return


@app.cell
def _():
    import tiff_pipeline_bridge_functions as tpbf

    return (tpbf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bridge metadata loop
    """)
    return


@app.cell
def _(Path, annotated_output, mo, pl, tpbf):
    _md_loop_enable = False
    #single slide all preprocessing functions
    mo.stop(_md_loop_enable is False, mo.md("*bridge metadata loop disabled (`_md_loop_enable` is False)*"))

    BRIDGE_ROOT = Path(str(annotated_output[2]))

    meta_paths = sorted(BRIDGE_ROOT.rglob("*_bridge_meta.json"))
    slide_inputs = {}
    inv_dfs = []
    for _p in meta_paths:
        _lm, _meta = tpbf.load_bridge_inputs(_p)
        slide_inputs[_meta["slide_name"]] = (_lm, _meta)
        inv_dfs.append(tpbf.inventory_sections(
            _lm, _meta["slide_name"], _meta["scale_y"], _meta["scale_x"]
        ))

    print(f"{len(inv_dfs)} slides, {sum(d.height for d in inv_dfs)} sections total")
    pl.concat(inv_dfs).sort("need_raw", descending=True).head(10)
    return inv_dfs, slide_inputs


@app.cell
def _(inv_dfs, pl):
    assert not pl.concat(inv_dfs)["touches_edge"].any(), "Found section(s) touching a slide edge"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bridge slide to single section loop
    """)
    return


@app.cell
def _(inv_dfs, tpbf):
    canvas_size = tpbf.plan_canvas(inv_dfs, margin_frac=0.05, round_to=64)
    return (canvas_size,)


@app.cell
def _(Path, slide_out_folder):
    single_section_out_dir = str(Path(slide_out_folder).parent) + '/single_section_autoraw/'
    return (single_section_out_dir,)


@app.cell
def _(single_section_out_dir):
    single_section_out_dir
    return


@app.cell
def _(canvas_size, mo, pl, single_section_out_dir, slide_inputs, tpbf):
    _bridge_loop_enable = False
    #single slide all preprocessing functions
    mo.stop(_bridge_loop_enable is False, mo.md("*bridge slide to single section loop disabled (`_bridge_loop_enable` is False)*"))

    manifests = []
    for _name, (_lm, _meta) in slide_inputs.items():
        print(_name)
        _inv = tpbf.inventory_sections(_lm, _name, _meta["scale_y"], _meta["scale_x"])
        manifests.append(tpbf.extract_sections(
            slide_path=_meta["raw_slide_path"],
            slide_name=_name,
            label_mask=_lm,
            inv_df=_inv,
            canvas_size=canvas_size,
            out_dir=single_section_out_dir,
            scale_y=_meta["scale_y"],
            scale_x=_meta["scale_x"],
        ))
    pl.concat(manifests)
    return


@app.cell
def _(slide_name):
    slide_name
    return


@app.cell
def _(mo, np, plt, slide_name, tifffile):
    #compare autoraw and autoaligned with matlab single section raw and processed
    _compare_enable = False
    #single slide all preprocessing functions
    mo.stop(_compare_enable is False, mo.md("*_compare_enable disabled (`_compare_enable` is False)*"))


    _slide_name = slide_name
    _section_num = '1'
    with tifffile.TiffFile(f'/bigdata/isaac/rabies_img_processing/129_02/single_section_autoraw/{_slide_name}.tif_section_{_section_num}.tiff') as _tif:
        _initial_auto = _tif.series[0].asarray()  

    with tifffile.TiffFile(f'/bigdata/isaac/rabies_img_processing/129_02/single_section/{_slide_name}.tif_section_{_section_num}.tiff') as _tif:
        _initial_matlab = _tif.series[0].asarray()  

    #with tifffile.TiffFile(f'/bigdata/isaac/rabies_img_processing/129_02/single_section_autoalign/{_slide_name}.tif_section_{_section_num}.tiff') as _tif:
        #_processed_auto = _tif.series[0].asarray()  
    _processed_auto = np.load(f'/bigdata/isaac/rabies_img_processing/129_02/single_section_autoalign/imgs/img_slice_{_slide_name}_section_{_section_num}.npy')

    with tifffile.TiffFile(f'/bigdata/isaac/rabies_img_processing/129_02/single_section/{_slide_name}.tif_section_{_section_num}_f.tiff') as _tif:
        _processed_matlab = _tif.series[0].asarray()  

    _ch = 0
    _vmax_cmp = np.percentile(_processed_auto[_ch], 99)
    _fig_cmp, _ax_cmp = plt.subplots(2,2, figsize=(12, 6))
    _ax_cmp[0,0].imshow(_initial_auto[_ch], cmap="gray", vmax = _vmax_cmp)
    _ax_cmp[0,0].set_title(f"initial autoraw ch{_ch} — {_initial_auto.shape}, {_slide_name}, section {_section_num}")

    _ax_cmp[0,1].imshow(_initial_matlab[_ch], cmap="gray", vmax = _vmax_cmp)
    _ax_cmp[0,1].set_title(f"initial matlab ch{_ch} — {_initial_matlab.shape}, {_slide_name}, section {_section_num}")

    _ax_cmp[1,0].imshow(_processed_auto[_ch], cmap="gray", vmax = _vmax_cmp)
    _ax_cmp[1,0].set_title(f"processed autoraw \n (w/ initial matlab input) ch{_ch} — {_processed_auto.shape}, {_slide_name}, section {_section_num}")

    _ax_cmp[1,1].imshow(_processed_matlab[_ch], cmap="gray", vmax = _vmax_cmp)
    _ax_cmp[1,1].set_title(f"processed matlab ch{_ch} — {_processed_matlab.shape}, {_slide_name}, section {_section_num}")

    #plt.savefig(f'/bigdata/isaac/rabies_img_processing/129_02/single_section_comp_{_slide_name}_section_{_section_num}.png')

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Single section processing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## single section functions
    """)
    return


@app.cell
def _():
    import tiff_pipeline_ss_functions as tpssf

    return (tpssf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Single Section Autoalignment loop
    """)
    return


@app.cell
def _(mo, os, tifffile, time, tpssf):
    _autoalign_loop_enable = False
    #single slide all preprocessing functions
    mo.stop(_autoalign_loop_enable is False, mo.md("*bridge slide to single section loop disabled (`_autoalign_loop_enable` is False)*"))


    #load amit single section for reference
    brain_id = '129_02'

    #_slide_num = 'S001'
    #_section_num = '1'

    #get all single secton filenames for a given brain id

    #matlab single section dir (for initial buildout/testing)
    #_single_section_dir = f"/bigdata/isaac/rabies_img_processing/{brain_id}/single_section/"

    #autoraw single section dir
    _single_section_dir = f"/bigdata/isaac/rabies_img_processing/{brain_id}/single_section_autoraw/"

    _single_tiff_filenames = sorted(
        f for f in os.listdir(_single_section_dir)
        if f.endswith(".tiff") and f[:-len(".tiff")][-1:].isdigit()
    )
    _num_files = len(_single_tiff_filenames)

    #create directory to save plots,imgs, masks
    _save_dir = f'/bigdata/isaac/rabies_img_processing/{brain_id}/single_section_autoalign/'
    os.makedirs(_save_dir, exist_ok=True)

    _start_time_all_imgs = time.perf_counter()
    _counter  = 1
    for i,stf in enumerate(_single_tiff_filenames):
        _slide_num = _single_tiff_filenames[i].split('.')[0]
        _section_num = int(_single_tiff_filenames[i].split('_')[2].split('.')[0])
        _start_time = time.perf_counter()

        print (f'loading: {_slide_num}.tif_section_{_section_num}.tiff...')
        with tifffile.TiffFile(f'{_single_section_dir}{_slide_num}.tif_section_{_section_num}.tiff') as _tif:
            _initial = _tif.series[0].asarray()  
        print ('done.')

        print('computing binary mask...')
        _initial_mask = tpssf.compute_binary_mask(_initial, channel=0, method="li")
        print('done.')

        #print ('dilating...')
        #_ellipse_mask, _ellipse_info = dilate_to_ellipse(_initial_mask, return_info=True)
        #print ('done.')

        print('calculating midline and rotation angle...')
        _res = tpssf.find_midline_rotation(_initial_mask)
        _mirror_angle = _res['rotation_deg']
        print('done.')

        print('rotating around centroid...')
        _rotated_centroid, _rotated_centroid_mask = tpssf.rotate_around_centroid(_initial, _mirror_angle, mask=_initial_mask,
                                                 save_dir=None,
                                                 slide_num=_slide_num, 
                                                   section_num=_section_num)
        print('done.')

        print('aligning centroid to image center...')
        _centered_img, _centered_mask = tpssf.center_img(_rotated_centroid_mask, _rotated_centroid, _initial,
                                                save_dir=_save_dir,
                                                 slide_num=_slide_num, 
                                                   section_num=_section_num,
                                                      brain_id=brain_id)
        print(f'done.' )


        _end_time = time.perf_counter()
        _execution_time = _end_time - _start_time
        _counter+=1
        print (f'completed in {_execution_time:6f} seconds')
        print (f'{_counter}/{_num_files}')


    _end_time_all_imgs = time.perf_counter()
    _execution_time_all_imgs = _end_time_all_imgs - _start_time_all_imgs

    print (f'completed autoalignment of brain {brain_id}, containing {_num_files} tiff img files, in {_execution_time_all_imgs:6f} seconds')
    return (brain_id,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Script for stiching autoalign output to single pdf
    """)
    return


@app.cell
def _(BytesIO, PdfReader, PdfWriter, brain_id, canvas, glob, os, re):

    _plot_dir = f'/bigdata/isaac/rabies_img_processing/{brain_id}/single_section_autoalign/plots/'



    # Find every per-section plot and parse slice/section from its filename,
    # e.g. autorotated_centered_slice_S001_section_1.pdf -> ("S001", 1).
    _pdf_pattern = re.compile(r"autorotated_centered_slice_(S\d+)_section_(\d+)\.pdf$")
    _pdf_entries = []
    for _fp in glob.glob(os.path.join(_plot_dir, "*.pdf")):
        _match = _pdf_pattern.search(os.path.basename(_fp))
        if _match:
            _slice_id = _match.group(1)
            _section_num = int(_match.group(2))
            _pdf_entries.append((_slice_id, _section_num, _fp))

    # Group all sections together for each slice: order by slice (S001, S002, ...)
    # then by section number within each slice (1, 2, ...).
    _pdf_entries.sort(key=lambda _e: (_e[0], _e[1]))

    # New /plots/combined/ directory for the stitched output.
    _combined_dir = os.path.join(_plot_dir, "combined")
    os.makedirs(_combined_dir, exist_ok=True)
    _combined_path = os.path.join(
        _combined_dir, "autorotated_centered_all_slices_all_sections.pdf"
    )

    _writer = PdfWriter()
    for _slice_id, _section_num, _fp in _pdf_entries:
        _reader = PdfReader(_fp)
        for _page in _reader.pages:
            _pw = float(_page.mediabox.width)
            _ph = float(_page.mediabox.height)

            # Overlay the slice/section label at the upper center of the page.
            _buf = BytesIO()
            _c = canvas.Canvas(_buf, pagesize=(_pw, _ph))
            _c.setFont("Helvetica-Bold", 14)
            _c.drawCentredString(_pw / 2.0, _ph - 20, f"{_slice_id}  section {_section_num}")
            _c.save()
            _buf.seek(0)

            _overlay_page = PdfReader(_buf).pages[0]
            _page.merge_page(_overlay_page)
            _writer.add_page(_page)

    with open(_combined_path, "wb") as _out_f:
        _writer.write(_out_f)

    print(f"stitched {len(_pdf_entries)} plot(s) -> {_combined_path}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## single section processing step-by-step (for debugging)
    """)
    return


@app.cell
def _(mo, os, tifffile):
    _ss_debug_enable = False
    #single slide all preprocessing functions
    mo.stop(_ss_debug_enable is False, mo.md("_ss_debug_enable disabled (`_ss_debug_enable` is False)*"))

    _brain_id = '129_02'
    _single_section_dir = f"/bigdata/isaac/rabies_img_processing/{_brain_id}/single_section/"
    single_tiff_filenames = sorted(
        f for f in os.listdir(_single_section_dir)
        if f.endswith(".tiff") and f[:-len(".tiff")][-1:].isdigit()
    )
    len(single_tiff_filenames)


    slide_num = 'S016'
    section_num = '6'
    #load single section autoraw output
    with tifffile.TiffFile(f'/bigdata/isaac/rabies_img_processing/129_02/single_section_autoraw/{slide_num}.tif_section_{section_num}.tiff') as _tif:
        initial = _tif.series[0].asarray()  

    #amit single section for reference
    #with tifffile.TiffFile(f'/bigdata/isaac/rabies_img_processing/129_02/single_section/{slide_num}.tif_section_{section_num}.tiff') as _tif:
    #    initial = _tif.series[0].asarray()  
    #load processed amit single section for reference
    with tifffile.TiffFile(f'/bigdata/isaac/rabies_img_processing/129_02/single_section/{slide_num}.tif_section_{section_num}_f.tiff') as _tif_f:
        processed = _tif_f.series[0].asarray()  
    return initial, processed, section_num, single_tiff_filenames, slide_num


@app.cell
def _(initial):
    initial
    return


@app.cell
def _(processed):
    processed
    return


@app.cell
def _(initial, np, plt, processed):
    _ch = 0
    _vmax_cmp = np.percentile(processed[_ch], 99)
    _fig_cmp, _ax_cmp = plt.subplots(1, 2, figsize=(12, 6))
    _ax_cmp[0].imshow(initial[_ch], cmap="gray", vmax=_vmax_cmp)
    _ax_cmp[0].set_title(f"initial ch{_ch} — {initial.shape}")
    _ax_cmp[1].imshow(processed[_ch], cmap="gray", vmax=_vmax_cmp)
    _ax_cmp[1].set_title(f"processed ch{_ch} — {processed.shape}")
    for _a in _ax_cmp:
        _a.axis("off")
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(initial, np, plt, tpssf):

    initial_mask = tpssf.compute_binary_mask(initial, channel=0, method="li")

    _base_im = (initial[0] if initial.ndim == 3 else initial).astype(np.float32)
    _vmax_im = np.percentile(_base_im[initial_mask], 99) if initial_mask.any() else _base_im.max()
    _fig_mask, _ax_mask = plt.subplots(1, 3, figsize=(15, 5))
    _ax_mask[0].imshow(_base_im, cmap="gray", vmax=_vmax_im)
    _ax_mask[0].set_title(f"initial ch0 — {_base_im.shape}")
    _ax_mask[1].imshow(initial_mask, cmap="gray")
    _ax_mask[1].set_title("binary mask")
    _ax_mask[2].imshow(_base_im, cmap="gray", vmax=_vmax_im)
    _ax_mask[2].imshow(initial_mask, cmap="Reds", alpha=0.35)
    _ax_mask[2].set_title("mask overlay")
    for _a in _ax_mask:
        _a.axis("off")
    plt.tight_layout()
    plt.gca()
    return (initial_mask,)


@app.cell
def _():
    #ellipse_mask, ellipse_info = dilate_to_ellipse(initial_mask, return_info=True)
    #print("best IoU vs fitted ellipse:", round(ellipse_info["best_iou"], 4),
    #      "| iterations:", ellipse_info["n_iter"])

    #_fig_el, _ax_el = plt.subplots(1, 3, figsize=(15, 5))
    #_ax_el[0].imshow(initial_mask, cmap="gray")
    #_ax_el[0].set_title("initial_mask")
    #_ax_el[1].imshow(ellipse_mask, cmap="gray")
    #_ax_el[1].set_title(f"dilated to ellipse (IoU={ellipse_info['best_iou']:.3f})")
    #_ax_el[2].imshow(initial_mask, cmap="gray")
    #_ax_el[2].imshow(ellipse_mask & ~initial_mask, cmap="Reds", alpha=0.5)
    #_ax_el[2].set_title("added by dilation (red)")
    #for _a in _ax_el:
    #    _a.axis("off")
    #plt.tight_layout()
    #plt.gca()

    return


@app.cell
def _():
    #rot_angle_rp, props_rp, cy_rp, cx_rp = compute_rot_angle_regionprops(ellipse_mask)
    #print("regionprops rotation angle (deg):", round(rot_angle_rp, 3))
    #plt.gca()
    return


@app.cell
def _():
    #rot_angle, rot_info, cy, cx = compute_rot_angle(ellipse_mask)
    #print("rotation angle (deg):", round(rot_angle, 3))
    #plt.gca()
    return


@app.cell
def _():
    return


@app.cell
def _(single_tiff_filenames):
    _slide_num = single_tiff_filenames[0].split('.')[0]
    _slide_num
    return


@app.cell
def _(single_tiff_filenames):
    _section_num = int(single_tiff_filenames[0].split('_')[2].split('.')[0])
    _section_num
    return


@app.cell
def _(initial_mask, plt, tpssf):

    _res = tpssf.find_midline_rotation(initial_mask)
    mirror_angle = _res['rotation_deg']
    _fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.plot(_res["curve_angles"], _res["curve_scores"], "-", lw=1.5)
    _ax.axvline(_res["rotation_deg"], color="red", ls="--", lw=1)
    _ax.set_xlabel("candidate rotation (deg)")
    _ax.set_ylabel("mirror IoU")
    _ax.set_title(f"peak {_res['rotation_deg']:.2f}°  IoU {_res['iou']:.3f}")
    plt.gca()
    print (_res['rotation_deg'])
    return (mirror_angle,)


@app.cell
def _(initial, initial_mask, mirror_angle, plt, section_num, slide_num, tpssf):



    rotated_centroid, rotated_centroid_mask = tpssf.rotate_around_centroid(initial, mirror_angle, mask=initial_mask,
                                             save_dir='/bigdata/isaac/rabies_img_processing/',
                                             slide_num=slide_num, section_num=section_num)

    plt.gca()
    return rotated_centroid, rotated_centroid_mask


@app.cell
def _():
    #sym_angle, sym_info, cy_sym, cx_sym = compute_symmetry_line_pca(ellipse_mask, initial)
    #print("PCA symmetry angle (deg):", round(sym_angle, 3))
    #plt.gca()
    return


@app.cell
def _(
    initial,
    plt,
    rotated_centroid,
    rotated_centroid_mask,
    section_num,
    slide_num,
    tpssf,
):
    _brain_id = '129_02'
    centered_img, centered_mask = tpssf.center_img(rotated_centroid_mask, rotated_centroid, initial, brain_id = _brain_id, slide_num = slide_num, section_num=section_num, save_dir = f'/bigdata/isaac/rabies_img_processing/{_brain_id}/')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Check tiff properties of autoaligned output, matlab vs python
    """)
    return


@app.cell
def _(mo, np, tifffile):
    _check_tiff_props_enable = False
    #single slide all preprocessing functions
    mo.stop(_check_tiff_props_enable is False, mo.md("_check_tiff_props_enable disabled (`_check_tiff_props_enable` is False)*"))


    python_section_path = '/bigdata/isaac/rabies_img_processing/129_02/imgs/129_02_img_slice_S001_section_1.tiff'
    matlab_section_path = '/bigdata/isaac/rabies_img_processing/129_02/imgs/S010.tif_section_1_f.tiff'

    for _p in [python_section_path, matlab_section_path]:
        _a = tifffile.imread(_p)
        print(_p, _a.dtype, _a.min(), _a.max(), np.percentile(_a[_a>0], 99.9))
    return matlab_section_path, python_section_path


@app.cell
def _(matlab_section_path, np, python_section_path, tifffile):
    _qs = [1, 25, 50, 75, 90, 99, 99.9, 99.99]
    for _name, _p in {"python": python_section_path, "matlab": matlab_section_path}.items():
        _a = tifffile.imread(_p)
        _t = _a[_a > 0]
        _mode = np.argmax(np.bincount(_a.ravel())[1:]) + 1
        _pct = " ".join(f"p{_q}={np.percentile(_t, _q):.0f}" for _q in _qs)
        print(f"{_name:7s} mode={_mode:6d} {_pct} max={_a.max()}")
        print(f"        px at max={(_a == _a.max()).sum():6d}   "
              f"px >= 0.99*max={(_a >= 0.99 * _a.max()).sum():6d}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Slide to Single Section (mega) loop
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## mega loop function (wraps preprocessing, bridge, and single section function calls into single loop)
    """)
    return


@app.cell
def _():
    import tiff_pipeline_mega_loop as tpml

    return (tpml,)


@app.cell
def _(mo, tpml):
    _enable_mega_loop = True
    mo.stop(_enable_mega_loop is not True, mo.md("*mega loop disabled (`_enable_mega_loop` is not True)*"))

    _brain_ids_to_process = ['129_02']#['140_01','141_03','137_02','137_04']
    for _brain_id in _brain_ids_to_process:
        tpml.mega_loop(_brain_id)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Review Sections
    ```Note:  ensure 'autorun' is selected under 'on cell change' in reactivity settings```
    """)
    return


@app.cell
def _():
    import tiff_pipeline_review_functions as tprf

    return (tprf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## load section for review
    """)
    return


@app.cell
def _(tprf):
    brain_id_r = '140_01'
    slide_num_r = 'S001'
    section_paths = tprf.find_section_paths(brain_id_r,slide_num_r)
    review_controls = tprf.build_review_controls(len(section_paths))
    thumbs = tprf.load_review_thumbs(section_paths)
    return brain_id_r, review_controls, section_paths, slide_num_r, thumbs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## render section
    """)
    return


@app.cell
def _(mo, section_paths, tprf):
    edit_state, set_edit_state = mo.state(tprf.empty_edits(len(section_paths)))
    return edit_state, set_edit_state


@app.cell
def _(mo, os, section_paths):
    focus = mo.ui.dropdown(
        options={f"{i+1} · {os.path.basename(p)}": i
                 for i, p in enumerate(section_paths)},
        value=f"1 · {os.path.basename(section_paths[0])}",
        label="edit section",
    )
    focus
    return (focus,)


@app.cell
def _(edit_state, focus, mo, os, review_controls, section_paths, thumbs, tprf):
    _i = focus.value
    _v = review_controls.value[_i]
    _fig, _ax = tprf.build_edit_canvas(
        tprf._as_display(thumbs[_i], 0),
        tprf.pending_angle(_v), _v["flip_lr"],
        edit_state()[_i],
        title=os.path.basename(section_paths[_i]),
    )
    canvas_r = mo.ui.matplotlib(_ax, debounce=True)
    mo.vstack([
        mo.md("drag = box · **shift**+drag = lasso"),
        canvas_r,
    ])
    return (canvas_r,)


@app.cell
def _(canvas_r, edit_state, focus, mo, review_controls, set_edit_state, tprf):
    _armed = bool(canvas_r.value)

    def _do_line(_):
        _i = focus.value
        _m = tprf.selection_to_mask(canvas_r.value, canvas_r.axes.images[0].get_array().shape)
        _ang, _msg = tprf.midline_angle_from_mask(_m)
        print(_msg)
        if _ang is None:
            return
        _e = dict(edit_state())
        _cur = dict(_e[_i])
        _cur["extra_rot"] = _cur["extra_rot"] + _ang
        _cur["drawn_at"] = tprf.pending_angle(review_controls.value[_i]) + _cur["extra_rot"]
        _e[_i] = _cur
        set_edit_state(_e)

    def _do_remove(_):
        _i = focus.value
        _m = tprf.selection_to_mask(canvas_r.value, canvas_r.axes.images[0].get_array().shape)
        if _m.sum() == 0:
            print("empty selection")
            return
        _e = dict(edit_state())
        _cur = dict(_e[_i])
        _cur["removals"] = _cur["removals"] + [_m]
        _e[_i] = _cur
        set_edit_state(_e)
        print(f"removed {int(_m.sum())} thumbnail px")

    draw_sym_line = mo.ui.button(
        label="draw sym line", kind="warn" if _armed else "neutral",
        on_change=_do_line,
    )
    remove_selection = mo.ui.button(
        label="remove selection", kind="warn" if _armed else "neutral",
        on_change=_do_remove,
    )
    mo.hstack([draw_sym_line, remove_selection], justify="start", gap=0.5)
    return


@app.cell
def _(edit_state, review_controls, section_paths, thumbs, tprf):
    tprf.render_review_grid(thumbs, section_paths, review_controls, edits=edit_state(),channel=0)
    return


@app.cell
def _():
    return


@app.cell
def _(brain_id_r, os, slide_num_r):
    review_root = f"/bigdata/isaac/rabies_img_processing/{brain_id_r}/single_section_review"
    applied_dir = os.path.join(review_root, "imgs")
    manifest_path = os.path.join(
        review_root, f"{brain_id_r}_slide_{slide_num_r}_review.csv"
    )
    return applied_dir, manifest_path


@app.cell
def _(manifest_path):
    manifest_path
    return


@app.cell
def _(
    brain_id_r,
    edit_state,
    review_controls,
    section_paths,
    slide_num_r,
    tprf,
):
    transforms_df = tprf.load_slide_transforms(brain_id_r, slide_num_r)
    decisions = tprf.collect_decisions(
            section_paths, review_controls, edits=edit_state(),
            brain_id=brain_id_r, slide_num=slide_num_r,
            transforms=transforms_df,
        )
    decisions
    return (decisions,)


@app.cell
def _(brain_id_r, decisions, mo, pl, slide_num_r):
    _n_keep = decisions.filter(pl.col("status") == "keep").height
    _n_edit = decisions.filter(
        (pl.col("status") == "keep")
        & ((pl.col("rotate_ccw_deg") != 0) | pl.col("flip_lr"))
    ).height
    mo.md(
        f"**{brain_id_r} · slide {slide_num_r}** — {decisions.height} sections  \n"
        f"keep **{_n_keep}** · reject **{decisions.height - _n_keep}** · "
        f"of the kept, **{_n_edit}** transformed"
    )
    return


@app.cell
def _(mo):
    save_manifest_button = mo.ui.run_button(label="save review manifest", kind="neutral")
    save_manifest_button
    return (save_manifest_button,)


@app.cell
def _(decisions, manifest_path, mo, save_manifest_button, tprf):
    mo.stop(not save_manifest_button.value, mo.md("*manifest not saved*"))
    tprf.save_decisions(decisions, manifest_path)
    return


@app.cell
def _(mo):
    dry_run_button = mo.ui.run_button(label="preview apply plan", kind="neutral")
    dry_run_button
    return (dry_run_button,)


@app.cell
def _(applied_dir, decisions, dry_run_button, edit_state, mo, tprf):
    mo.stop(not dry_run_button.value, mo.md("*click to preview — nothing is written*"))
    _planned, _skipped = tprf.apply_decisions(decisions, applied_dir, dry_run=True, edits=edit_state())
    mo.md(f"would write **{len(_planned)}** to `{applied_dir}`, skip **{len(_skipped)}**")
    return


@app.cell
def _(mo):
    apply_button = mo.ui.run_button(
        label="⚠ WRITE full-resolution TIFFs", kind="neutral"
    )
    apply_button
    return (apply_button,)


@app.cell
def _(
    applied_dir,
    apply_button,
    decisions,
    edit_state,
    manifest_path,
    mo,
    tprf,
):
    mo.stop(not apply_button.value, mo.md("*no files written*"))
    written, skipped = tprf.apply_decisions(
        decisions, applied_dir, edits=edit_state(), dry_run=False, overwrite=False, verbose=True
    )
    tprf.save_decisions(decisions, manifest_path)
    mo.md(f"wrote **{len(written)}** · skipped **{len(skipped)}**")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Validation of manual review edits
    """)
    return


@app.cell
def _(applied_dir, brain_id_r, os, tprf):
    _base = "140_01_img_slice_S001_section_1.tiff"
    _autoalign_path = os.path.join(
        f"/bigdata/isaac/rabies_img_processing/{brain_id_r}/single_section_autoalign/imgs",
        _base,
    )
    _out_fn = _base.split('.')[0] + '_reviewed' + '.' + _base.split('.')[1]
    _reviewed_path = os.path.join(applied_dir, _out_fn)
    _example_paths = [_autoalign_path, _reviewed_path]
    _labels= [x.split('.')[0].split('/')[-1] for x in _example_paths]
    _example_thumbs = tprf.load_review_thumbs(_example_paths, verbose=False)


    tprf.plot_comparison(_example_thumbs, _labels, channel=0)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
