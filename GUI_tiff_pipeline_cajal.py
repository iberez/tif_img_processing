import marimo

__generated_with = "0.25.0"
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

    return Path, mo, np, os, pl, re


@app.cell
def _(mo):
    mo.md("""
    # TIFF processing pipeline
    Process raw microscope slide TIFFs into aligned single sections, then review the output.

    1. **Slide → single sections** — pick brain id(s) and run the automated pipeline.
    2. **Review sections** — pick a brain and slide, inspect each section, fix rotation / flips, and write reviewed TIFFs.
    """)
    return


@app.cell
def _():
    import tiff_pipeline_pp_functions as tppf

    return


@app.cell
def _():
    import tiff_pipeline_bridge_functions as tpbf

    return


@app.cell
def _():
    import tiff_pipeline_ss_functions as tpssf

    return


@app.cell
def _():
    import tiff_pipeline_mega_loop as tpml

    return (tpml,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1 · Slide to single sections (auto processing)
    Toggle **batch process** to queue all brains in root folder, or select the folders you want to process. Then click **run**.
    """)
    return


@app.cell
def _(mo):
    # browse to a root folder that contains one subfolder per brain id
    mega_root_browser = mo.ui.file_browser(
        initial_path="/bigdata/isaac/rabies_img_processing",
        selection_mode="directory",
        multiple=True,
        label="slide workdir root (contains brain_id folders)",
    )
    mega_root_browser
    return (mega_root_browser,)


@app.cell(hide_code=True)
def _(Path, mega_root_browser, re):
    # The browser (multiple=True) lets the user pick brain-id folders directly, OR
    # a single root folder that contains brain-id subfolders. Either way we resolve
    # the workdir root (passed to mega_loop) and the list of brain ids.
    _brain_id_pat = re.compile(r"\d+_\d+$")
    if mega_root_browser.value:
        _selected = [Path(_f.path) for _f in mega_root_browser.value]
        _as_brains = sorted({_p.name for _p in _selected
                             if _brain_id_pat.fullmatch(_p.name)})
        if _as_brains:
            # selection(s) are brain-id folders -> root is their common parent
            mega_brain_ids = _as_brains
            mega_workdir_root = _selected[0].parent
        else:
            # selection is a root folder -> expand its brain-id subfolders
            mega_workdir_root = _selected[0]
            mega_brain_ids = sorted(
                _d.name for _d in mega_workdir_root.iterdir()
                if _d.is_dir() and _brain_id_pat.fullmatch(_d.name)
            )
    else:
        mega_workdir_root = Path("/bigdata/isaac/rabies_img_processing")
        mega_brain_ids = sorted(
            _d.name for _d in mega_workdir_root.iterdir()
            if _d.is_dir() and _brain_id_pat.fullmatch(_d.name)
        )
    return mega_brain_ids, mega_workdir_root


@app.cell(hide_code=True)
def _(mo):
    batch_process = mo.ui.switch(value=False, label="batch process")
    return (batch_process,)


@app.cell(hide_code=True)
def _(mega_brain_ids, mo):
    brain_id_single = mo.ui.dropdown(
        options=mega_brain_ids,
        value=mega_brain_ids[0] if mega_brain_ids else None,
        label="brain id(s) to process",
    )
    return (brain_id_single,)


@app.cell(hide_code=True)
def _(mo):
    run_mega_button = mo.ui.run_button(label="▶ run slide → single section pipeline", kind="success")
    return (run_mega_button,)


@app.cell
def _(batch_process, brain_id_single, mega_brain_ids, mo, run_mega_button):
    mo.vstack([
        mo.md("### Auto-process raw microscope slides → single sections"),
        batch_process,
        (
            mo.md(
                f"**batch process enabled** — all **{len(mega_brain_ids)}** brain ids "
                f"in the root will be processed:\n\n" + ", ".join(f"`{_b}`" for _b in mega_brain_ids)
            )
            if batch_process.value
            else brain_id_single
        ),
        run_mega_button,
    ])
    return


@app.cell(hide_code=True)
def _(
    batch_process,
    brain_id_single,
    mega_brain_ids,
    mega_workdir_root,
    mo,
    run_mega_button,
    tpml,
):
    mo.stop(not run_mega_button.value, mo.md("*select brain id(s) above, then click run*"))
    _ids = list(mega_brain_ids) if batch_process.value else (
        [brain_id_single.value] if brain_id_single.value else []
    )
    mo.stop(not _ids, mo.md("*no brain id selected — choose one and click run again*"))
    for _bid in mo.status.progress_bar(_ids, title="processing", subtitle="slide → single sections"):
        tpml.mega_loop(_bid, root=str(mega_workdir_root))
    mo.md(f"✅ finished: **{', '.join(_ids)}** (root: `{mega_workdir_root}`)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2 · Review sections
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
    ## load sections for review
    """)
    return


@app.cell
def _(mo):
    # browse to the review root folder (contains brain_id folders with
    # single_section_autoalign/imgs output)
    review_root_browser = mo.ui.file_browser(
        initial_path="/bigdata/isaac/rabies_img_processing",
        selection_mode="directory",
        multiple=False,
        label="review root (contains brain_id folders)",
    )
    review_root_browser
    return (review_root_browser,)


@app.cell
def _(Path, mo, review_root_browser):
    review_root_dir = (
        Path(review_root_browser.path(index=0))
        if review_root_browser.value
        else Path("/bigdata/isaac/rabies_img_processing")
    )
    review_brain_ids = sorted(
        _d.name for _d in review_root_dir.iterdir()
        if (_d / "single_section_autoalign" / "imgs").is_dir()
    )
    brain_select = mo.ui.dropdown(options=review_brain_ids, label="brain id")
    load_all_slides = mo.ui.switch(value=False, label="load all slides (S001, S002, …)")
    mo.hstack([brain_select, load_all_slides], justify="start", gap=2)
    return brain_select, load_all_slides, review_root_dir


@app.cell(hide_code=True)
def _(brain_select, mo, os, re, review_root_dir):
    mo.stop(brain_select.value is None, mo.md("*select a brain id to begin*"))
    brain_id_r = brain_select.value
    _img_dir = review_root_dir / brain_id_r / "single_section_autoalign" / "imgs"
    _slide_pat = re.compile(r"_img_slice_(S\d+)_section_\d+\.tiff$")
    slide_ids_r = sorted({_m.group(1) for _f in os.listdir(_img_dir) if (_m := _slide_pat.search(_f))})
    mo.stop(not slide_ids_r, mo.md(f"*no aligned sections found for `{brain_id_r}`*"))
    mo.md(f"**{brain_id_r}** — **{len(slide_ids_r)}** slides available ({slide_ids_r[0]} … {slide_ids_r[-1]})")
    return brain_id_r, slide_ids_r


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Slide Review
    """)
    return


@app.cell(hide_code=True)
def _(mo, slide_ids_r):
    slide_select = mo.ui.dropdown(options=slide_ids_r, value=slide_ids_r[0], label="slide to review")
    slide_select
    return (slide_select,)


@app.cell(hide_code=True)
def _(brain_id_r, load_all_slides, slide_ids_r, tprf):
    # 'load all slides' preloads every slide's thumbnails once; otherwise only the
    # selected slide is loaded on demand.
    if load_all_slides.value:
        loaded_section_paths = {_s: tprf.find_section_paths(brain_id_r, _s) for _s in slide_ids_r}
        loaded_thumbs = {_s: tprf.load_review_thumbs(_p, verbose=False)
                         for _s, _p in loaded_section_paths.items()}
    else:
        loaded_section_paths, loaded_thumbs = None, None
    return loaded_section_paths, loaded_thumbs


@app.cell(hide_code=True)
def _(brain_id_r, loaded_section_paths, loaded_thumbs, mo, slide_select, tprf):
    mo.stop(slide_select.value is None, mo.md("*select a slide*"))
    slide_num_r = slide_select.value
    if loaded_thumbs is not None:
        section_paths = loaded_section_paths[slide_num_r]
        thumbs = loaded_thumbs[slide_num_r]
    else:
        section_paths = tprf.find_section_paths(brain_id_r, slide_num_r)
        thumbs = tprf.load_review_thumbs(section_paths, verbose=False)
    review_controls = tprf.build_review_controls(len(section_paths))
    return review_controls, section_paths, slide_num_r, thumbs


@app.cell
def _(edit_state, review_controls, section_paths, thumbs, tprf):
    tprf.render_review_grid(thumbs, section_paths, review_controls, edits=edit_state(),channel=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section Review
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
    _flip = bool(_v["flip_lr"]) ^ bool(edit_state()[_i].get("flip_lr", False))
    _fig, _ax = tprf.build_edit_canvas(
        tprf._as_display(thumbs[_i], 0),
        tprf.pending_angle(_v), _flip,
        edit_state()[_i],
        title=os.path.basename(section_paths[_i]),
    )
    canvas_r = mo.ui.matplotlib(_ax, debounce=True)
    mo.vstack([
        mo.md(
            "**remove selection** — hold **shift** and drag a **lasso** around the region "
            "to cut (a plain box drag also works)  \n"
            "**draw sym line** — hold **shift** and drag a straight stroke between the two "
            "midline points; the line through those points sets the symmetry axis"
        ),
        canvas_r,
    ])

    return (canvas_r,)


@app.cell
def _(
    canvas_r,
    edit_state,
    focus,
    mo,
    np,
    review_controls,
    set_edit_state,
    tprf,
):
    _armed = bool(canvas_r.value)


    def _line_angle_from_selection(sel):
        """CCW degrees to bring the line drawn between two points to vertical.

        The user shift+drags a straight stroke from one midline point to the other.
        The stroke is a lasso path; we fit its principal axis (the straight line
        connecting the endpoints) and return the rotation that makes it vertical.
        """
        _verts = getattr(sel, "vertices", None)
        if not _verts or len(_verts) < 2:
            return None, "shift+drag a straight stroke between the two midline points"
        _pts = np.asarray(_verts, dtype=float)
        _c = _pts - _pts.mean(axis=0)
        _, _s, _vt = np.linalg.svd(_c, full_matrices=False)
        _vx, _vy = float(_vt[0, 0]), float(_vt[0, 1])
        _len = float(np.ptp(_c @ np.array([_vx, _vy])))
        if _len < 5:
            return None, "line too short — drag between two points across the midline"
        # image y points down, so negate it for a standard math-coords angle
        _phi = np.degrees(np.arctan2(-_vy, _vx))
        _theta = (90.0 - _phi) % 180.0
        if _theta > 90.0:
            _theta -= 180.0
        return _theta, f"line {_len:.0f}px \u00b7 rotate {_theta:+.2f}\u00b0 CCW"


    def _do_line(_):
        _i = focus.value
        _ang, _msg = _line_angle_from_selection(canvas_r.value)
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
            print("empty selection \u2014 shift+drag a lasso around the region to cut")
            return
        _e = dict(edit_state())
        _cur = dict(_e[_i])
        _cur["removals"] = _cur["removals"] + [_m]
        _e[_i] = _cur
        set_edit_state(_e)
        print(f"removed {int(_m.sum())} thumbnail px")


    def _do_undo(_):
        _i = focus.value
        _e = dict(edit_state())
        _cur = dict(_e[_i])
        if not _cur["removals"]:
            print("nothing to undo")
            return
        _last = _cur["removals"][-1]
        _cur["removals"] = _cur["removals"][:-1]
        _e[_i] = _cur
        set_edit_state(_e)
        print(f"undid last selection ({int(np.asarray(_last).sum())} px) \u2014 "
              f"{len(_cur['removals'])} removal(s) left")


    def _do_flip(_):
        _i = focus.value
        _e = dict(edit_state())
        _cur = dict(_e[_i])
        _cur["flip_lr"] = not _cur.get("flip_lr", False)
        _e[_i] = _cur
        set_edit_state(_e)
        print(f"section {_i + 1} flip L/R: {_cur['flip_lr']}")


    draw_sym_line = mo.ui.button(
        label="draw sym line", kind="warn" if _armed else "neutral",
        on_change=_do_line,
    )
    remove_selection = mo.ui.button(
        label="remove selection (lasso)", kind="warn" if _armed else "neutral",
        on_change=_do_remove,
    )
    undo_selection = mo.ui.button(
        label="undo selection", kind="neutral",
        on_change=_do_undo,
    )
    flip_lr_r = mo.ui.button(
        label="flip L/R", kind="neutral",
        on_change=_do_flip,
    )
    mo.vstack([
        mo.hstack([draw_sym_line, remove_selection, undo_selection, flip_lr_r],
                  justify="start", gap=0.5),
        mo.md("[\u2191 back to Slide Review](#slide-review)"),
    ])

    return


@app.cell
def _(brain_id_r, os, review_root_dir):
    review_root = os.path.join(str(review_root_dir), brain_id_r, "single_section_review")
    applied_dir = os.path.join(review_root, "imgs")
    review_transforms_dir = os.path.join(review_root, "review_transforms")
    return applied_dir, review_transforms_dir


@app.cell
def _(
    brain_id_r,
    edit_state,
    pl,
    review_controls,
    review_root_dir,
    section_paths,
    slide_num_r,
    tprf,
):
    transforms_df = tprf.load_slide_transforms(brain_id_r, slide_num_r,
                                               root=str(review_root_dir))
    decisions = tprf.collect_decisions(
            section_paths, review_controls, edits=edit_state(),
            brain_id=brain_id_r, slide_num=slide_num_r,
            transforms=transforms_df,
        )
    # Fold the Section Review flip (edit_state) into the grid flip (controls) so
    # the canvas, apply_decisions, and save_review_transforms all agree.
    _edit_flip = [bool(edit_state().get(i, {}).get("flip_lr", False))
                  for i in decisions["idx"].to_list()]
    decisions = decisions.with_columns(pl.Series("_edit_flip", _edit_flip))
    decisions = decisions.with_columns(
        (pl.col("flip_lr") ^ pl.col("_edit_flip")).alias("flip_lr")
    ).drop("_edit_flip")
    decisions

    return decisions, transforms_df


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
def _(
    brain_id_r,
    decisions,
    mo,
    review_transforms_dir,
    save_manifest_button,
    slide_num_r,
    tprf,
    transforms_df,
):
    mo.stop(not save_manifest_button.value, mo.md("*manifest not saved*"))
    _review_tf_path = tprf.save_review_transforms(
        decisions, transforms_df, review_transforms_dir, brain_id_r, slide_num_r
    )
    mo.md(f"saved review transforms → `{_review_tf_path}`")
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
    mo.vstack([
        apply_button,
        mo.md("[\u2191 back to Slide Review](#slide-review)"),
    ])

    return (apply_button,)


@app.cell
def _(
    applied_dir,
    apply_button,
    brain_id_r,
    decisions,
    edit_state,
    mo,
    review_transforms_dir,
    slide_num_r,
    tprf,
    transforms_df,
):
    mo.stop(not apply_button.value, mo.md("*no files written*"))
    written, skipped = tprf.apply_decisions(
        decisions, applied_dir, edits=edit_state(), dry_run=False, overwrite=False, verbose=True
    )
    tprf.save_review_transforms(
        decisions, transforms_df, review_transforms_dir, brain_id_r, slide_num_r
    )
    mo.md(f"wrote **{len(written)}** · skipped **{len(skipped)}**")
    return


if __name__ == "__main__":
    app.run()
