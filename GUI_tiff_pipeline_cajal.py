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

    return Path, mo, os, pl, re


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
    Toggle **batch process** to queue several brains, or leave it off to process a single brain. Then click **run**.
    """)
    return


@app.cell
def _(mo):
    # browse to a root folder that contains one subfolder per brain id
    mega_root_browser = mo.ui.file_browser(
        initial_path="/bigdata/isaac/rabies_img_processing",
        selection_mode="directory",
        multiple=False,
        label="slide workdir root (contains brain_id folders)",
    )
    mega_root_browser
    return (mega_root_browser,)


@app.cell(hide_code=True)
def _(Path, mega_root_browser, re):
    # brain ids = brain-id-pattern subfolders (e.g. "129_02") of the selected
    # workdir root; mirrors the folder mega_loop reads raw slides from.
    mega_workdir_root = (
        Path(mega_root_browser.path(index=0))
        if mega_root_browser.value
        else Path("/bigdata/isaac/rabies_img_processing")
    )
    _brain_id_pat = re.compile(r"\d+_\d+$")
    mega_brain_ids = sorted(
        _d.name for _d in mega_workdir_root.iterdir()
        if _d.is_dir() and _brain_id_pat.fullmatch(_d.name)
    )
    return (mega_brain_ids,)


@app.cell(hide_code=True)
def _(mo):
    batch_process = mo.ui.switch(value=False, label="batch process")
    return (batch_process,)


@app.cell(hide_code=True)
def _(mega_brain_ids, mo):
    brain_id_single = mo.ui.dropdown(
        options=mega_brain_ids,
        value=mega_brain_ids[0] if mega_brain_ids else None,
        label="brain id to process",
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
        tpml.mega_loop(_bid)
    mo.md(f"✅ finished: **{', '.join(_ids)}**")
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
def _(brain_id_r, os, review_root_dir, slide_num_r):
    review_root = os.path.join(str(review_root_dir), brain_id_r, "single_section_review")
    applied_dir = os.path.join(review_root, "imgs")
    manifest_path = os.path.join(
        review_root, f"{brain_id_r}_slide_{slide_num_r}_review.csv"
    )
    return applied_dir, manifest_path


@app.cell
def _(
    brain_id_r,
    edit_state,
    review_controls,
    section_paths,
    slide_num_r,
    tprf,
):
    decisions = tprf.collect_decisions(
            section_paths, review_controls, edits=edit_state(),
            brain_id=brain_id_r, slide_num=slide_num_r,
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


if __name__ == "__main__":
    app.run()
