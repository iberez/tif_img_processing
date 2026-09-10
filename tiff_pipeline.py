import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import tifffile
    import zarr
    import dask.array as da

    # Remote box is headless (no DISPLAY): force the non-interactive backend
    # before pyplot is imported, or figure creation can fail.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pathlib import Path
    from skimage import exposure, filters, morphology, measure

    return (
        Path,
        da,
        exposure,
        filters,
        measure,
        mo,
        morphology,
        np,
        plt,
        tifffile,
    )


@app.cell
def _(mo):
    mo.md("""
    # Large TIFF viewer & pipeline bench

    Everything below executes on the remote host. Only rendered PNGs travel
    back through the SSH tunnel, so previews are downsampled and DPI-capped
    server-side before display.
    """)
    return


@app.cell
def _(Path, mo):
    # Browse the *remote* filesystem from the browser -- no need to know the
    # path in advance or shell in separately to look around.
    tiff_browser = mo.ui.file_browser(
        initial_path=Path("/bigdata/isaac/rabies_img_processing/129_02"),
        filetypes=[".tif", ".tiff", ".ome.tif"],
        multiple=False,
        label="Select a TIFF on the remote host",
    )
    tiff_browser
    return (tiff_browser,)


@app.cell
def _(mo, tiff_browser):
    mo.stop(not tiff_browser.value, mo.md("*Select a TIFF above to begin.*"))
    tiff_path = tiff_browser.path(index=0)
    tiff_path
    return (tiff_path,)


@app.cell
def _(np, tiff_path, tifffile):
    # Header-only inspection: reads metadata, not pixels. Cheap on any file size.
    with tifffile.TiffFile(tiff_path) as _tif:
        _series = _tif.series[0]
        tiff_info = {
            "shape": _series.shape,
            "dtype": str(_series.dtype),
            "axes": _series.axes,
            "n_pages": len(_tif.pages),
            "n_levels": len(_series.levels),
            "compression": str(_tif.pages[0].compression),
            "is_ome": _tif.is_ome,
            "gigabytes": round(
                np.prod(_series.shape) * _series.dtype.itemsize / 1e9, 2
            ),
        }
    tiff_info
    return (tiff_info,)


@app.cell
def _(da, mo, tiff_info, tifffile):
    # Lazy handle: zarr store -> dask array. No pixel data is read here.
    # For pyramidal TIFFs, level=N selects a pre-computed downsampled level --
    # far cheaper than decimating level 0.
    @mo.cache
    def open_lazy(path, level=0):
        store = tifffile.imread(path, aszarr=True, level=level)
        return da.from_zarr(store)

    pyramid_level = mo.ui.number(
        value=0, start=0, stop=max(0, tiff_info["n_levels"] - 1),
        label="Pyramid level (0 = full res)",
    )
    pyramid_level
    return open_lazy, pyramid_level


@app.cell
def _(open_lazy, pyramid_level, tiff_path):
    image = open_lazy(tiff_path, level=pyramid_level.value)
    image
    return (image,)


@app.cell
def _(mo):
    # Tunnel budget. max_dim caps pixels; dpi caps the PNG that gets shipped.
    # Both are upstream of nothing expensive, so they are cheap to move.
    max_dim = mo.ui.slider(
        256, 1536, value=768, step=128, label="Max preview dimension (px)"
    )
    fig_dpi = mo.ui.slider(50, 150, value=80, step=10, label="Figure DPI")
    channel = mo.ui.number(value=0, label="Channel / page index")
    mo.hstack([max_dim, fig_dpi, channel])
    return channel, fig_dpi, max_dim


@app.cell
def _(channel, fig_dpi, image, max_dim, plt):
    def to_2d(arr, ch):
        """Reduce an N-d stack to a single 2-D plane."""
        plane = arr
        while plane.ndim > 2:
            plane = plane[min(ch, plane.shape[0] - 1)]
        return plane

    def downsample(arr, max_px):
        """Strided decimation -- dask reads only the sampled chunks."""
        step = max(1, int(max(arr.shape) // max_px))
        return arr[::step, ::step]

    preview = downsample(to_2d(image, channel.value), max_dim.value).compute()

    plt.figure(figsize=(6, 6), dpi=fig_dpi.value)
    plt.imshow(preview, cmap="gray")
    plt.title(f"Preview {preview.shape} of {image.shape}")
    plt.axis("off")
    plt.gca()
    return (to_2d,)


@app.cell
def _(channel, image, mo, to_2d):
    # ROI selection, in coordinates of the currently loaded pyramid level.
    _h, _w = to_2d(image, channel.value).shape
    roi_y = mo.ui.range_slider(
        0, _h, value=[_h // 2 - 512, _h // 2 + 512], step=64,
        label="ROI rows", full_width=True,
    )
    roi_x = mo.ui.range_slider(
        0, _w, value=[_w // 2 - 512, _w // 2 + 512], step=64,
        label="ROI cols", full_width=True,
    )
    mo.vstack([roi_y, roi_x])
    return roi_x, roi_y


@app.cell
def _(channel, image, roi_x, roi_y, to_2d):
    roi = to_2d(image, channel.value)[
        roi_y.value[0]:roi_y.value[1],
        roi_x.value[0]:roi_x.value[1],
    ].compute()
    roi.shape
    return (roi,)


@app.cell
def _(mo):
    # Pipeline parameters -- tuned interactively against the ROI only.
    sigma = mo.ui.slider(0, 10, value=2, step=0.5, label="Gaussian sigma")
    min_size = mo.ui.slider(0, 500, value=50, step=10, label="Min object size")
    mo.hstack([sigma, min_size])
    return min_size, sigma


@app.cell
def _(exposure, filters, measure, morphology, np):
    def pipeline(plane, sigma_val, min_size_val):
        """Single source of truth: identical code runs on ROI and full-res."""
        rescaled = exposure.rescale_intensity(plane.astype(np.float32))
        smoothed = filters.gaussian(rescaled, sigma=sigma_val)
        binary = smoothed > filters.threshold_otsu(smoothed)
        cleaned = morphology.remove_small_objects(binary, min_size=min_size_val)
        labels = measure.label(cleaned)
        return smoothed, cleaned, labels

    return (pipeline,)


@app.cell
def _(fig_dpi, min_size, pipeline, plt, roi, sigma):
    roi_smoothed, roi_mask, roi_labels = pipeline(roi, sigma.value, min_size.value)

    _fig, _axes = plt.subplots(1, 3, figsize=(12, 4), dpi=fig_dpi.value)
    _axes[0].imshow(roi, cmap="gray")
    _axes[0].set_title("ROI (raw)")
    _axes[1].imshow(roi_smoothed, cmap="gray")
    _axes[1].set_title(f"Smoothed (sigma={sigma.value})")
    _axes[2].imshow(roi_labels, cmap="nipy_spectral")
    _axes[2].set_title(f"{roi_labels.max()} objects")
    for _ax in _axes:
        _ax.axis("off")
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    run_full = mo.ui.run_button(
        label="Run pipeline on FULL resolution", kind="danger"
    )
    mo.vstack([
        mo.md(
            "Full-resolution run is expensive and holds the plane in remote RAM. "
            "Tune on the ROI first. Results below are returned as a **table**, "
            "not an image -- no large render crosses the tunnel."
        ),
        run_full,
    ])
    return (run_full,)


@app.cell
def _(channel, image, measure, min_size, mo, pipeline, run_full, sigma, to_2d):
    mo.stop(not run_full.value, mo.md("*Not run yet.*"))

    import polars as pl

    full_plane = to_2d(image, channel.value).compute()
    _, _, full_labels = pipeline(full_plane, sigma.value, min_size.value)

    results = pl.DataFrame(
        measure.regionprops_table(
            full_labels,
            properties=("label", "area", "centroid", "eccentricity"),
        )
    )
    results
    return full_labels, results


@app.cell
def _(mo, run_full):
    mo.stop(not run_full.value, mo.md(""))

    # Persist on the remote next to the source image, rather than downloading.
    save_button = mo.ui.run_button(label="Save results + label mask to remote")
    save_button
    return (save_button,)


@app.cell
def _(Path, full_labels, mo, np, results, save_button, tiff_path, tifffile):
    mo.stop(not save_button.value, mo.md(""))

    _out = Path(tiff_path).with_suffix("")
    results.write_parquet(f"{_out}_objects.parquet")
    tifffile.imwrite(f"{_out}_labels.tif", full_labels.astype(np.int32))
    mo.md(f"Wrote `{_out}_objects.parquet` and `{_out}_labels.tif` on the remote.")
    return


if __name__ == "__main__":
    app.run()
