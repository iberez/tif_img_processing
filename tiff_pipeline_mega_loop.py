import tiff_pipeline_pp_functions as tppf
import tiff_pipeline_bridge_functions as tpbf
import tiff_pipeline_ss_functions as tpssf

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
import gc
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

'''mega loop used to auto process raw microscope slide images to 
preprocessed, autorotated, and autoaligned single section images'''

def mega_loop(brain_id, root="/bigdata/isaac/rabies_img_processing"):
    _start_time_mega_loop =  time.perf_counter()
    #single slide all preprocessing functions
    #_brain_id = '141_02'
    _brain_id = brain_id
    _channel = 0
    #isaac's workdir for testing:
    _WORKDIR = Path(f"{root}/{str(_brain_id)}")
    #main workdir, see: /bigdata/microscope_images/Lin
    #_WORKDIR = Path(f"/bigdata/microscope_images/Lin/{str(_brain_id)}")

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
    _counter_l1  = 0

    #_slide_path = _all_slide_paths[0]
    for _slide_path in mo.status.progress_bar(
        _all_slide_paths, title="running preprocessing...",
        subtitle=f"brain {_brain_id}", completion_title="✅ preprocessing — Done",
        remove_on_exit=False,
    ):
        _start_time_l1 = time.perf_counter()
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

        #close any open figures out of precaution
        plt.close("all")
        del _down_area, _cleaned, _artifact_mask, _foreground, _tissue_mask
        del _repaired, _crack_stats, _separated, _split_stats
        del _final, _final_mask, _dewhiskered, _dewhiskered_mask
        gc.collect()

        _end_time_l1 = time.perf_counter()
        _execution_time_l1 = _end_time_l1 - _start_time_l1
        _execution_time_l1_fmt = timedelta(seconds = _execution_time_l1)
        _counter_l1+=1
        print (f'completed in {_execution_time_l1:6f} seconds')
        print (f'{_counter_l1}/{_num_slides}')

    mo.output.append(mo.md("✅ **preprocessing** — Done"))

    _end_time_all_slides = time.perf_counter()
    _execution_time_all_slides = _end_time_all_slides - _start_time_all_slides

    print (f'completed preprocessing of brain {_brain_id}, containing {_num_slides} raw tiff slide img files, in {_execution_time_all_slides:6f} seconds')

    print ('running bridge functions...')

    _BRIDGE_ROOT = Path(str(_annotated_output[2]))

    _meta_paths = sorted(_BRIDGE_ROOT.rglob("*_bridge_meta.json"))
    _slide_inputs = {}
    _inv_dfs = []
    for _p in mo.status.progress_bar(
        _meta_paths, title="configuring single section metadata...",
        subtitle=f"brain {_brain_id}", completion_title="✅ single section metadata — Done",
        remove_on_exit=False,
    ):
        _lm, _meta = tpbf.load_bridge_inputs(_p)
        _slide_inputs[_meta["slide_name"]] = (_lm, _meta)
        _inv_dfs.append(tpbf.inventory_sections(
            _lm, _meta["slide_name"], _meta["scale_y"], _meta["scale_x"]
        ))

    mo.output.append(mo.md("✅ **single section metadata** — Done"))
    print(f"{len(_inv_dfs)} slides, {sum(d.height for d in _inv_dfs)} sections total")

    _canvas_size = tpbf.plan_canvas(_inv_dfs, margin_frac=0.05, round_to=64)
    _single_section_out_dir = str(Path(_slide_out_folder).parent) + '/single_section_autoraw/'

    _manifests = []
    for _name, (_lm, _meta) in mo.status.progress_bar(
        list(_slide_inputs.items()), title="extracting single sections...",
        subtitle=f"brain {_brain_id}", completion_title="✅ extracting single sections — Done",
        remove_on_exit=False,
    ):
        print(_name)
        _inv = tpbf.inventory_sections(_lm, _name, _meta["scale_y"], _meta["scale_x"])
        _manifests.append(tpbf.extract_sections(
            slide_path=_meta["raw_slide_path"],
            slide_name=_name,
            label_mask=_lm,
            inv_df=_inv,
            canvas_size=_canvas_size,
            out_dir=_single_section_out_dir,
            scale_y=_meta["scale_y"],
            scale_x=_meta["scale_x"],
        ))
    _bridge_manifest = pl.concat(_manifests)
    # The per-slide label masks/inventories are no longer needed once the raw
    # crops are written; drop them so they are not held through autoalignment.
    del _slide_inputs, _inv_dfs, _manifests
    gc.collect()
    mo.output.append(mo.md("✅ **extracting single sections** — Done"))
    print ('done. Entering single section processing loop...')

    #autoraw single section dir
    _single_section_dir = f"{root}/{_brain_id}/single_section_autoraw/"

    _single_tiff_filenames = sorted(
        f for f in os.listdir(_single_section_dir)
        if f.endswith(".tiff") and f[:-len(".tiff")][-1:].isdigit()
    )
    _num_files = len(_single_tiff_filenames)

    #create directory to save plots,imgs, masks
    _save_dir = f'{root}/{_brain_id}/single_section_autoalign/'
    os.makedirs(_save_dir, exist_ok=True)

    _start_time_all_imgs = time.perf_counter()
    _counter_l2  = 0
    _tf_rows_by_slide = {}
    for _i,_stf in enumerate(mo.status.progress_bar(
        _single_tiff_filenames, title="processing single sections...",
        subtitle=f"brain {_brain_id}", completion_title="✅ processing single sections — Done",
        remove_on_exit=False,
    )):
        _slide_num = _single_tiff_filenames[_i].split('.')[0]
        _section_num = int(_single_tiff_filenames[_i].split('_')[2].split('.')[0])
        _start_time_l2 = time.perf_counter()

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
        _rotated_centroid, _rotated_centroid_mask, _rot_tf = tpssf.rotate_around_centroid(_initial, _mirror_angle, mask=_initial_mask,
                                                 save_dir=None,
                                                 slide_num=_slide_num, 
                                                   section_num=_section_num,
                                                   return_transform=True)
        print('done.')

        print('aligning centroid to image center...')
        _centered_img, _centered_mask, _ctr_tf = tpssf.center_img(_rotated_centroid_mask, _rotated_centroid, _initial,
                                                save_dir=_save_dir,
                                                 slide_num=_slide_num, 
                                                   section_num=_section_num,
                                                      brain_id=_brain_id,
                                                      return_transform=True)
        print(f'done.' )

        #track linear transforms so the mapping back to the raw slide is reproducible
        _tf_rows_by_slide.setdefault(_slide_num, []).append({
            "slide": _slide_num,
            "section": int(_section_num),
            "aligned_filename": f"{_brain_id}_img_slice_{_slide_num}_section_{_section_num}.tiff",
            **_rot_tf,
            **_ctr_tf,
        })


        # Release this section's arrays and the center_img() figure (plot=True
        # is not closed by the function) before the next section.
        plt.close("all")
        del _initial, _initial_mask, _res, _mirror_angle
        del _rotated_centroid, _rotated_centroid_mask, _rot_tf
        del _centered_img, _centered_mask, _ctr_tf
        gc.collect()

        _end_time_l2 = time.perf_counter()
        _execution_time_l2 = _end_time_l2 - _start_time_l2
        _execution_time_l2_fmt = timedelta(seconds = _execution_time_l2)
        _counter_l2+=1
        print (f'completed in {_execution_time_l2}')
        print (f'{_counter_l2}/{_num_files}')


    mo.output.append(mo.md("✅ **processing single sections** — Done"))

    _end_time_all_imgs = time.perf_counter()
    _execution_time_all_imgs = _end_time_all_imgs - _start_time_all_imgs
    _execution_time_all_imgs_fmt = timedelta(seconds = _execution_time_all_imgs)

    print (f'completed autoalignment of brain {_brain_id}, containing {_num_files} tiff img files, in {_execution_time_all_imgs}')

    print ('saving per-slide transform manifests (bridge metadata + autoalign transforms)...')
    for _sl_num, _sl_rows in _tf_rows_by_slide.items():
        tpssf.save_slide_transforms(_sl_rows, _brain_id, _sl_num, _save_dir,
                                    bridge_manifest=_bridge_manifest)


    _plot_dir = f'{root}/{_brain_id}/single_section_autoalign/plots/'

    print ('stiching autoaligned output into single pdf...')
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

    print(f"done. stitched {len(_pdf_entries)} plot(s) -> {_combined_path}")

    # Final sweep so nothing carries over to the next brain in a batch run.
    plt.close("all")
    gc.collect()

    _end_time_mega_loop =  time.perf_counter()

    _total_runtime = _end_time_mega_loop - _start_time_mega_loop
    _total_runtime_formatted = timedelta(seconds = _total_runtime)

    print (f'completed in {_total_runtime_formatted}')