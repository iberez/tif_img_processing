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

'''single section processing functions called in tiff_pipeline_cajal.py'''

#single section processing functions

def compute_binary_mask(
    img,
    channel=0,               # detection channel; None -> max-projection across channels
    method="li",             # "otsu" | "triangle" | "li" | numeric threshold
    smooth_sigma=2,          # Gaussian blur before thresholding (noise suppression)
    open_radius=2,           # morphological opening: removes speckle
    min_size_frac=0.0005,    # drop connected components smaller than this frac of image
    close_radius=5,          # close + fill holes so interiors stay solid
    dilate=2,                # grow mask slightly to keep dim edges
):
    """Compute a boolean foreground mask for `initial` (CYX or YX)."""
    def _disk(r):
        _l = np.arange(-r, r + 1)
        _x, _y = np.meshgrid(_l, _l)
        return (_x ** 2 + _y ** 2) <= r ** 2

    _arr = np.asarray(img)
    _is_stack = _arr.ndim == 3
    if _is_stack:
        _det = _arr.max(axis=0) if channel is None else _arr[channel]
    else:
        _det = _arr
    _det = ndi.gaussian_filter(_det.astype(np.float32), smooth_sigma)

    if isinstance(method, (int, float)):
        _thr = float(method)
    else:
        _thr = {"otsu": threshold_otsu, "triangle": threshold_triangle,
                "li": threshold_li}[method](_det)

    _mask = _det > _thr
    if open_radius:
        _mask = ndi.binary_opening(_mask, structure=_disk(open_radius))

    _lbl, _n = ndi.label(_mask)
    if _n > 0:
        _sizes = np.bincount(_lbl.ravel())
        _min_size = int(min_size_frac * _det.size)
        _keep = _sizes >= _min_size
        _keep[0] = False
        _mask = _keep[_lbl]

    if close_radius:
        _mask = ndi.binary_closing(_mask, structure=_disk(close_radius))
    _mask = ndi.binary_fill_holes(_mask)
    if dilate:
        _mask = ndi.binary_dilation(_mask, structure=_disk(dilate))

    return _mask

def dilate_to_ellipse(
    mask,
    max_iter=200,
    iou_target=0.9,
    patience=5,
    structure=None,
    return_info=False,
):
    """Iteratively dilate a binary mask until its shape best matches an ellipse.

    On each step we fit the equivalent ellipse to the (single largest) component
    of the current mask -- centroid, orientation and axis lengths from
    regionprops -- rasterise that ellipse, and measure the IoU between the
    dilated mask and its fitted ellipse. Dilation continues until the IoU stops
    improving (early stop with `patience`) or crosses `iou_target`.
    """
    def _fit_ellipse(_m):
        _lbl = label(_m)
        _props = regionprops(_lbl)
        if not _props:
            return None, None
        _p = max(_props, key=lambda r: r.area)
        _yy, _xx = np.mgrid[0:_m.shape[0], 0:_m.shape[1]]
        _cy, _cx = _p.centroid
        _theta = _p.orientation
        _a = max(_p.axis_major_length / 2.0, 1e-6)
        _b = max(_p.axis_minor_length / 2.0, 1e-6)
        _yr = _yy - _cy
        _xr = _xx - _cx
        # skimage orientation: angle between the y-axis and the major axis
        _u = np.cos(_theta) * _xr - np.sin(_theta) * _yr
        _v = np.sin(_theta) * _xr + np.cos(_theta) * _yr
        _ell = (_u / _b) ** 2 + (_v / _a) ** 2 <= 1.0
        return _ell, _p

    _mask = np.asarray(mask).astype(bool)
    _struct = structure if structure is not None else ndi.generate_binary_structure(2, 1)

    _best_iou = -1.0
    _best_mask = _mask.copy()
    _history = []
    _since_improve = 0

    _cur = _mask.copy()
    for _i in range(max_iter):
        _ell, _p = _fit_ellipse(_cur)
        if _ell is None:
            break
        _inter = np.logical_and(_cur, _ell).sum()
        _union = np.logical_or(_cur, _ell).sum()
        _iou = _inter / _union if _union else 0.0
        _history.append(_iou)

        if _iou > _best_iou:
            _best_iou = _iou
            _best_mask = _cur.copy()
            _since_improve = 0
        else:
            _since_improve += 1

        if _iou >= iou_target or _since_improve >= patience:
            break

        _cur = ndi.binary_dilation(_cur, structure=_struct)

    if return_info:
        return _best_mask, {"best_iou": float(_best_iou),
                            "n_iter": len(_history),
                            "iou_history": _history}
    return _best_mask

def compute_rot_angle_regionprops(mask, plot=True):
    """Estimate a rotation angle from the mask's major axis (via regionprops).

    The centroid is computed exactly as in `compute_rot_angle`. A regionprops
    table (`props_rp`) is built with centroid, orientation and axis lengths.
    The rotation angle `rot_angle_rp` is the tilt of the fitted major axis
    relative to the horizontal line drawn through the centroid.
    """
    _m = np.asarray(mask) > 0
    _ys, _xs = np.nonzero(_m)

    # Centroid (same convention as compute_rot_angle).
    cy_rp, cx_rp = _ys.mean(), _xs.mean()

    # Build props dataframe, just like `props`.
    props_rp = regionprops_table(
        _m.astype(np.uint8),
        properties=('centroid', 'orientation',
                    'axis_major_length', 'axis_minor_length'),
    )
    props_rp = pd.DataFrame(props_rp)

    _orient = float(props_rp['orientation'].iloc[0])
    _major = float(props_rp['axis_major_length'].iloc[0])
    _minor = float(props_rp['axis_minor_length'].iloc[0])

    # Major-axis direction vector (skimage convention, matching the axis plot).
    _dx, _dy = np.sin(_orient), -np.cos(_orient)
    # Angle of the major axis relative to the horizontal line through centroid.
    rot_angle_rp = -float(np.degrees(np.arctan2(_dy, _dx)))

    if plot:
        _H, _W = _m.shape
        _fig_rp, _ax_rp = plt.subplots(figsize=(8, 10))
        _ax_rp.imshow(_m, cmap="gray")

        # Horizontal line through the centroid.
        _ax_rp.plot([0, _W - 1], [cy_rp, cy_rp], "-", color="cyan", lw=2,
                    label="l_centroid_horizontal")

        # Major axis through the centroid.
        _x1 = cx_rp + _dx * 0.5 * _major
        _y1 = cy_rp + _dy * 0.5 * _major
        _x2 = cx_rp - _dx * 0.5 * _major
        _y2 = cy_rp - _dy * 0.5 * _major
        _ax_rp.plot([_x1, _x2], [_y1, _y2], "-", color="red", lw=2,
                    label=f"major axis ({rot_angle_rp:.2f}°)")

        _ax_rp.plot(cx_rp, cy_rp, "o", color="yellow", markersize=8,
                    label="centroid")

        _ax_rp.set_xlim(0, _W)
        _ax_rp.set_ylim(_H, 0)
        _ax_rp.set_title(f"compute_rot_angle_regionprops — major-axis tilt "
                         f"{rot_angle_rp:.2f}°")
        _ax_rp.legend(loc="upper right")
        _ax_rp.axis("off")
        plt.tight_layout()

    return rot_angle_rp, props_rp, cy_rp, cx_rp

def rotate_around_centroid(
    img,
    angle_deg,
    fill=0,
    mask=None,
    slide_num=None, section_num=None,
    save_dir = None,
    return_transform=False,   # True -> also return the applied linear transform
):
    """Rotate an image about an arbitrary centroid (cx, cy) by `angle_deg`.

    Works on a single YX plane or a CYX stack (each channel rotated the same
    way so the stack stays aligned). Rotation is counter-clockwise for a
    positive angle, pivoting on the supplied centroid rather than the image
    centre. The supplied mask is rotated with the same transform and returned
    alongside the rotated image.
    """
    _m = np.asarray(mask) > 0
    _ys, _xs = np.nonzero(_m)
    cy, cx = _ys.mean(), _xs.mean()

    _arr = np.asarray(img)
    _is_stack = _arr.ndim == 3

    _h = _arr.shape[1] if _is_stack else _arr.shape[0]
    _w = _arr.shape[2] if _is_stack else _arr.shape[1]
    _M = cv2.getRotationMatrix2D((float(cx), float(cy)), float(angle_deg), 1.0)

    def _rotate(plane):
        return cv2.warpAffine(
            plane.astype(np.float32), _M, (_w, _h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float(fill),
        )

    if _is_stack:
        _out = np.stack([_rotate(_arr[_c]) for _c in range(_arr.shape[0])], axis=0)
    else:
        _out = _rotate(_arr)

    _out = _out.astype(_arr.dtype)

    _out_mask = cv2.warpAffine(
        _m.astype(np.float32), _M, (_w, _h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    ) > 0.5

    if save_dir:
        _ch_rc = 0
        _base_rc = (_out[_ch_rc] if _out.ndim == 3 else _out).astype(np.float32)
        _orig_rc = (_arr[_ch_rc] if _arr.ndim == 3 else _arr).astype(np.float32)
        _vmax_rc = (np.percentile(_orig_rc[mask], 99)
                    if (mask is not None and mask.any()) else _orig_rc.max())

        _fig_rc, _ax_rc = plt.subplots(1, 2, figsize=(14, 7))
        _ax_rc[0].imshow(_orig_rc, cmap="gray", vmax=_vmax_rc)
        _ax_rc[0].plot(cx, cy, "o", color="yellow", markersize=7, label="centroid")
        _ax_rc[0].set_title(f"original ch{_ch_rc}")
        _ax_rc[0].legend(loc="upper right")
        _ax_rc[1].imshow(_base_rc, cmap="gray", vmax=_vmax_rc)
        _ax_rc[1].plot(cx, cy, "o", color="yellow", markersize=7)
        _ax_rc[1].set_title(f"rotated {angle_deg:.2f}° about centroid")
        for _a in _ax_rc:
            _a.axis("off")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/autorotated_slice_{slide_num}_section_{section_num}.pdf")

    if return_transform:
        # CCW rotation by angle_deg about (rot_center_x, rot_center_y), in
        # canvas pixel coords. Invert with a CW rotation about the same point.
        _transform = {
            "auto_rot_ccw_deg": float(angle_deg),
            "rot_center_x": float(cx),
            "rot_center_y": float(cy),
        }
        return _out, _out_mask, _transform
    return _out, _out_mask

def compute_rot_angle(mask, plot=True):
    """Estimate a rotation angle from the tangent to the mask's top points.

    A vertical line `l_centroid_split` (x = cx) is drawn through the centroid.
    On each side of that line we take the mask pixel with the smallest y
    (highest point). Those two extrema define a tangent line `l_tangent`; the
    angle of that tangent relative to horizontal is the rotation angle.
    """
    _m = np.asarray(mask) > 0
    _ys, _xs = np.nonzero(_m)

    #compute centroid
    #ys, xs = np.nonzero(mask > 0)
    cy, cx = _ys.mean(), _xs.mean()   
    _left = _xs < cx
    _right = _xs >= cx
    if not _left.any() or not _right.any():
        raise ValueError("centroid split does not separate mask into two sides")

    # Top-most (min y) pixel on each side of the vertical split line.
    _li = np.argmin(_ys[_left])
    _ri = np.argmin(_ys[_right])
    x_l, y_l = float(_xs[_left][_li]), float(_ys[_left][_li])
    x_r, y_r = float(_xs[_right][_ri]), float(_ys[_right][_ri])

    # Tangent line through the two extrema.
    _dx, _dy = (x_r - x_l), (y_r - y_l)
    rot_angle = float(np.degrees(np.arctan2(_dy, _dx)))

    # Intersection of l_tangent with l_centroid_split (x = cx).
    if abs(_dx) < 1e-9:
        _iy = 0.5 * (y_l + y_r)
    else:
        _iy = y_l + _dy * (cx - x_l) / _dx
    intersection = (float(cx), float(_iy))

    result = {
        "rot_angle_deg": rot_angle,
        "left_point": (x_l, y_l),
        "right_point": (x_r, y_r),
        "intersection": intersection,
    }

    if plot:
        _H, _W = _m.shape
        _fig_rot, _ax_rot = plt.subplots(figsize=(8, 10))
        _ax_rot.imshow(_m, cmap="gray")

        # l_centroid_split: vertical line through the centroid.
        _ax_rot.plot([cx, cx], [0, _H - 1], "-", color="cyan", lw=2,
                     label="l_centroid_split")

        # l_tangent: extended across the mask width through the two extrema.
        _t = _W
        _tx0, _ty0 = x_l - _dx * _t, y_l - _dy * _t
        _tx1, _ty1 = x_r + _dx * _t, y_r + _dy * _t
        _ax_rot.plot([_tx0, _tx1], [_ty0, _ty1], "-", color="red", lw=2,
                     label=f"l_tangent ({rot_angle:.2f}°)")

        _ax_rot.plot([x_l, x_r], [y_l, y_r], "o", color="lime", markersize=7,
                     label="side extrema")
        _ax_rot.plot(cx, cy, "o", color="yellow", markersize=8, label="centroid")
        _ax_rot.plot(*intersection, "x", color="magenta", markersize=12,
                     markeredgewidth=3, label="intersection")

        _ax_rot.set_xlim(0, _W)
        _ax_rot.set_ylim(_H, 0)
        _ax_rot.set_title(f"compute_rot_angle — tangent tilt {rot_angle:.2f}°")
        _ax_rot.legend(loc="upper right")
        _ax_rot.axis("off")
        plt.tight_layout()

    return rot_angle, result, cy, cx

def _block_max_downsample(mask, max_dim=512):
    """Downsample a binary mask by block-max so thin structures survive."""
    _m = np.asarray(mask, dtype=bool)
    _h, _w = _m.shape
    _f = int(np.ceil(max(_h, _w) / max_dim))
    if _f <= 1:
        return _m
    _ph, _pw = (-_h) % _f, (-_w) % _f
    _p = np.pad(_m, ((0, _ph), (0, _pw)), constant_values=False)
    _H, _W = _p.shape
    return _p.reshape(_H // _f, _f, _W // _f, _f).max(axis=(1, 3))

def _mirror_iou(m):
    """IoU of a mask with its reflection about the best-fitting vertical line.

    The flip column comes from the autoconvolution of the column profile,
    so an off-centre midline does not bias the score.
    """
    _prof = m.sum(axis=0).astype(np.float64)
    if _prof.sum() == 0:
        return 0.0, np.nan
    _k = int(np.argmax(np.convolve(_prof, _prof, mode="full")))
    _W = m.shape[1]
    _src = np.arange(_W)
    _dst = _k - _src
    _ok = (_dst >= 0) & (_dst < _W)
    _flip = np.zeros_like(m)
    _flip[:, _dst[_ok]] = m[:, _src[_ok]]
    _inter = np.logical_and(m, _flip).sum()
    _union = np.logical_or(m, _flip).sum()
    return (float(_inter) / _union if _union else 0.0), _k / 2.0

def find_midline_rotation(mask, angle_range=(-30.0, 30.0),
                          coarse_step=1.0, fine_step=0.1, max_dim=512):
    """Find the rotation that stands a bilaterally symmetric mask upright.

    `rotation_deg` is the value to pass straight to ndimage.rotate — it is
    defined by the search itself, so there is no sign convention to guess.
    """
    _m0 = _block_max_downsample(np.asarray(mask) > 0, max_dim).astype(np.float32)

    def _score(theta):
        _r = ndi.rotate(_m0, theta, reshape=True, order=1,
                            mode="constant", cval=0.0) > 0.5
        return _mirror_iou(_r)[0]

    _lo, _hi = angle_range
    _coarse = np.arange(_lo, _hi + coarse_step / 2, coarse_step)
    _cs = np.array([_score(_t) for _t in _coarse])
    _peak = float(_coarse[int(np.argmax(_cs))])

    _fine = np.arange(_peak - coarse_step,
                      _peak + coarse_step + fine_step / 2, fine_step)
    _fs = np.array([_score(_t) for _t in _fine])

    return {
        "rotation_deg": float(_fine[int(np.argmax(_fs))]),
        "iou": float(_fs.max()),
        "peak_contrast": float(_fs.max() - np.median(_cs)),
        "curve_angles": _coarse,
        "curve_scores": _cs,
        }

def compute_symmetry_line_pca(mask, img, channel=0, plot=True,
                              slide_num=None, section_num=None,
                              save_dir=None):
    """Find a mask's principal axis via PCA and draw it on the image.

    Returns the axis tilt in degrees relative to the horizontal, using
    screen convention: positive means the line rises to the right.
    """
    _m = np.asarray(mask) > 0
    _ys, _xs = np.nonzero(_m)

    cy_sym, cx_sym = _ys.mean(), _xs.mean()

    _coords = np.column_stack([_xs.astype(np.float64) - cx_sym,
                               _ys.astype(np.float64) - cy_sym])

    _cov = np.cov(_coords, rowvar=False)
    _evals, _evecs = np.linalg.eigh(_cov)
    _principal = _evecs[:, np.argmax(_evals)]

    _dx, _dy = float(_principal[0]), float(_principal[1])
    if _dx < 0:
        _dx, _dy = -_dx, -_dy

    sym_angle = float(np.degrees(np.arctan2(-_dy, _dx)))

    _H, _W = _m.shape
    _t = float(max(_H, _W))
    _p0 = (cx_sym - _dx * _t, cy_sym - _dy * _t)
    _p1 = (cx_sym + _dx * _t, cy_sym + _dy * _t)

    result = {
        "sym_angle_deg": sym_angle,
        "centroid": (float(cx_sym), float(cy_sym)),
        "principal_vector": (_dx, _dy),
        "eigenvalue_ratio": float(np.sqrt(_evals.max() / max(_evals.min(), 1e-12))),
        "endpoints": (_p0, _p1),
    }

    if plot:
        _base = (img[channel] if img.ndim == 3 else img).astype(np.float32)
        _vmax = np.percentile(_base[_m], 99) if _m.any() else max(_base.max(), 1)

        _fig, _ax = plt.subplots(figsize=(8, 10))
        _ax.imshow(_base, cmap="gray", vmax=_vmax)
        _ax.plot([_p0[0], _p1[0]], [_p0[1], _p1[1]], "-", color="red",
                 lw=2, label=f"principal axis ({sym_angle:.2f}°)")
        _ax.plot(cx_sym, cy_sym, "o", color="yellow", markersize=8,
                 label="centroid")
        _ax.set_xlim(0, _W)
        _ax.set_ylim(_H, 0)
        _ax.set_title(f"PCA principal axis — {sym_angle:.2f}°")
        _ax.legend(loc="upper right")
        _ax.axis("off")
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(f"{save_dir}/sym_pca_slice_{slide_num}_section_{section_num}.png")

    return sym_angle, result, cy_sym, cx_sym

def center_img(mask, img, initial_img, channel=0, brain_id = None, plot=True,
               slide_num=None, section_num=None,
               save_dir=None,
               return_transform=False,   # True -> also return the applied linear transform
               ):
    """Center a mask/image so the mask centroid aligns with the image center.

    Computes the centroid exactly as in `compute_symmetry_line_pca`, the image
    center as (width/2, height/2), then shifts both the mask and image so the
    centroid lands on the image center. Returns the centered image and mask.
    """
    dot_size = 5
    _m = np.asarray(mask) > 0
    _ys, _xs = np.nonzero(_m)

    cy_c, cx_c = _ys.mean(), _xs.mean()

    _arr = np.asarray(img)
    _is_stack = _arr.ndim == 3
    _H = _arr.shape[1] if _is_stack else _arr.shape[0]
    _W = _arr.shape[2] if _is_stack else _arr.shape[1]

    center_coords = (_W / 2.0, _H / 2.0)  # (x, y)

    _shift_x = center_coords[0] - cx_c
    _shift_y = center_coords[1] - cy_c

    _M = np.array([[1.0, 0.0, _shift_x],
                   [0.0, 1.0, _shift_y]], dtype=np.float32)

    def _shift_plane(plane, is_mask=False):
        _flag = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        return cv2.warpAffine(
            plane.astype(np.float32), _M, (_W, _H),
            flags=_flag,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

    if _is_stack:
        _centered_img = np.stack(
            [_shift_plane(_arr[_c]) for _c in range(_arr.shape[0])], axis=0
        ).astype(_arr.dtype)
    else:
        _centered_img = _shift_plane(_arr).astype(_arr.dtype)

    _centered_mask = _shift_plane(_m.astype(np.float32), is_mask=True) > 0.5

    _new_ys, _new_xs = np.nonzero(_centered_mask)
    if _new_ys.size:
        _new_cy, _new_cx = _new_ys.mean(), _new_xs.mean()
    else:
        _new_cy, _new_cx = center_coords[1], center_coords[0]

    if plot:
        _base_initial = (initial_img[channel] if initial_img.ndim == 3
                         else initial_img).astype(np.float32)
        _base_before = (img[channel] if img.ndim == 3 else img).astype(np.float32)
        _base_after = (_centered_img[channel] if _centered_img.ndim == 3
                       else _centered_img).astype(np.float32)

        # Centroid of img (rotated) from its display channel.
        _ry, _rx = np.nonzero(_base_before > 0)
        if _ry.size:
            cy_r, cx_r = _ry.mean(), _rx.mean()
        else:
            cy_r, cx_r = center_coords[1], center_coords[0]

        # Centroid of initial_img from its display channel.
        _iy, _ix = np.nonzero(_base_initial > 0)
        if _iy.size:
            cy_i, cx_i = _iy.mean(), _ix.mean()
        else:
            cy_i, cx_i = center_coords[1], center_coords[0]

        _vmax = np.percentile(_base_before[_m], 99) if _m.any() else max(_base_before.max(), 1)

        # Line through the centroid and image center, extended across the image.
        def _centroid_center_line(_cx, _cy):
            _ddx = center_coords[0] - _cx
            _ddy = center_coords[1] - _cy
            _norm = np.hypot(_ddx, _ddy)
            if _norm < 1e-9:
                return None
            _ux, _uy = _ddx / _norm, _ddy / _norm
            _t = float(max(_H, _W))
            return ([_cx - _ux * _t, center_coords[0] + _ux * _t],
                    [_cy - _uy * _t, center_coords[1] + _uy * _t])

        _fig, _ax = plt.subplots(1, 3, figsize=(24, 8))
        _ax[0].imshow(_base_initial, cmap="gray", vmax=_vmax)
        _line_i = _centroid_center_line(cx_i, cy_i)
        if _line_i is not None:
            _ax[0].plot(_line_i[0], _line_i[1], "--", color="red", lw=1.5,
                        label="centroid → center")
        _ax[0].plot(cx_i, cy_i, "o", color="yellow", markersize=dot_size, label="centroid")
        _ax[0].plot(center_coords[0], center_coords[1], "o", color="lime",
                    markersize=dot_size+2, label="image center")
        _ax[0].set_title("initial image", y= 0.95, color = 'white')
        _ax[0].legend(loc="upper right")

        _ax[1].imshow(_base_before, cmap="gray", vmax=_vmax)
        _line_r = _centroid_center_line(cx_r, cy_r)
        if _line_r is not None:
            _ax[1].plot(_line_r[0], _line_r[1], "--", color="red", lw=1.5,
                        label="centroid → center")
        _ax[1].plot(cx_r, cy_r, "o", color="yellow", markersize=dot_size, label="centroid")
        _ax[1].plot(center_coords[0], center_coords[1], "o", color="lime",
                    markersize=dot_size+2, label="image center")
        _ax[1].set_title("centroid rotated", y= 0.95, color = 'white')
        _ax[1].legend(loc="upper right")

        _ax[2].imshow(_base_after, cmap="gray", vmax=_vmax)
        _ax[2].plot([_new_cx, _new_cx], [0, _H], "--", color="red", lw=1.5,
                    label="centroid vertical")
        _ax[2].plot(center_coords[0], center_coords[1], "o", color="lime",
                    markersize=dot_size+2, label="image center")
        _ax[2].plot(_new_cx, _new_cy, "o", color="yellow", markersize=3, label="centroid")
        _ax[2].set_title("centroid rotated + centered", y= 0.95, color = 'white')
        _ax[2].legend(loc="upper right")
        #add main title
        _fig.suptitle(f'slide: {slide_num} section: {section_num}', x = 0.825, y = 0.8, fontsize = 30, color = 'gold')

        for _a in _ax:
            _a.set_xlim(0, _W)
            _a.set_ylim(_H, 0)
            _a.axis("off")
        plt.tight_layout()
        if save_dir is not None:
            _plot_out_folder = save_dir + "/plots/"
            _mask_out_folder = save_dir + "/masks/"
            _img_out_folder = save_dir + "/imgs/"
            os.makedirs(_plot_out_folder, exist_ok=True)
            os.makedirs(_mask_out_folder, exist_ok=True)
            os.makedirs(_img_out_folder, exist_ok=True)

            plt.savefig(f"{_plot_out_folder}/{brain_id}_autorotated_centered_slice_{slide_num}_section_{section_num}.pdf")
            np.save(f'{_mask_out_folder}/{brain_id}_mask_slice_{slide_num}_section_{section_num}' , _centered_mask)
            tifffile.imwrite(
                f'{_img_out_folder}/{brain_id}_img_slice_{slide_num}_section_{section_num}.tiff',
                _centered_img,
                photometric="minisblack",
                metadata={"axes": "CYX" if _centered_img.ndim == 3 else "YX"},
            )

        plt.close(_fig)   # free the figure so it does not accumulate across sections

    if return_transform:
        # Pure translation by (shift_x, shift_y), in canvas pixel coords.
        # Invert by translating (-shift_x, -shift_y).
        _transform = {
            "shift_x": float(_shift_x),
            "shift_y": float(_shift_y),
            "centroid_x_pre_center": float(cx_c),
            "centroid_y_pre_center": float(cy_c),
            "centroid_x_post_center": float(_new_cx),
            "centroid_y_post_center": float(_new_cy),
            "canvas_w": int(_W),
            "canvas_h": int(_H),
        }
        return _centered_img, _centered_mask, _transform
    return _centered_img, _centered_mask


def save_slide_transforms(
    transform_rows,
    brain_id,
    slide_num,
    save_dir,
    bridge_manifest=None,
    verbose=True,
):
    """Write one CSV per slide recording the autoalignment linear transforms.

    One row per section. Columns combine:

    * bridge metadata from extract_sections() (crop_y0/crop_x0 = canvas origin
      in raw slide coords, centroid_*_raw, scale_y/scale_x, canvas_size, ...)
    * the rotation applied by rotate_around_centroid():
      auto_rot_ccw_deg about (rot_center_x, rot_center_y)
    * the translation applied by center_img(): (shift_x, shift_y)

    To map an autoaligned section image back onto the raw slide, invert the
    forward chain in reverse order:

    1. translate by (-shift_x, -shift_y)                     [undo center_img]
    2. rotate auto_rot_ccw_deg degrees CW about
       (rot_center_x, rot_center_y)              [undo rotate_around_centroid]
    3. add (crop_x0, crop_y0) to x/y coords to go from canvas coords to raw
       slide coords                              [undo extract_sections crop]

    Review-time edits (flip / extra rotation / removals) are tracked separately
    by tiff_pipeline_review_functions.collect_decisions(), which merges this
    CSV into the review manifest.

    Returns the path of the written CSV.
    """
    df = pl.DataFrame(transform_rows).sort("section")
    if bridge_manifest is not None and bridge_manifest.height:
        bm = bridge_manifest.filter(pl.col("slide") == str(slide_num))
        if "filename" in bm.columns:
            bm = bm.rename({"filename": "autoraw_filename"})
        if bm.height:
            df = bm.join(df, on=["slide", "section"], how="full", coalesce=True).sort("section")
        elif verbose:
            print(f"  WARNING: no bridge manifest rows for slide {slide_num}; "
                  f"transforms saved without bridge metadata")

    out_dir = Path(save_dir) / "transforms"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{brain_id}_slide_{slide_num}_transforms.csv"
    df.write_csv(out_path)
    if verbose:
        print(f"  wrote {df.height} section transforms -> {out_path}")
    return str(out_path)
