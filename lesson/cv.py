from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_BARE_RGB = (245, 240, 230)


@dataclass
class Measurement:
    checkable: bool            # False if the step has (almost) nothing visible to paint
    coverage: float            # painted fraction of the pixels that need paint (0..1)
    delta_l: float | None      # mean L*(canvas) - L*(reference) over painted pixels; <0 = too dark
    delta_e: float | None      # colour distance between mean painted colour and reference mean
    missing: np.ndarray        # bool HxW at mask size: needs paint but is still bare
    painted_px: int
    mean_l_canvas: float | None = None  # mean L* of the painted pixels (0-100)
    mean_l_ref: float | None = None     # mean L* of the reference at those same pixels
    value_off: np.ndarray | None = None  # bool HxW: painted areas whose lightness is off locally


def to_lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def bare_lab(bare_rgb=DEFAULT_BARE_RGB) -> np.ndarray:
    return to_lab(np.array([[bare_rgb]], np.uint8))[0, 0]


def estimate_bare_rgb(canvas_rgb: np.ndarray) -> tuple[int, int, int]:
    """Calibrate from an (empty) canvas capture: per-channel median."""
    return tuple(int(v) for v in np.median(canvas_rgb.reshape(-1, 3), axis=0))


def measure(canvas_rgb: np.ndarray, ref_rgb: np.ndarray, mask: np.ndarray, *,
            bare_rgb=DEFAULT_BARE_RGB, bare_image: np.ndarray | None = None,
            before_rgb: np.ndarray | None = None, paint_de: float = 12.0, erode_px: int = 5,
            min_needed_px: int = 200, min_painted_px: int = 50,
            value_dl: float = 12.0, blur_px: int = 25) -> Measurement:
    h, w = mask.shape[:2]
    canvas = cv2.resize(canvas_rgb, (w, h), interpolation=cv2.INTER_AREA)
    ref = cv2.resize(ref_rgb, (w, h), interpolation=cv2.INTER_AREA)
    region = mask > 127
    if erode_px > 1:  # ignore a thin border so small registration error doesn't read as bare canvas
        region = cv2.erode(region.astype(np.uint8), np.ones((erode_px, erode_px), np.uint8)).astype(bool)

    lab_c, lab_r, lab_b = to_lab(canvas), to_lab(ref), bare_lab(bare_rgb)
    # Painted = changed from bare. With a capture of the empty canvas (bare_image), compare
    # each pixel with that same spot bare: the capture light is not even (brighter centre,
    # darker edges), so one bare colour would count dim-but-bare paper as paint.
    # In a layer stack the "bare" capture is the canvas as the step began (earlier steps'
    # paint included), and before_rgb is the expected picture before the step: only what
    # the step changes needs paint.
    lab_bare_px = lab_b if bare_image is None else         to_lab(cv2.resize(bare_image, (w, h), interpolation=cv2.INTER_AREA))
    lab_before = lab_b if before_rgb is None else         to_lab(cv2.resize(before_rgb, (w, h), interpolation=cv2.INTER_AREA))
    painted_any = np.linalg.norm(lab_c - lab_bare_px, axis=2) > paint_de
    needs_paint = region & (np.linalg.norm(lab_r - lab_before, axis=2) > paint_de)

    if needs_paint.sum() < min_needed_px:
        return Measurement(False, 1.0, None, None, np.zeros_like(region), 0)

    painted = needs_paint & painted_any
    coverage = float(painted.sum() / needs_paint.sum())
    delta_l = delta_e = mean_lc = mean_lr = value_off = None
    if painted.sum() >= min_painted_px:
        mean_c, mean_r = lab_c[painted].mean(axis=0), lab_r[painted].mean(axis=0)
        delta_l = float(mean_c[0] - mean_r[0])
        delta_e = float(np.linalg.norm(mean_c - mean_r))
        mean_lc, mean_lr = float(mean_c[0]), float(mean_r[0])
        value_off = _local_value_off(lab_c[..., 0] - lab_r[..., 0], painted, value_dl, blur_px)
    return Measurement(True, coverage, delta_l, delta_e, needs_paint & ~painted_any, int(painted.sum()),
                       mean_lc, mean_lr, value_off)


def _local_value_off(diff_l: np.ndarray, painted: np.ndarray, threshold: float, blur_px: int) -> np.ndarray:
    """Where is the lightness error large locally? Blur over painted pixels only, so brush
    texture doesn't speckle the map and bare canvas doesn't bleed into it."""
    k = blur_px | 1
    weight = cv2.GaussianBlur(painted.astype(np.float32), (k, k), 0)
    smooth = cv2.GaussianBlur(diff_l * painted, (k, k), 0) / np.maximum(weight, 1e-3)
    return painted & (weight > 0.2) & (np.abs(smooth) > threshold)


