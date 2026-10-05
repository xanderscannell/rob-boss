from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DEFAULT_STYLE = {
    "fill": True,              # region fill
    "fill_rgb": (255, 255, 255),
    "fill_image": None,        # path/array: fill the region with this image's colours instead of
                               # fill_rgb (e.g. a layers.stack RGBA layer; its alpha is ignored)
    "fill_alpha": 0.35,        # fill brightness 0-1 (it's light, not paint: 1 = full colour)
    "outline": True,           # boundary lines
    "outline_mask": None,      # path/array: outline this region instead of the filled one (e.g. the
                               # part of a layer that stays visible, while the fill covers all of it)
    "outline_rgb": (255, 255, 255),
    "outline_px": 3,
    "arrows": False,           # stroke-direction arrows
    "stroke_dir_deg": 0.0,     # 0 = left-to-right, 90 = bottom-to-top (counter-clockwise, canvas up)
    "arrow_rgb": (255, 255, 0),
    "arrow_spacing_px": 120,
    "arrow_px": 3,
}


def load_mask(mask):
    """Path or array -> uint8 single-channel mask (255 = region)."""
    if isinstance(mask, (str, Path)):
        img = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(mask)
        return img
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.dtype == bool:
        return mask.astype(np.uint8) * 255
    return mask.astype(np.uint8)


def bgr(rgb):
    r, g, b = rgb
    return int(b), int(g), int(r)


def draw_arrows(img, region, deg, spacing, colour, thickness):
    """Arrows on a staggered grid, pointing along deg, kept only where they lie wholly in region."""
    h, w = region.shape
    d = np.array([np.cos(np.radians(deg)), -np.sin(np.radians(deg))])     # canvas y points down
    half = spacing * 0.35
    ys, xs = np.nonzero(region)
    if not len(xs):
        return
    for row, y in enumerate(np.arange(ys.min() + spacing / 2, ys.max(), spacing)):
        x0 = xs.min() + spacing / 2 + (spacing / 2 if row % 2 else 0)
        for x in np.arange(x0, xs.max(), spacing):
            p = np.array([x, y])
            a, b = p - d * half, p + d * half
            pts = [a, p, b]
            if all(0 <= q[0] < w and 0 <= q[1] < h and region[int(q[1]), int(q[0])] for q in pts):
                cv2.arrowedLine(img, tuple(int(v) for v in a), tuple(int(v) for v in b), colour,
                                thickness, cv2.LINE_AA, tipLength=0.35)


def render(mask, style=None, size=None):
    """Canvas-space overlay. `size` = (w, h) to render at; defaults to the mask's size."""
    st = {**DEFAULT_STYLE, **(style or {})}
    m = load_mask(mask)
    if size is not None and (m.shape[1], m.shape[0]) != tuple(size):
        m = cv2.resize(m, tuple(size), interpolation=cv2.INTER_LINEAR)
    region = m > 127
    img = np.zeros((*region.shape, 3), np.uint8)
    if st["fill"] and st["fill_image"] is not None:
        src = st["fill_image"]
        src = cv2.imread(str(src), cv2.IMREAD_COLOR) if isinstance(src, (str, Path)) else np.asarray(src)[..., :3]
        if src is None:
            raise FileNotFoundError(st["fill_image"])
        if src.shape[:2] != region.shape:
            src = cv2.resize(src, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_AREA)
        img[region] = (src[region].astype(np.float32) * float(st["fill_alpha"])).astype(np.uint8)
    elif st["fill"]:
        img[region] = np.array(bgr(st["fill_rgb"]), np.float32) * float(st["fill_alpha"])
    if st["arrows"]:
        draw_arrows(img, region, float(st["stroke_dir_deg"]), float(st["arrow_spacing_px"]),
                    bgr(st["arrow_rgb"]), int(st["arrow_px"]))
    if st["outline"]:
        edge = region
        if st["outline_mask"] is not None:
            om = load_mask(st["outline_mask"])
            if om.shape != region.shape:
                om = cv2.resize(om, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_LINEAR)
            edge = om > 127
        contours, _ = cv2.findContours(edge.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, bgr(st["outline_rgb"]), int(st["outline_px"]), cv2.LINE_AA)
    return img
