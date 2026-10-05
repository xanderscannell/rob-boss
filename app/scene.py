from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def scene_steps(scene: Path) -> list[dict]:
    """Steps in order: {name, mask, layer, rgb, stroke_deg}.

    stroke_deg is in overlay's convention (counter-clockwise, canvas up). layers.stack
    measures it in image coordinates (y down, so clockwise): negate it. It is an axis
    (0-180), not a direction: fold it into [-90, 90) so horizontal strokes point right
    and near-vertical ones point up.
    """
    axis = lambda deg: (-deg + 90.0) % 180.0 - 90.0
    return [{"name": s["name"].lower().replace(" ", "-"), "mask": scene / s["mask_path"],
             "layer": scene / s["layer_path"], "rgb": s.get("target_rgb"),
             "stroke_deg": axis(float(s["stroke_dir_deg"])) if "stroke_dir_deg" in s else None}
            for s in json.loads((scene / "report.json").read_text())["steps"]]


def compiled_frames(layer_paths, bare=None):
    """BGR float canvas after each layer, as layers.stack.paint lays them down:
    canvas = canvas * (1 - alpha) + colour * alpha, from bare canvas."""
    if bare is None:
        from layers.stack import BARE_CANVAS as bare
    canvas, frames = None, []
    for path in layer_paths:
        rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)   # BGRA
        if canvas is None:
            canvas = np.full(rgba.shape[:2] + (3,), float(bare), np.float32)
        alpha = rgba[..., 3:] / 255.0
        canvas = canvas * (1.0 - alpha) + rgba[..., :3] * alpha
        frames.append(canvas.copy())
    return frames


def masked_step(frame, layer_path, opacity):
    """The compiled frame, shown only where this layer is painted (BGR uint8). Mask = the
    layer's alpha / its opacity: 1 inside the pass, soft across its feathered edge."""
    alpha = cv2.imread(str(layer_path), cv2.IMREAD_UNCHANGED)[..., 3:].astype(np.float32) / 255.0
    mask = np.clip(alpha / max(opacity, 1e-3), 0.0, 1.0)
    return np.clip(np.round(frame * mask), 0, 255).astype(np.uint8)


def step_overlay(step, image, fill_only=False):
    """(region to fill, style) for one step's guide. `image`: the step's masked frame."""
    style = {"outline": not fill_only, "outline_px": 3,
             "arrows": not fill_only and step["stroke_deg"] is not None,
             "stroke_dir_deg": step["stroke_deg"] or 0.0,
             "fill_image": image, "fill_alpha": 1.0, "outline_mask": step["mask"]}
    return np.full(image.shape[:2], 255, np.uint8), style


def step_images(scene: Path, steps) -> list[np.ndarray]:
    """Each step's masked frame: the compiled picture, only where that step's layer is."""
    rows = json.loads((scene / "report.json").read_text())["steps"]
    frames = compiled_frames([s["layer"] for s in steps])
    return [masked_step(f, s["layer"], r["opacity"]) for f, s, r in zip(frames, steps, rows)]
