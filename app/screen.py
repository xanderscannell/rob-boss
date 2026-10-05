from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import overlay

GUIDE_FILL_ALPHA = 90          # 0-255: how strongly the step's colours ghost over the canvas
GUIDE_LINES = {"outline_rgb": (30, 30, 30), "arrow_rgb": (194, 65, 12)}


class Screen:
    def __init__(self, panel, size: tuple[int, int], bare: int):
        self.panel, self.bare = panel, bare
        panel.canvas_setup(size, bare)

    def canvas(self) -> np.ndarray:
        return self.panel.canvas()

    def show_step(self, mask, style=None):
        st = {**(style or {}), **GUIDE_LINES}
        fill = overlay.render(mask, {**st, "outline": False, "arrows": False})
        lines = overlay.render(mask, {**st, "fill": False})
        on_line, on_fill = lines.any(2), fill.any(2)
        bgr = np.where(on_line[..., None], lines, fill)
        alpha = np.where(on_line, 255, np.where(on_fill, GUIDE_FILL_ALPHA, 0)).astype(np.uint8)
        self.panel.guide(np.dstack([bgr, alpha]))

    def save(self, path: Path) -> bool:
        """Write the painting to `path` as a PNG; False (nothing written) if it is still blank."""
        img = self.canvas()
        if (img == self.bare).all():
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(cv2.imencode(".png", img)[1].tobytes())   # not imwrite: it can't do non-ASCII paths
        return True
