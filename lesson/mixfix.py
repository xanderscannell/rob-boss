from __future__ import annotations

import numpy as np

from lesson.schema import Step
from layers.stack import rgb_to_lab
from layers.palette import PIGMENTS

# L* of each tube straight from layers.palette (masstone, not a tint: a rough model anyway).
PIGMENT_L = {name: float(rgb_to_lab(np.array(rgb, np.float64))[0]) for name, rgb in PIGMENTS}
WHITE = "titanium white"
DARKEST = min(PIGMENT_L, key=PIGMENT_L.get)


def parts_to_add(total_parts: float, l_canvas: float, l_ref: float, l_pigment: float) -> float:
    """Parts of a pigment at `l_pigment` that move the mix from l_canvas to l_ref (capped at T)."""
    gap = l_pigment - l_ref
    if abs(gap) < 5.0 or (l_ref - l_canvas) * gap < 0:   # pigment can't get there from here
        return float(total_parts)
    return min(float(total_parts), max(0.0, total_parts * (l_ref - l_canvas) / gap))


def fmt_parts(p: float) -> str:
    halves = max(1, round(p * 2))                       # half-part resolution, at least a half
    if halves == 1:
        return "half a part"
    n = halves / 2
    text = f"{int(n)}" if halves % 2 == 0 else f"{n:.1f}"
    return f"{text} part{'s' if n > 1 else ''}"


def darkest_in_mix(step: Step) -> str:
    known = [m.pigment for m in step.mix if m.pigment in PIGMENT_L]
    return min(known, key=PIGMENT_L.get) if known else DARKEST


def advise(step: Step, l_canvas: float, l_ref: float) -> str:
    total = sum(m.parts for m in step.mix)
    if l_canvas < l_ref:
        pigment = WHITE
        word = "dark"
    else:
        pigment = darkest_in_mix(step)
        word = "light"
    p = parts_to_add(total, l_canvas, l_ref, PIGMENT_L.get(pigment, 50.0))
    return (f"That paint's just a touch too {word}, and that's okay - we'll fix it. Mix about "
            f"{fmt_parts(p)} of {pigment} into your {total}-part mix and give it another go.")
