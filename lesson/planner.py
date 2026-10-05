from __future__ import annotations

import json
from pathlib import Path

from lesson.schema import Step
from layers.palette import describe_mix

# role -> (brush, technique, done when). "{dir}" is the stroke direction in words.
ROLES = {
    "base": ("2in flat", "Cover the whole canvas with a thin, even wash using long {dir} strokes. It's just a "
                         "happy little ground, so a few streaks are fine.",
             "no bare white canvas is showing anywhere"),
    "sky": ("2in flat", "Lay the colour in with {dir} criss-cross strokes, then soften where it meets the band "
                        "next to it while the paint is still wet.",
            "the whole band is covered, evenly, in the colour on the swatch"),
    "distant": ("1in flat", "Keep it soft and pale, it's far away. Block the shape in with {dir} strokes and "
                            "don't fuss over detail.",
                "the shape is filled in and stays lighter than anything in front of it"),
    "water": ("2in flat", "Pull long, straight {dir} strokes, so the water lies flat. Keep them level.",
              "the water is covered edge to edge with level strokes"),
    "near": ("1in flat", "This is the closest, darkest part of the picture. Fill it in with firm {dir} strokes "
                         "and plenty of paint.",
             "the shape is solidly covered and reads darker than the distance behind it"),
    "accent": ("liner brush", "Just touch these in with the tip of the brush, a few small {dir} strokes. "
                              "Less is more.",
               "each little highlight is in place"),
}
GENERIC = ("1in flat", "Fill this area in with {dir} strokes, keeping the colour even.",
           "the whole area is covered in the colour on the swatch")


def direction(axis_deg: float) -> str:
    """A stroke axis in degrees (0 = horizontal, 90 = vertical) in words."""
    a = axis_deg % 180
    if a < 25 or a > 155:
        return "horizontal"
    if 65 < a < 115:
        return "vertical"
    return "diagonal"


def plan(scene_dir: Path) -> list[Step]:
    """The lesson for a layers.stack scene: one step per layer, in the stack's order."""
    steps = []
    for i, r in enumerate(json.loads((Path(scene_dir) / "report.json").read_text())["steps"], 1):
        brush, technique, success = ROLES.get(r["role"], GENERIC)
        steps.append(Step(index=i, name=r["name"], mask_path=r["mask_path"],
                          target_rgb=tuple(int(v) for v in r["target_rgb"]), mix=r["mix"],
                          mix_description=describe_mix(r["mix"]), brush=brush,
                          technique=technique.format(dir=direction(r["stroke_dir_deg"])),
                          stroke_dir_deg=int(round(r["stroke_dir_deg"])) % 180, success=success))
    return steps
