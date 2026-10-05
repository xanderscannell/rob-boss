from __future__ import annotations

import itertools
from functools import lru_cache
from typing import Sequence

import numpy as np

# The Floral White palette as used on the show, eyeballed to sRGB.
PIGMENTS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("titanium white", (252, 252, 250)),
    ("midnight black", (20, 20, 22)),
    ("phthalo blue", (18, 40, 96)),
    ("prussian blue", (14, 46, 86)),
    ("phthalo green", (16, 66, 54)),
    ("sap green", (62, 96, 40)),
    ("cadmium yellow", (250, 206, 24)),
    ("yellow ochre", (198, 150, 48)),
    ("indian yellow", (226, 150, 40)),
    ("bright red", (206, 44, 38)),
    ("alizarin crimson", (142, 24, 38)),
    ("van dyke brown", (62, 44, 34)),
    ("dark sienna", (60, 32, 28)),
)

PIGMENT_NAMES = tuple(name for name, _ in PIGMENTS)

# Reflectance has to stay off 0 and 1 or K/S blows up.
_FLOOR = 0.004
_CEILING = 0.996

# Ratios run to 384:1 because tinting strength is wildly uneven: phthalo blue
# is strong enough that a pale sky really is a speck of it in a tub of white,
# and a low ceiling leaves every sky visibly over-saturated. Ratios that large
# are not sayable as "parts", so `describe_mix` renders them as "a touch of".
_PAIR_RATIOS = tuple(
    (a, b)
    for a in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384)
    for b in (1, 2, 3, 4)
    if a >= b
)
_TRIPLE_UNITS = (1, 2, 4, 8, 16, 32, 64, 128)
TOUCH_RATIO = 24
SPECK_RATIO = 96


def _to_linear(rgb: np.ndarray) -> np.ndarray:
    scaled = np.asarray(rgb, dtype=np.float64) / 255.0
    return np.where(scaled <= 0.04045, scaled / 12.92, ((scaled + 0.055) / 1.055) ** 2.4)


def _to_srgb(linear: np.ndarray) -> np.ndarray:
    clipped = np.clip(linear, 0.0, 1.0)
    encoded = np.where(
        clipped <= 0.0031308, clipped * 12.92, 1.055 * clipped ** (1.0 / 2.4) - 0.055
    )
    return encoded * 255.0


def _ks(reflectance: np.ndarray) -> np.ndarray:
    r = np.clip(reflectance, _FLOOR, _CEILING)
    return (1.0 - r) ** 2 / (2.0 * r)


def _reflectance(ks: np.ndarray) -> np.ndarray:
    k = np.maximum(ks, 0.0)
    return 1.0 + k - np.sqrt(k * k + 2.0 * k)


def _pigment_ks() -> np.ndarray:
    return _ks(_to_linear(np.array([rgb for _, rgb in PIGMENTS], dtype=np.float64)))


def mix_rgb(parts: Sequence[tuple[int, int]]) -> np.ndarray:
    """sRGB of a mix given (pigment index, parts) pairs."""
    table = _pigment_ks()
    total = float(sum(count for _, count in parts))
    blended = np.zeros(3, dtype=np.float64)
    for index, count in parts:
        blended += (count / total) * table[index]
    return _to_srgb(_reflectance(blended))


def _normalise(parts: Sequence[int]) -> tuple[int, ...]:
    divisor = np.gcd.reduce(np.array(parts, dtype=np.int64))
    return tuple(int(value // max(1, divisor)) for value in parts)


def _recipes() -> list[tuple[tuple[int, int], ...]]:
    count = len(PIGMENTS)
    recipes: set[tuple[tuple[int, int], ...]] = set()

    for index in range(count):
        recipes.add(((index, 1),))
    for first, second in itertools.combinations(range(count), 2):
        for ratio in _PAIR_RATIOS:
            a, b = _normalise(ratio)
            recipes.add(((first, a), (second, b)))
            recipes.add(((first, b), (second, a)))
    for triple in itertools.combinations(range(count), 3):
        for units in itertools.product(_TRIPLE_UNITS, repeat=3):
            a, b, c = _normalise(units)
            recipes.add(((triple[0], a), (triple[1], b), (triple[2], c)))

    return sorted(recipes)


@lru_cache(maxsize=1)
def _table() -> tuple[list[tuple[tuple[int, int], ...]], np.ndarray]:
    """Every candidate recipe and its colour, as a Lab lookup table."""
    from layers.stack import rgb_to_lab  # late import: layers imports this module

    recipes = _recipes()
    weights = np.zeros((len(recipes), len(PIGMENTS)), dtype=np.float64)
    for row, recipe in enumerate(recipes):
        for index, parts in recipe:
            weights[row, index] = parts
    weights /= weights.sum(axis=1, keepdims=True)
    colours = _to_srgb(_reflectance(weights @ _pigment_ks()))
    return recipes, rgb_to_lab(colours)


def nearest_mix(target_rgb: Sequence[float]) -> tuple[list[dict[str, object]], np.ndarray]:
    """Closest achievable mix to a target colour.

    Returns the recipe as `step.json` wants it, plus the colour the mix actually
    produces -- which is what gets painted, not the target.
    """
    from layers.stack import rgb_to_lab

    recipes, lab_table = _table()
    target = rgb_to_lab(np.array(target_rgb, dtype=np.float64).reshape(1, 3))
    distances = np.sum((lab_table - target) ** 2, axis=-1)
    best = int(np.argmin(distances))
    recipe = recipes[best]
    mix = [
        {"pigment": PIGMENT_NAMES[index], "parts": int(parts)}
        for index, parts in sorted(recipe, key=lambda item: -item[1])
    ]
    return mix, mix_rgb(recipe)


def describe_mix(mix: Sequence[dict[str, object]]) -> str:
    """Say the mix the way a painter would, not the way the ratio reads.

    "384 parts titanium white to 1 part phthalo blue" is an arithmetically
    correct and completely unusable instruction. Past a ratio nobody could
    measure, the modifier becomes a touch or a speck.
    """
    if not mix:
        return ""
    base, *rest = mix
    if not rest:
        return str(base["pigment"])

    base_parts = int(base["parts"])
    if all(base_parts / int(entry["parts"]) < TOUCH_RATIO for entry in rest):
        return " to ".join(
            f"{int(entry['parts'])} part{'s' if int(entry['parts']) != 1 else ''} "
            f"{entry['pigment']}"
            for entry in mix
        )

    modifiers = []
    for entry in rest:
        ratio = base_parts / int(entry["parts"])
        if ratio >= SPECK_RATIO:
            modifiers.append(f"a speck of {entry['pigment']}")
        elif ratio >= TOUCH_RATIO:
            modifiers.append(f"a touch of {entry['pigment']}")
        else:
            modifiers.append(f"{int(entry['parts'])} parts {entry['pigment']}")
    if len(modifiers) == 1:
        tail = modifiers[0]
    else:
        tail = ", ".join(modifiers[:-1]) + " and " + modifiers[-1]
    return f"{base['pigment']} with {tail}"
