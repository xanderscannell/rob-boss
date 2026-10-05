from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from layers.palette import describe_mix, nearest_mix

MAX_EDGE = 1024
CLUSTER_COUNT = 24
KMEANS_ITERATIONS = 24
SMOOTHING_PASSES = 3
SMOOTHING_RADIUS = 3
MIN_STAGE_FRACTION = 0.025
ACCENT_PERCENTILE = 99.3
# Just enough to take the canvas weave and sensor grain off the target, and no
# more. This used to be a radius-10 box blur, on the reasoning that a layer is
# one worked colour family rather than a photograph -- but that blur stripped
# edges along with texture and cost ~95% of the scene's high-frequency content
# before any layer was built. Reducing each layer to three palette mixes does
# the flattening now, and does it without destroying silhouettes.
TEXTURE_RADIUS = 2
DETAIL_RADIUS = 2
MAX_FEATHER = 22
MIN_FEATHER = 2
RNG_SEED = 7
BARE_CANVAS = 245.0

# How thickly each kind of pass goes down. Nothing but the final accents is
# fully opaque -- a thin pass lets the ground wash and the layer behind it show
# through, which is where the optical mixing comes from.
ROLE_OPACITY = {
    "base": 0.55,
    "sky": 0.80,
    "distant": 0.80,
    "water": 0.85,
    "near": 0.90,
    "accent": 1.00,
}

# Mixes a painter is willing to keep on the brush for one pass. A sky is two
# mixes blended wet; a foreground mass can carry three.
ROLE_MIX_LIMIT = {"base": 1, "sky": 3, "distant": 2, "water": 3, "near": 3, "accent": 2}
MAX_MIXES_PER_STEP = 3

# Returning to a role you already finished is the signature of a scene with no
# depth planes. Three re-entries is a sky revisited after a ridge; eight is a
# decomposition that has lost the plot.
MAX_ROLE_REENTRIES = 3

# Radius the mixes are worked into each other across. Without this a two-mix
# sky has a hard line down the middle of it.
# Hard-assigning each pixel to its nearest mix gives flat fields meeting at a
# line. A painter blends the two mixes into each other across the transition,
# so weight them by how close the target is to each. The palette is still only
# those mixes; the gradient between them is made on the canvas.
#
# This replaced a per-role box blur over the quantized field. The blur blended
# the mixes, but it blended the silhouettes with them: it cost roughly a third
# of the scene's surviving high-frequency content and bought nothing in ΔE,
# because it softens the edges where the target genuinely jumps -- exactly the
# edges a painting needs. Distance weighting smooths the gradients and leaves
# the jumps alone.
MIX_BLEND_DELTA_E = 14.0

MIX_MERGE_DELTA_E = 7.0
MIN_MIX_FRACTION = 0.05

# A thin pass cannot put a dark over a light: solving for it asks for pigment
# darker than black. Where that happens over much of a layer, the answer is the
# painter's answer -- lay it on thicker -- not to clip and accept a grey tree.
UNREACHABLE_TOLERANCE = 0.08


# --------------------------------------------------------------------------
# colour


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float64) / 255.0
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    linear = srgb_to_linear(rgb)
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = linear @ matrix.T
    white = np.array([0.95047, 1.00000, 1.08883])
    scaled = xyz / white
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(
        scaled > epsilon,
        np.cbrt(np.clip(scaled, 1e-12, None)),
        (kappa * scaled + 16.0) / 116.0,
    )
    lightness = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([lightness, a, b], axis=-1)


def delta_e76(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((first - second) ** 2, axis=-1))


# --------------------------------------------------------------------------
# separable box operations (integral images)


def box_sum(values: np.ndarray, radius: int) -> np.ndarray:
    """Sum over a (2r+1) square window, edge-clamped."""
    if radius < 1:
        return values.astype(np.float64)
    padded = np.pad(values.astype(np.float64), radius + 1, mode="edge")
    if values.ndim != 2:
        raise ValueError("box_sum expects a 2-D array")
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    height, width = values.shape
    size = 2 * radius + 1
    top = 0
    left = 0
    bottom = top + size
    right = left + size
    return (
        integral[bottom : bottom + height, right : right + width]
        - integral[top : top + height, right : right + width]
        - integral[bottom : bottom + height, left : left + width]
        + integral[top : top + height, left : left + width]
    )


def box_mean(values: np.ndarray, radius: int) -> np.ndarray:
    count = box_sum(np.ones(values.shape[:2]), radius)
    return box_sum(values, radius) / count


def masked_blur(colour: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    """Blur a colour field using only pixels inside the mask.

    This is the within-layer blend: a painter mixes one colour family and works
    it across the wet region, so the layer keeps its gradient but loses texture.
    """
    weight = mask.astype(np.float64)
    denominator = box_sum(weight, radius)
    denominator = np.where(denominator < 1e-9, 1.0, denominator)
    channels = [
        box_sum(colour[..., index] * weight, radius) / denominator for index in range(3)
    ]
    return np.stack(channels, axis=-1)


# --------------------------------------------------------------------------
# morphology


def _shifts(mask: np.ndarray) -> list[np.ndarray]:
    up = np.zeros_like(mask)
    up[:-1] = mask[1:]
    down = np.zeros_like(mask)
    down[1:] = mask[:-1]
    left = np.zeros_like(mask)
    left[:, :-1] = mask[:, 1:]
    right = np.zeros_like(mask)
    right[:, 1:] = mask[:, :-1]
    return [up, down, left, right]


def dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    current = mask.copy()
    for _ in range(iterations):
        neighbours = _shifts(current)
        current = current | neighbours[0] | neighbours[1] | neighbours[2] | neighbours[3]
    return current


def inner_distance(mask: np.ndarray, limit: int) -> np.ndarray:
    """Distance from each mask pixel to the nearest outside pixel.

    The canvas edge is not an edge of the *region*: a layer that runs off the
    side of the picture was cropped, not feathered, so erosion must not eat in
    from the border or every layer ends up with a transparent frame.
    """
    distance = np.zeros(mask.shape, dtype=np.float64)
    current = mask.copy()
    for step in range(limit):
        up, down, left, right = _shifts(current)
        up[-1] = current[-1]
        down[0] = current[0]
        left[:, -1] = current[:, -1]
        right[:, 0] = current[:, 0]
        eroded = current & up & down & left & right
        distance += eroded.astype(np.float64)
        current = eroded
        if not current.any():
            break
    return distance


def grassfire_fill(
    colour: np.ndarray, known: np.ndarray, target: np.ndarray, limit: int = 600
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a colour field outward from `known` to cover `target`.

    This is the amodal extension: the sky continues behind the mountain, so the
    sky layer needs a plausible colour there even though nobody will ever see it.
    """
    filled = colour.copy()
    settled = known.copy()
    remaining = target & ~settled
    for _ in range(limit):
        if not remaining.any():
            break
        masked = filled * settled[..., None]
        neighbour_count = np.zeros(settled.shape, dtype=np.float64)
        accumulator = np.zeros(colour.shape, dtype=np.float64)
        for shifted_weight, shifted_colour in zip(_shifts(settled), _shift_colour(masked)):
            neighbour_count += shifted_weight.astype(np.float64)
            accumulator += shifted_colour
        newly = remaining & (neighbour_count > 0)
        if not newly.any():
            break
        safe = np.where(neighbour_count < 1e-9, 1.0, neighbour_count)
        averaged = accumulator / safe[..., None]
        filled[newly] = averaged[newly]
        settled = settled | newly
        remaining = remaining & ~newly
    return filled, settled


def _shift_colour(colour: np.ndarray) -> list[np.ndarray]:
    up = np.zeros_like(colour)
    up[:-1] = colour[1:]
    down = np.zeros_like(colour)
    down[1:] = colour[:-1]
    left = np.zeros_like(colour)
    left[:, :-1] = colour[:, 1:]
    right = np.zeros_like(colour)
    right[:, 1:] = colour[:, :-1]
    return [up, down, left, right]


# --------------------------------------------------------------------------
# connected components (run-length + union-find; fast enough in pure python)


def component_areas(mask: np.ndarray) -> list[int]:
    height, width = mask.shape
    parent: list[int] = []
    area: list[int] = []

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root == second_root:
            return
        parent[second_root] = first_root
        area[first_root] += area[second_root]
        area[second_root] = 0

    previous_runs: list[tuple[int, int, int]] = []
    for y in range(height):
        row = mask[y]
        if not row.any():
            previous_runs = []
            continue
        padded = np.concatenate(([False], row, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        starts, ends = edges[0::2], edges[1::2]
        current_runs: list[tuple[int, int, int]] = []
        for start, end in zip(starts.tolist(), ends.tolist()):
            label = len(parent)
            parent.append(label)
            area.append(end - start)
            for previous_start, previous_end, previous_label in previous_runs:
                if previous_start < end and start < previous_end:
                    union(previous_label, label)
            current_runs.append((start, end, label))
        previous_runs = current_runs

    return sorted(
        (area[label] for label in range(len(parent)) if parent[label] == label and area[label] > 0),
        reverse=True,
    )


# --------------------------------------------------------------------------
# scene analysis


def estimate_horizon(lightness: np.ndarray) -> int:
    height = lightness.shape[0]
    profile = lightness.mean(axis=1)
    smoothed = np.convolve(profile, np.ones(9) / 9.0, mode="same")
    gradient = np.abs(np.gradient(smoothed))
    low, high = int(height * 0.25), int(height * 0.85)
    window = gradient[low:high]
    return int(low + int(np.argmax(window)))


def kmeans(features: np.ndarray, clusters: int, iterations: int) -> np.ndarray:
    generator = np.random.default_rng(RNG_SEED)
    sample_count = features.shape[0]
    centres = features[generator.choice(sample_count, clusters, replace=False)].copy()
    assignment = np.zeros(sample_count, dtype=np.int32)
    for _ in range(iterations):
        distances = np.empty((sample_count, clusters), dtype=np.float32)
        for index in range(clusters):
            difference = features - centres[index]
            distances[:, index] = np.einsum("ij,ij->i", difference, difference)
        updated = np.argmin(distances, axis=1).astype(np.int32)
        if np.array_equal(updated, assignment):
            break
        assignment = updated
        for index in range(clusters):
            members = features[assignment == index]
            if members.size:
                centres[index] = members.mean(axis=0)
    return assignment


def majority_smooth(labels: np.ndarray, count: int, radius: int, passes: int) -> np.ndarray:
    current = labels
    for _ in range(passes):
        scores = np.stack(
            [box_sum((current == index).astype(np.float64), radius) for index in range(count)],
            axis=-1,
        )
        current = np.argmax(scores, axis=-1).astype(np.int32)
    return current


# --------------------------------------------------------------------------
# stages


@dataclass
class Stage:
    index: int
    name: str
    role: str
    paint_mask: np.ndarray
    visible_mask: np.ndarray
    feather: int
    opacity: float
    depth: float
    # What this pass is trying to be: the reference colour over everything it
    # covers, extended amodally behind its occluders.
    target: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # What it actually puts down: at most three palette mixes, worked together.
    colour: np.ndarray = field(default_factory=lambda: np.zeros(0))
    alpha: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mixes: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def describe_region(
    lab: np.ndarray, rgb: np.ndarray, mask: np.ndarray, horizon: int
) -> dict[str, float]:
    ys, xs = np.nonzero(mask)
    height = mask.shape[0]
    lightness = lab[..., 0][mask]
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)[mask]
    return {
        "y_centroid": float(ys.mean() / height),
        "y_base": float(np.percentile(ys, 90) / height),
        "lightness": float(lightness.mean()),
        "chroma": float(chroma.mean()),
        "lightness_spread": float(lightness.std()),
        "above_horizon": float(np.count_nonzero(ys < horizon) / max(1, ys.size)),
    }


def name_region(
    stats: dict[str, float],
    blueness: float,
    crispness: float,
    sky_lightness: float = 100.0,
) -> tuple[str, str]:
    """Label a region by its place in a landscape, from aerial-perspective cues.

    Distance washes a region out and softens its edges, so a dark *crisp* mass is
    foreground no matter how high in frame it sits -- that is the near conifer,
    not the far treeline.
    """
    above = stats["above_horizon"]
    lightness = stats["lightness"]
    chroma = stats["chroma"]
    crisp = crispness > 0.52

    if crisp and lightness < 34:
        return "Foreground mass", "near"
    if above > 0.85:
        # Terrain reads as a silhouette: darker than the sky it sits against.
        # That comparison has to be relative, because a sunset sky is not a
        # noon sky and no absolute lightness threshold survives both.
        if lightness < sky_lightness - 12.0:
            return ("Far treeline", "distant") if lightness < 30.0 else ("Distant ridges", "distant")
        if lightness > sky_lightness + 14.0 and chroma > 20:
            return "Sun", "accent"
        if chroma > 40:
            return "Sunset band", "sky"
        if blueness > 6:
            return "Upper sky", "sky"
        return "Cloud bank", "sky"
    if above > 0.4:
        if lightness < 34:
            return "Far treeline", "distant"
        return "Distant ridges", "distant"
    if lightness < 28:
        return "Foreground mass", "near"
    if chroma > 34:
        return "Sun path on water", "water"
    return "Water plane", "water"


def build_stages(
    rgb: np.ndarray, lab: np.ndarray, labels: np.ndarray, cluster_count: int, horizon: int
) -> list[Stage]:
    height, width = labels.shape
    canvas_area = height * width
    lightness = lab[..., 0]
    sky_lightness = float(lightness[: max(1, int(height * 0.35))].mean())
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    contrast = np.sqrt(
        np.maximum(box_mean(lightness**2, 6) - box_mean(lightness, 6) ** 2, 0.0)
    )

    records = []
    for index in range(cluster_count):
        mask = labels == index
        area = int(np.count_nonzero(mask))
        if area < canvas_area * 0.002:
            continue
        stats = describe_region(lab, rgb, mask, horizon)
        normalised_chroma = min(1.0, stats["chroma"] / 70.0)
        normalised_contrast = min(1.0, float(contrast[mask].mean()) / 18.0)
        # Aerial perspective: distant regions sit high, wash out, and flatten.
        # Ground-plane contact, not centroid: a conifer is tall and its centroid
        # rides high, but its *base* is low in frame, and that is what says near.
        # A ridge meets the ground at the horizon, which says far.
        depth = (
            0.38 * (1.0 - stats["y_base"])
            + 0.18 * (1.0 - normalised_chroma)
            + 0.32 * (1.0 - normalised_contrast)
            + 0.12 * min(1.0, stats["lightness"] / 100.0)
        )
        # Above the waterline, height in frame *is* the depth order: sky, then
        # the bands below it, then the ridges, then the treeline at its foot.
        if stats["above_horizon"] > 0.92 and normalised_contrast < 0.45:
            depth = 1.0 + 0.5 * (1.0 - stats["y_centroid"])
        blueness = float(-lab[..., 2][mask].mean())
        records.append(
            {
                "cluster": index,
                "mask": mask,
                "area": area,
                "depth": depth,
                "stats": stats,
                "blueness": blueness,
                "crispness": normalised_contrast,
            }
        )

    records.sort(key=lambda item: item["depth"], reverse=True)

    # Merge neighbouring-in-depth clusters until every stage is worth a step.
    groups: list[list[dict[str, Any]]] = []
    pending: list[dict[str, Any]] = []
    pending_area = 0
    for record in records:
        pending.append(record)
        pending_area += record["area"]
        if pending_area >= canvas_area * MIN_STAGE_FRACTION:
            groups.append(pending)
            pending, pending_area = [], 0
    if pending:
        if groups:
            groups[-1].extend(pending)
        else:
            groups.append(pending)

    def group_profile(group: list[dict[str, Any]]) -> tuple[np.ndarray, str, str, float]:
        mask = np.zeros(labels.shape, dtype=bool)
        for record in group:
            mask |= record["mask"]
        weight = float(sum(record["area"] for record in group))
        crispness = sum(record["crispness"] * record["area"] for record in group) / weight
        blueness = sum(record["blueness"] * record["area"] for record in group) / weight
        name, role = name_region(
            describe_region(lab, rgb, mask, horizon), blueness, crispness, sky_lightness
        )
        return mask, name, role, float(np.mean([record["depth"] for record in group]))

    # A painter does not stop and remix between two adjacent passes of the same
    # material, so fold consecutive same-role groups into one step.
    merged: list[list[dict[str, Any]]] = []
    for group in groups:
        _, _, role, depth = group_profile(group)
        if merged:
            _, _, previous_role, previous_depth = group_profile(merged[-1])
            combined_area = sum(record["area"] for record in merged[-1] + group)
            close_in_depth = abs(previous_depth - depth) < 0.05
            if (
                role == previous_role
                and close_in_depth
                and combined_area < canvas_area * 0.24
            ):
                merged[-1].extend(group)
                continue
        merged.append(list(group))
    groups = merged

    stages: list[Stage] = []

    base_colour = np.empty_like(rgb, dtype=np.float64)
    base_colour[:] = rgb.reshape(-1, 3).mean(axis=0)
    stages.append(
        Stage(
            index=1,
            name="Ground wash",
            role="base",
            paint_mask=np.ones(labels.shape, dtype=bool),
            visible_mask=np.zeros(labels.shape, dtype=bool),
            feather=0,
            opacity=ROLE_OPACITY["base"],
            depth=1.0,
            target=base_colour,
        )
    )

    for group in groups:
        mask, name, role, depth = group_profile(group)
        stages.append(
            Stage(
                index=0,
                name=name,
                role=role,
                paint_mask=mask,
                visible_mask=mask.copy(),
                feather=0,
                opacity=ROLE_OPACITY.get(role, 0.85),
                depth=depth,
            )
        )

    # Final accents: the thick, bright, last-touched marks.
    threshold = float(np.percentile(lightness, ACCENT_PERCENTILE))
    accent = lightness >= threshold
    accent = dilate(accent, 1)
    if np.count_nonzero(accent) > canvas_area * 0.0004:
        stages.append(
            Stage(
                index=0,
                name="Highlights and accents",
                role="accent",
                paint_mask=accent,
                visible_mask=accent.copy(),
                feather=MIN_FEATHER,
                opacity=ROLE_OPACITY["accent"],
                depth=0.0,
            )
        )

    # Deduplicate repeated names so step titles stay readable.
    seen: dict[str, int] = {}
    for position, stage in enumerate(stages, start=1):
        stage.index = position
        seen[stage.name] = seen.get(stage.name, 0) + 1
        if seen[stage.name] > 1:
            stage.name = f"{stage.name} {seen[stage.name]}"
    return stages


def resolve_occlusion(stages: Sequence[Stage]) -> None:
    """Visible region = what nothing in front of it covers."""
    for position, stage in enumerate(stages):
        covered = np.zeros(stage.paint_mask.shape, dtype=bool)
        for later in stages[position + 1 :]:
            if later.role == "accent" and later is not stages[-1]:
                continue
            covered |= later.paint_mask
        stage.visible_mask = stage.paint_mask & ~covered


def extend_layers(stages: Sequence[Stage], rgb: np.ndarray) -> None:
    """Give every layer a target colour field and an alpha, seen or not.

    This is geometry and intent only. What pigment ends up on the brush is
    decided later by `paint`, because it depends on what is already wet.
    """
    for position, stage in enumerate(stages):
        if stage.role == "base":
            stage.alpha = np.full(stage.paint_mask.shape, stage.opacity)
            continue

        visible = stage.visible_mask if stage.visible_mask.any() else stage.paint_mask
        # The layer in front is translucent across its own feather band, so this
        # layer is genuinely seen there and must carry the *real* colour -- not
        # an extrapolation. Seed from the visible region plus that band.
        source = dilate(visible, MAX_FEATHER)
        blurred = masked_blur(rgb.astype(np.float64), source, TEXTURE_RADIUS)

        # Paint behind whatever sits in front of this layer.
        occluded = np.zeros(stage.paint_mask.shape, dtype=bool)
        for later in stages[position + 1 :]:
            occluded |= later.paint_mask
        extended_region = stage.paint_mask | (occluded & _behind_envelope(stage, stages, position))
        target, settled = grassfire_fill(blurred, source, extended_region)
        stage.target = target
        stage.paint_mask = stage.paint_mask | (settled & extended_region)

        # Aerial perspective also governs edge softness: a hazy distant
        # transition blends wide, a near silhouette stays crisp.
        softness = stage.depth
        feather = int(round(MIN_FEATHER + (MAX_FEATHER - MIN_FEATHER) * softness))
        if stage.role in ("near", "accent"):
            feather = MIN_FEATHER
        stage.feather = max(MIN_FEATHER, feather)
        distance = inner_distance(stage.paint_mask, stage.feather)
        ramp = np.clip(distance / float(stage.feather), 0.0, 1.0)
        stage.alpha = np.where(stage.paint_mask, np.maximum(ramp, 0.0), 0.0) * stage.opacity
        interior = stage.paint_mask & (distance >= stage.feather)
        stage.alpha = np.where(interior, stage.opacity, stage.alpha)


def _behind_envelope(stage: Stage, stages: Sequence[Stage], position: int) -> np.ndarray:
    """How far behind the occluders this layer is allowed to continue."""
    # Continue the layer a little way behind its occluders -- far enough that
    # nothing shows through at the seam, not so far that it invents whole
    # regions it was never responsible for.
    return dilate(stage.paint_mask, 28)


def _mix_clusters(pixels: np.ndarray, limit: int) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Group a layer's colours into at most `limit` families.

    Returns (lab centre, mean sRGB, share of the layer) per surviving family.
    Families closer than `MIX_MERGE_DELTA_E` are the same mix in practice, and
    families covering less than `MIN_MIX_FRACTION` are not worth remixing for.
    """
    sample = pixels[:: max(1, pixels.shape[0] // 20000)]
    lab = rgb_to_lab(sample)
    clusters = max(1, min(limit, sample.shape[0]))
    if clusters == 1:
        return [(lab.mean(axis=0), sample.mean(axis=0), 1.0)]

    assignment = kmeans(lab.astype(np.float32), clusters, KMEANS_ITERATIONS)
    families = []
    for index in range(clusters):
        members = assignment == index
        count = int(np.count_nonzero(members))
        if not count:
            continue
        families.append(
            (lab[members].mean(axis=0), sample[members].mean(axis=0), count / sample.shape[0])
        )
    families.sort(key=lambda family: -family[2])

    kept: list[tuple[np.ndarray, np.ndarray, float]] = []
    for centre, mean, share in families:
        if kept and share < MIN_MIX_FRACTION:
            continue
        if any(delta_e76(centre, other[0]) < MIX_MERGE_DELTA_E for other in kept):
            continue
        kept.append((centre, mean, share))
    return kept or families[:1]


def quantize_to_mixes(
    field: np.ndarray, region: np.ndarray, limit: int
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Reduce a continuous colour field to a handful of real palette mixes.

    A photograph's region holds tens of thousands of distinct values. A pass of
    a brush holds one to three. Collapsing the field onto mixes of actual tubes
    is what stops a step from being a copy of the reference; weighting them by
    distance is the wet-on-wet blend between those mixes on the canvas.
    """
    out = np.zeros_like(field)
    pixels = field[region]
    if pixels.size == 0:
        return out, []

    families = _mix_clusters(np.clip(pixels, 0.0, 255.0), limit)
    recipes = [nearest_mix(mean) for _, mean, _ in families]
    palette = np.array([colour for _, colour in recipes], dtype=np.float64)
    centres = np.stack([centre for centre, _, _ in families], axis=0)

    lab = rgb_to_lab(np.clip(field, 0.0, 255.0))
    distances = np.stack(
        [np.sum((lab - centre) ** 2, axis=-1) for centre in centres], axis=-1
    )
    if centres.shape[0] == 1:
        out[region] = palette[0]
    else:
        weights = np.exp(-distances / (MIX_BLEND_DELTA_E**2))
        weights /= np.maximum(weights.sum(axis=-1, keepdims=True), 1e-12)
        blended = weights @ palette
        out[region] = blended[region]

    mixes = [
        {
            "mix": recipe,
            "description": describe_mix(recipe),
            "rgb": [int(round(value)) for value in colour],
            "coverage": round(float(share), 4),
        }
        for (recipe, colour), (_, _, share) in zip(recipes, families)
    ]
    return out, mixes


def _opacity_ladder(start: float) -> list[float]:
    steps = [round(min(1.0, start + 0.1 * index), 3) for index in range(11)]
    ladder = []
    for value in steps:
        if value not in ladder:
            ladder.append(value)
        if value >= 1.0:
            break
    return ladder


def _unreachable_fraction(required: np.ndarray, region: np.ndarray) -> float:
    inside = required[region]
    if inside.size == 0:
        return 0.0
    outside = (inside < -2.0) | (inside > 257.0)
    return float(np.count_nonzero(outside) / inside.size)


def paint(stages: Sequence[Stage], shape: tuple[int, int]) -> list[np.ndarray]:
    """Lay the stack down in order, mixing each pass into what is already wet."""
    canvas = np.full((shape[0], shape[1], 3), BARE_CANVAS)
    frames = []
    for stage in stages:
        if stage.role == "base":
            required = stage.target
        else:
            # Solve for the pigment that *lands* on the target once it goes over
            # what is already on the canvas at this pass's opacity. This is the
            # whole difference between painting and revealing: the mix a step
            # calls for depends on every step before it. Solve against the
            # interior opacity, not the per-pixel alpha, or the feather band
            # divides by almost nothing and the answer explodes.
            opacity = max(stage.opacity, 1e-3)
            required = (stage.target - (1.0 - opacity) * canvas) / opacity
            for candidate in _opacity_ladder(opacity):
                trial = (stage.target - (1.0 - candidate) * canvas) / candidate
                required, opacity = trial, candidate
                if _unreachable_fraction(trial, stage.paint_mask) <= UNREACHABLE_TOLERANCE:
                    break
            if opacity != stage.opacity:
                stage.alpha = stage.alpha * (opacity / stage.opacity)
                stage.opacity = opacity

        colour, mixes = quantize_to_mixes(
            required, stage.paint_mask, ROLE_MIX_LIMIT.get(stage.role, 2)
        )
        stage.colour = colour
        stage.mixes = mixes

        alpha = stage.alpha[..., None]
        canvas = canvas * (1.0 - alpha) + colour * alpha
        frames.append(canvas.copy())
    return frames


# --------------------------------------------------------------------------
# reporting


def detail_energy(rgb: np.ndarray) -> float:
    """High-frequency content: mean distance of lightness from its local mean.

    A painting carries less of this than a photograph, and it should arrive late
    -- big shapes first, accents last. What it must not do is never arrive. This
    is the number that catches a decomposition quietly blurring the scene away.
    """
    lightness = rgb_to_lab(np.clip(rgb, 0.0, 255.0))[..., 0]
    return float(np.abs(lightness - box_mean(lightness, DETAIL_RADIUS)).mean())


def principal_axis(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if ys.size < 8:
        return 0.0
    x = xs - xs.mean()
    y = ys - ys.mean()
    covariance = np.array(
        [[float((x * x).mean()), float((x * y).mean())], [float((x * y).mean()), float((y * y).mean())]]
    )
    values, vectors = np.linalg.eigh(covariance)
    major = vectors[:, int(np.argmax(values))]
    return float((math.degrees(math.atan2(major[1], major[0])) + 180.0) % 180.0)


def representative_values(rgb: np.ndarray, mask: np.ndarray, count: int = 3) -> list[list[int]]:
    pixels = rgb[mask].astype(np.float64)
    if pixels.shape[0] < count:
        return [[int(round(value)) for value in pixels.mean(axis=0)]] if pixels.size else []
    sample = pixels[:: max(1, pixels.shape[0] // 20000)]
    assignment = kmeans(sample.astype(np.float32), count, 16)
    out = []
    for index in range(count):
        members = sample[assignment == index]
        if members.size:
            out.append([int(round(value)) for value in members.mean(axis=0)])
    return sorted(out, key=lambda colour: sum(colour))


def save_rgba(stage: Stage, path: Path) -> None:
    rgba = np.zeros((*stage.alpha.shape, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(stage.colour, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.clip(stage.alpha * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(path, format="PNG")


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(path, format="PNG")


def save_frame(frame: np.ndarray, path: Path) -> None:
    Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8), mode="RGB").save(path, format="PNG")


def contact_sheet(tiles: list[tuple[str, Image.Image]], path: Path, columns: int = 4) -> None:
    tile_width, tile_height, label_height = 360, 260, 24
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "#1b1b1b")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(tiles):
        thumbnail = image.convert("RGB").copy()
        thumbnail.thumbnail((tile_width - 8, tile_height - label_height - 8), Image.Resampling.LANCZOS)
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        sheet.paste(thumbnail, (x + 4, y + label_height))
        draw.text((x + 8, y + 6), label, fill="white")
    sheet.save(path, format="PNG")


# --------------------------------------------------------------------------


def decompose(source: Path, output_dir: Path) -> dict[str, Any]:
    with Image.open(source) as handle:
        image = handle.convert("RGB")
        scale = MAX_EDGE / max(image.size)
        if scale < 1.0:
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)),
                Image.Resampling.LANCZOS,
            )
        rgb = np.asarray(image).astype(np.float64)

    height, width = rgb.shape[:2]
    canvas_area = height * width
    lab = rgb_to_lab(rgb)
    horizon = estimate_horizon(lab[..., 0])

    ys, xs = np.mgrid[0:height, 0:width]
    features = np.stack(
        [
            lab[..., 0] * 1.0,
            lab[..., 1] * 1.1,
            lab[..., 2] * 1.1,
            (ys / height) * 46.0,
            (xs / width) * 10.0,
        ],
        axis=-1,
    ).reshape(-1, 5).astype(np.float32)

    # Cluster the two sides of the waterline independently. Colour alone cannot
    # tell the orange sky band from its own orange reflection, and no painter
    # ever lays those down in the same pass.
    above = (ys < horizon).reshape(-1)
    half = CLUSTER_COUNT // 2
    flat = np.zeros(features.shape[0], dtype=np.int32)
    flat[above] = kmeans(features[above], half, KMEANS_ITERATIONS)
    flat[~above] = kmeans(features[~above], CLUSTER_COUNT - half, KMEANS_ITERATIONS) + half
    assignment = flat.reshape(height, width)
    labels = majority_smooth(assignment, CLUSTER_COUNT, SMOOTHING_RADIUS, SMOOTHING_PASSES)

    stages = build_stages(rgb, lab, labels, CLUSTER_COUNT, horizon)
    resolve_occlusion(stages)
    extend_layers(stages, rgb)
    resolve_occlusion(stages)
    frames = paint(stages, (height, width))

    if output_dir.exists():
        shutil.rmtree(output_dir)
    for directory in ("layers", "masks", "steps"):
        (output_dir / directory).mkdir(parents=True)

    final = frames[-1]
    error = delta_e76(rgb_to_lab(final), lab)

    # Per-step convergence of the *whole* canvas. A reveal shows one region
    # snapping to the reference and everything else flat; a painting shows the
    # whole canvas walking down. A step that makes the canvas worse is a step in
    # the wrong place in the order, and this is the number that says so.
    bare = np.full_like(rgb, BARE_CANVAS)
    canvas_errors = [float(delta_e76(rgb_to_lab(bare), lab).mean())] + [
        float(delta_e76(rgb_to_lab(frame), lab).mean()) for frame in frames
    ]

    reference_detail = detail_energy(rgb)
    frame_detail = [detail_energy(frame) for frame in frames]

    steps: list[dict[str, Any]] = []
    tiles: list[tuple[str, Image.Image]] = [("reference", Image.fromarray(rgb.astype(np.uint8)))]
    previous_visible = np.zeros((height, width), dtype=bool)

    for stage, frame in zip(stages, frames):
        slug = f"{stage.index:02d}_{stage.name.lower().replace(' ', '-')}"
        layer_path = output_dir / "layers" / f"{slug}.png"
        mask_path = output_dir / "masks" / f"{slug}.png"
        step_path = output_dir / "steps" / f"{slug}.png"
        save_rgba(stage, layer_path)
        save_mask(stage.visible_mask if stage.role != "base" else stage.paint_mask, mask_path)
        save_frame(frame, step_path)

        visible_area = int(np.count_nonzero(stage.visible_mask))
        painted_area = int(np.count_nonzero(stage.paint_mask))
        new_area = int(np.count_nonzero(stage.visible_mask & ~previous_visible))
        previous_visible |= stage.visible_mask
        areas = component_areas(stage.visible_mask) if stage.role != "base" else [canvas_area]
        paintable = [area for area in areas if area >= canvas_area * 0.001]
        region = stage.visible_mask if visible_area else stage.paint_mask

        steps.append(
            {
                "index": stage.index,
                "name": stage.name,
                "role": stage.role,
                "mask_path": f"masks/{slug}.png",
                "layer_path": f"layers/{slug}.png",
                "step_path": f"steps/{slug}.png",
                "depth_score": round(stage.depth, 4),
                "visible_coverage": round(visible_area / canvas_area, 5),
                "painted_coverage": round(painted_area / canvas_area, 5),
                "new_coverage": round(new_area / canvas_area, 5),
                "component_count": len(areas),
                "paintable_component_count": len(paintable),
                "opacity": stage.opacity,
                "edge_softness_px": stage.feather,
                "mix": stage.mixes[0]["mix"] if stage.mixes else [],
                "mix_description": stage.mixes[0]["description"] if stage.mixes else "",
                "mixes": stage.mixes,
                "mix_count": len(stage.mixes),
                "canvas_delta_e": round(canvas_errors[stage.index], 3),
                "detail": round(frame_detail[stage.index - 1], 3),
                "improvement_delta_e": round(
                    canvas_errors[stage.index - 1] - canvas_errors[stage.index], 3
                ),
                "stroke_dir_deg": round(principal_axis(region), 1),
                "target_rgb": [
                    int(round(value)) for value in rgb[region].mean(axis=0)
                ]
                if region.any()
                else [0, 0, 0],
                "value_range_rgb": representative_values(rgb, region),
            }
        )
        tiles.append((f"{stage.index:02d} {stage.name}", Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8))))

    contact_sheet(tiles, output_dir / "steps-contact-sheet.png")
    contact_sheet(
        [("reference", Image.fromarray(rgb.astype(np.uint8)))]
        + [
            (
                f"{stage.index:02d} {stage.name}",
                Image.alpha_composite(
                    Image.new("RGBA", (width, height), (27, 27, 27, 255)),
                    Image.open(output_dir / "layers" / f"{stage.index:02d}_{stage.name.lower().replace(' ', '-')}.png"),
                ),
            )
            for stage in stages
        ],
        output_dir / "layers-contact-sheet.png",
    )
    save_frame(final, output_dir / "reconstruction.png")
    save_frame(
        np.clip(np.stack([error * 6.0] * 3, axis=-1), 0, 255), output_dir / "error-map.png"
    )

    painted_union = np.zeros((height, width), dtype=bool)
    for stage in stages:
        painted_union |= stage.alpha > 0.5

    pigments = sorted(
        {entry["pigment"] for step in steps for entry in step["mix"]}
    )
    max_mixes = max((step["mix_count"] for step in steps), default=0)
    regressing = [
        step["index"] for step in steps if step["improvement_delta_e"] < 0.0
    ]
    # A sound depth order finishes with a material and moves on. Coming back to
    # a role after painting something else over it means the ordering could not
    # decide what is in front -- which is exactly how a swamp fails.
    reentries = []
    for position, step in enumerate(steps):
        earlier = {other["role"] for other in steps[:position]}
        if position and step["role"] in earlier and step["role"] != steps[position - 1]["role"]:
            reentries.append(step["index"])

    report = {
        "version": 3,
        "model": "wet-on-wet back-to-front layer stack, palette-constrained",
        "source": str(source),
        "width": width,
        "height": height,
        "horizon_row": horizon,
        "stage_count": len(stages),
        "reconstruction": {
            "mean_delta_e": round(float(error.mean()), 3),
            "p95_delta_e": round(float(np.percentile(error, 95)), 3),
            "p99_delta_e": round(float(np.percentile(error, 99)), 3),
            "bare_canvas_fraction": round(
                float(np.count_nonzero(~painted_union) / canvas_area), 5
            ),
            "bare_canvas_delta_e": round(canvas_errors[0], 3),
        },
        "detail": {
            "reference": round(reference_detail, 3),
            "reconstruction": round(frame_detail[-1], 3),
            "retained": round(frame_detail[-1] / max(reference_detail, 1e-9), 3),
        },
        # Reproduction is necessary but not sufficient: a masked copy of the
        # reference scores perfectly and is not a painting. These are the two
        # metrics that tell the difference.
        "paintability": {
            "max_mixes_per_step": max_mixes,
            "mix_limit": MAX_MIXES_PER_STEP,
            "within_mix_limit": max_mixes <= MAX_MIXES_PER_STEP,
            "pigments_used": pigments,
            "pigment_count": len(pigments),
            "regressing_steps": regressing,
            "monotonic_improvement": not regressing,
            "role_order": [step["role"] for step in steps],
            "role_reentries": reentries,
            "depth_order_stable": len(reentries) <= MAX_ROLE_REENTRIES,
        },
        "steps": steps,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)

    report = decompose(arguments.source.resolve(), arguments.output_dir.resolve())
    reconstruction = report["reconstruction"]
    print(f"stages: {report['stage_count']}  horizon row: {report['horizon_row']}")
    print(
        f"reconstruction  mean dE {reconstruction['mean_delta_e']}  "
        f"p95 {reconstruction['p95_delta_e']}  p99 {reconstruction['p99_delta_e']}  "
        f"bare canvas {reconstruction['bare_canvas_fraction'] * 100:.2f}%"
    )
    print()
    paintability = report["paintability"]
    detail = report["detail"]
    print(
        f"detail  reference {detail['reference']}  reconstruction "
        f"{detail['reconstruction']}  retained {detail['retained'] * 100:.0f}%"
    )
    print(
        f"paintability  max mixes/step {paintability['max_mixes_per_step']}"
        f" (limit {paintability['mix_limit']})  "
        f"pigments {paintability['pigment_count']}  "
        f"regressing steps {paintability['regressing_steps'] or 'none'}  "
        f"role re-entries {len(paintability['role_reentries'])}"
        f"{'' if paintability['depth_order_stable'] else '  ORDER UNSTABLE'}"
    )
    print()
    header = (
        f"{'#':>2}  {'stage':<24}{'role':<9}{'vis%':>6}{'op':>6}{'mix':>4}"
        f"{'dE':>7}{'ddE':>7}  mix"
    )
    print(header)
    print("-" * len(header))
    for step in report["steps"]:
        print(
            f"{step['index']:>2}  {step['name']:<24}{step['role']:<9}"
            f"{step['visible_coverage'] * 100:>5.1f} {step['opacity']:>5.2f} "
            f"{step['mix_count']:>3} {step['canvas_delta_e']:>6.2f} "
            f"{step['improvement_delta_e']:>+6.2f}  {step['mix_description']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
