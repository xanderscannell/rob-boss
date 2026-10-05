from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from lesson import cv as cvmod
from lesson import mixfix
from lesson.machine import StepMachine
from lesson.schema import Step, Verdict


STACK_BARE = 245     # layers.stack' BARE_CANVAS: the picture before a stack's first step


@dataclass
class WatchConfig:
    settle_s: float = 2.0          # the step's region must be still this long before any check
    motion_t: float = 2.0          # mean abs gray diff (0-255) in the region above which the painter is "active"
    region_margin: float = 0.08    # region grown by this fraction of canvas width for motion
    change_t: float = 1.5          # diff vs the last checked frame that counts as "changed"
    idle_coverage_s: float = 6.0   # pause needed before "unpainted area" is a mistake, not progress
    confirm_gap_s: float = 2.0     # CV corrections must hold across two measurements this far apart
    complete_cov: float = 0.90     # coverage at which the step is done
    progress_strike_pts: float = 0.05  # a coverage nudge is no strike if coverage rose this much since the last
    min_cov_for_value: float = 0.25
    value_dl: float = 12.0         # |delta L*| beyond this is "too light/dark"
    paint_de: float = 12.0         # colour distance from bare canvas that counts as painted


@dataclass
class Event:
    kind: str                      # correction | advanced | complete
    step_index: int
    verdict: Verdict | None = None
    source: str = "cv"
    missing: np.ndarray | None = None  # mask-space bool map of still-bare areas
    off_value: np.ndarray | None = None  # mask-space bool map of painted areas that are too light/dark
    now: float = 0.0


def _small_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (96, 64), interpolation=cv2.INTER_AREA).astype(np.float32)


class Watcher:
    def __init__(self, machine: StepMachine, ref_rgb: np.ndarray, scene_dir: Path,
                 capture: Callable[[], np.ndarray], config: WatchConfig | None = None,
                 bare_rgb=cvmod.DEFAULT_BARE_RGB):
        self.machine, self.ref, self.scene_dir = machine, ref_rgb, Path(scene_dir)
        self.capture, self.cfg, self.bare_rgb = capture, config or WatchConfig(), bare_rgb
        self.bare_image: np.ndarray | None = None
        self._masks: dict[int, tuple] = {}
        self._frames: dict[str, tuple[Path | None, Path]] | None = None
        self._grown: dict[int, np.ndarray] = {}
        self._reset_step()
        self._prev: np.ndarray | None = None

    def calibrate(self, empty_canvas_bgr: np.ndarray) -> None:
        """Learn the bare canvas from the empty canvas: its colour, and per pixel how it looks."""
        rgb = cv2.cvtColor(empty_canvas_bgr, cv2.COLOR_BGR2RGB)
        self.bare_rgb = cvmod.estimate_bare_rgb(rgb)
        self.bare_image = rgb

    def _reset_step(self) -> None:
        self._still_since: float | None = None
        self._checked: np.ndarray | None = None   # peek frame at the last check
        self._recheck_at: float | None = None
        self._streak: tuple[str, float] | None = None  # (category, time first seen)
        self._nudged_cov = 0.0                         # coverage at the last coverage nudge (step starts bare)
        self.coverage: float | None = None             # latest measured coverage of this step, 0..1

    def rebase(self, canvas_bgr: np.ndarray) -> None:
        """A step is finished: the canvas as it is now is the starting point for the next one."""
        self.bare_image = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)

    def _stack_frames(self) -> dict[str, tuple[Path | None, Path]]:
        """Layer stack: mask stem -> (expected picture before the step, after it)."""
        if self._frames is None:
            self._frames, before = {}, None
            report = self.scene_dir / "report.json"
            for s in json.loads(report.read_text())["steps"] if report.exists() else []:
                if s.get("step_path"):
                    after = self.scene_dir / s["step_path"]
                    self._frames[Path(s["mask_path"]).stem] = (before, after)
                    before = after
        return self._frames

    def _target(self, step: Step) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """(reference, expected-before, mask) to measure the step against.

        Layer stack: the step's frame is the reference and the mask is where it differs from
        the previous frame (what this step changes, under later layers too, so it can be
        checked as it is painted). Otherwise: the final reference and the step's own mask."""
        if step.index not in self._masks:
            frames = self._stack_frames().get(Path(step.mask_path).stem)
            if frames:
                after = np.asarray(Image.open(frames[1]).convert("RGB"))
                before = np.asarray(Image.open(frames[0]).convert("RGB")) if frames[0] else \
                    np.full_like(after, STACK_BARE)
                change = np.linalg.norm(cvmod.to_lab(after) - cvmod.to_lab(before), axis=2) > self.cfg.paint_de
                self._masks[step.index] = (after, before, np.where(change, 255, 0).astype(np.uint8))
            else:
                mask = np.asarray(Image.open(self.scene_dir / step.mask_path).convert("L"))
                self._masks[step.index] = (self.ref, None, mask)
        return self._masks[step.index]

    def expected_tones(self, step: Step, n: int = 3) -> list[list[int]]:
        """The colours the finished step shows where it is judged, darkest to lightest (RGB): the
        mean of each lightness n-tile. Swatches for a digital canvas, so painting with them passes."""
        target, _, mask = self._target(step)
        if target.shape[:2] != mask.shape:
            target = cv2.resize(target, mask.shape[::-1], interpolation=cv2.INTER_AREA)
        px = target[mask > 127].reshape(-1, 3).astype(np.float32)
        order = np.argsort(px @ np.float32([0.299, 0.587, 0.114]))
        return [px[c].mean(axis=0).round().astype(int).tolist() for c in np.array_split(order, n) if len(c)]

    def _mask(self, step: Step) -> np.ndarray:
        return self._target(step)[2]

    def _grown_region(self, step: Step) -> np.ndarray:
        """The step's region grown by region_margin (bool, mask size): where hands and motion count."""
        if step.index not in self._grown:
            mask = self._mask(step) > 127
            r = max(1, int(round(self.cfg.region_margin * mask.shape[1])))
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            self._grown[step.index] = cv2.dilate(mask.astype(np.uint8), k).astype(bool)
        return self._grown[step.index]

    def _region_diff(self, small: np.ndarray, other: np.ndarray, step: Step) -> float:
        """Mean change between two small frames inside the grown region (whole frame if it is empty)."""
        region = cv2.resize(self._grown_region(step).astype(np.uint8), small.shape[::-1],
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        diff = np.abs(small - other)
        return float(diff[region].mean()) if region.any() else float(diff.mean())

    def tick(self, peek_bgr: np.ndarray, now: float) -> Event | None:
        cfg, step = self.cfg, self.machine.current
        small = _small_gray(peek_bgr)
        prev, self._prev = self._prev, small
        if step is None or self.machine.status != "active" or prev is None:
            return None

        if self._region_diff(small, prev, step) > cfg.motion_t:         # painter active in the region -> debounce
            self._still_since = None
            return None
        if self._still_since is None:
            self._still_since = now
        if now - self._still_since < cfg.settle_s:
            return None

        changed = self._checked is None or self._region_diff(small, self._checked, step) > cfg.change_t
        if not changed and (self._recheck_at is None or now < self._recheck_at):
            return None                                             # nothing new to look at
        self._checked, self._recheck_at = small, None

        target, before, mask = self._target(step)
        canvas = cv2.cvtColor(self.capture(), cv2.COLOR_BGR2RGB)
        m = cvmod.measure(canvas, target, mask, bare_rgb=self.bare_rgb, bare_image=self.bare_image,
                          before_rgb=before, paint_de=cfg.paint_de, value_dl=cfg.value_dl)
        self.coverage = m.coverage if m.checkable else None

        verdict = None
        if m.checkable and m.delta_l is not None and m.coverage >= cfg.min_cov_for_value \
                and abs(m.delta_l) > cfg.value_dl:
            verdict = self._value_verdict(step, m)
        elif m.checkable and m.coverage < cfg.complete_cov:
            if now - self._still_since >= cfg.idle_coverage_s:
                verdict = Verdict(verdict="ADJUST", category="coverage",
                                  adjustment=f"There's still a little bare canvas waiting for some love - about "
                                             f"{(1 - m.coverage) * 100:.0f}% of this area. Let's go "
                                             "back in and fill it right in.")
            else:                                                   # still working: progress, not a mistake
                self._recheck_at = self._still_since + cfg.idle_coverage_s
                self._streak = None
                return None
        else:                                                       # covered and on value: done
            self.rebase(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            return self._advance(step, Verdict(verdict="READY", category="none", adjustment=""), now)

        if self._streak is None or self._streak[0] != verdict.category:   # debounce corrections over time
            self._streak = (verdict.category, now)
            self._recheck_at = now + cfg.confirm_gap_s
            return None
        if now - self._streak[1] < cfg.confirm_gap_s:
            self._recheck_at = self._streak[1] + cfg.confirm_gap_s
            return None
        self._streak = None
        strike = True
        if verdict.category == "coverage":         # filling in steadily is progress, not a failed try
            strike = m.coverage - self._nudged_cov < cfg.progress_strike_pts
            self._nudged_cov = m.coverage
        return self._correct(step, verdict, now, strike,
                             missing=m.missing if verdict.category == "coverage" else None,
                             off_value=m.value_off if verdict.category == "value" else None)

    @staticmethod
    def _value_verdict(step: Step, m: cvmod.Measurement) -> Verdict:
        return Verdict(verdict="ADJUST", category="value",
                       adjustment=mixfix.advise(step, m.mean_l_canvas, m.mean_l_ref))

    def _correct(self, step, verdict, now, strike=True, missing=None, off_value=None) -> Event:
        self.machine.submit(verdict, strike=strike)
        return Event("correction", step.index, verdict, "cv", missing, off_value, now)

    def _advance(self, step, verdict, now) -> Event:
        status = self.machine.submit(verdict)
        self._reset_step()
        return Event("complete" if status == "complete" else "advanced", step.index, verdict, now=now)
