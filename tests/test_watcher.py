from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from lesson import cv as cvmod
from lesson import mixfix
from lesson.cv import DEFAULT_BARE_RGB, measure
from lesson.machine import StepMachine
from lesson.schema import Step
from lesson.simulate import SimFeed, run
from lesson.watcher import Watcher, _small_gray

W, H = 120, 80


def make_ref() -> np.ndarray:
    ref = np.zeros((H, W, 3), np.uint8)
    ref[:H // 2] = (70, 120, 200)      # blue sky, top half
    ref[H // 2:] = (190, 110, 60)      # orange ground, bottom half
    return ref


def make_step(i: int, name: str) -> Step:
    return Step(index=i, name=name, mask_path=f"layers/{i}.png", target_rgb=(1, 2, 3),
                mix=[{"pigment": "titanium white", "parts": 1}], brush="1in flat",
                technique="t", stroke_dir_deg=0, success="ok")


def paint(ref: np.ndarray, mask: np.ndarray, rows_frac: float = 1.0, scale: float = 1.0,
          base: np.ndarray | None = None) -> np.ndarray:
    """Canvas with `mask` painted from the top down to rows_frac, scaled in brightness."""
    out = np.full_like(ref, DEFAULT_BARE_RGB) if base is None else base.copy()
    ys = np.mgrid[0:H, 0:W][0]
    region = (mask > 127) & (ys < H * rows_frac) if rows_frac < 1 else mask > 127
    out[region] = np.clip(ref[region].astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return out


class MeasureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ref = make_ref()
        self.mask = np.zeros((H, W), np.uint8)
        self.mask[:H // 2] = 255

    def test_complete_correct(self) -> None:
        m = measure(paint(self.ref, self.mask), self.ref, self.mask)
        self.assertTrue(m.checkable)
        self.assertGreater(m.coverage, 0.99)
        self.assertLess(abs(m.delta_l), 1.0)

    def test_half_coverage(self) -> None:
        m = measure(paint(self.ref, self.mask, rows_frac=0.25), self.ref, self.mask)
        self.assertLess(m.coverage, 0.65)
        self.assertTrue(m.missing.any())

    def test_too_dark_is_negative_delta_l(self) -> None:
        m = measure(paint(self.ref, self.mask, scale=0.5), self.ref, self.mask)
        self.assertLess(m.delta_l, -12)

    def test_nothing_to_paint_is_not_checkable(self) -> None:
        blank_ref = np.full_like(self.ref, DEFAULT_BARE_RGB)
        self.assertFalse(measure(blank_ref, blank_ref, self.mask).checkable)

    def test_uneven_light_on_bare_canvas_is_not_paint_with_a_bare_capture(self) -> None:
        ref = make_ref()
        mask = np.zeros((H, W), np.uint8)
        mask[:H // 2] = 255
        falloff = np.linspace(1.0, 0.55, W)[None, :, None]          # brighter on the left, dim on the right
        bare = np.clip(np.full((H, W, 3), DEFAULT_BARE_RGB, np.float32) * falloff, 0, 255).astype(np.uint8)
        naive = measure(bare, ref, mask, bare_rgb=tuple(int(v) for v in np.median(bare.reshape(-1, 3), 0)))
        per_px = measure(bare, ref, mask, bare_image=bare)
        self.assertGreater(naive.coverage, 0.2)                      # dim paper counted as paint
        self.assertEqual(per_px.coverage, 0.0)                       # nothing painted

    def test_calibrates_bare_canvas_colour(self) -> None:
        self.assertEqual(cvmod.estimate_bare_rgb(np.full((4, 4, 3), (200, 190, 180), np.uint8)), (200, 190, 180))


class MixFixTest(unittest.TestCase):
    def test_added_parts_reach_the_reference_lightness(self) -> None:
        t, lc, lr, lp = 10, 50.0, 60.0, 96.0
        p = mixfix.parts_to_add(t, lc, lr, lp)
        self.assertAlmostEqual((t * lc + p * lp) / (t + p), lr, places=6)

    def test_capped_and_unreachable_cases_never_exceed_the_mix(self) -> None:
        self.assertEqual(mixfix.parts_to_add(10, 20.0, 80.0, 96.0), 10.0)   # would need >10 parts: capped
        self.assertEqual(mixfix.parts_to_add(10, 50.0, 60.0, 12.0), 10.0)   # black cannot lighten

    def test_part_formatting(self) -> None:
        self.assertEqual([mixfix.fmt_parts(x) for x in (0.1, 1.0, 1.5, 2.0, 0.9)],
                         ["half a part", "1 part", "1.5 parts", "2 parts", "1 part"])

    def test_advice_picks_white_to_lighten_and_the_mixs_darkest_to_darken(self) -> None:
        step = Step(**{**make_step(1, "sky").model_dump(),
                       "mix": [{"pigment": "titanium white", "parts": 4},
                               {"pigment": "phthalo blue", "parts": 2}]})
        self.assertIn("titanium white", mixfix.advise(step, 40.0, 60.0))
        self.assertIn("too dark", mixfix.advise(step, 40.0, 60.0))
        self.assertIn("phthalo blue", mixfix.advise(step, 70.0, 50.0))
        self.assertIn("6-part mix", mixfix.advise(step, 70.0, 50.0))


class ValueMapTest(unittest.TestCase):
    def test_map_points_at_the_too_dark_half_only(self) -> None:
        ref = make_ref()
        mask = np.zeros((H, W), np.uint8)
        mask[:H // 2] = 255
        canvas = paint(ref, mask)
        canvas[:H // 2, :W // 2] = (canvas[:H // 2, :W // 2] * 0.4).astype(np.uint8)  # left half too dark
        m = measure(canvas, ref, mask, blur_px=9)
        off = m.value_off
        self.assertGreater(off[:H // 2, :W // 4].mean(), 0.9)       # left: flagged
        self.assertLess(off[:H // 2, 3 * W // 4:].mean(), 0.05)      # right: fine


class WatcherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "layers").mkdir()
        self.ref = make_ref()
        self.masks = []
        for i, sl in enumerate((slice(0, H // 2), slice(H // 2, H)), 1):
            m = np.zeros((H, W), np.uint8)
            m[sl] = 255
            Image.fromarray(m).save(self.dir / "layers" / f"{i}.png")
            self.masks.append(m)
        self.steps = [make_step(1, "sky"), make_step(2, "ground")]
        self.feed = SimFeed((W, H))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def watcher(self) -> Watcher:
        return Watcher(StepMachine(self.steps), self.ref, self.dir, self.feed.capture)

    def test_hand_in_frame_never_triggers_a_check(self) -> None:
        w = self.watcher()
        events = run(w, self.feed, [(paint(self.ref, self.masks[0]), 20.0, 0.0)])
        self.assertEqual(events, [])

    def test_complete_step_advances_after_settling(self) -> None:
        w = self.watcher()
        events = run(w, self.feed, [(paint(self.ref, self.masks[0]), 2.0, 10.0)])
        self.assertEqual([e.kind for _, e in events], ["advanced"])
        self.assertEqual(w.machine.current.index, 2)
        self.assertGreaterEqual(events[0][0], 2.0 + w.cfg.settle_s)  # not before the settle time

    def test_progress_midstroke_is_quiet_then_idle_gives_coverage_correction(self) -> None:
        w = self.watcher()
        part = paint(self.ref, self.masks[0], rows_frac=0.2)
        quiet = run(w, self.feed, [(part, 1.0, 4.0)])           # paused only 4s: still working
        self.assertEqual(quiet, [])
        events = run(w, self.feed, [(part, 0.0, 12.0)], t0=5.0)  # idle long enough
        self.assertEqual([(e.kind, e.verdict.category, e.source) for _, e in events],
                         [("correction", "coverage", "cv")])
        self.assertTrue(events[0][1].missing.any())

    def test_too_dark_gives_value_correction(self) -> None:
        w = self.watcher()
        events = run(w, self.feed, [(paint(self.ref, self.masks[0], scale=0.5), 1.0, 10.0)])
        self.assertEqual([(e.kind, e.verdict.category, e.source) for _, e in events],
                         [("correction", "value", "cv")])
        ev = events[0][1]
        self.assertIn("too dark", ev.verdict.adjustment)
        self.assertIn("part", ev.verdict.adjustment)          # a concrete mix fix, not just "lighten"
        self.assertTrue(ev.off_value.any() and ev.missing is None)  # a map of where

    def test_correction_is_not_repeated_until_the_canvas_changes(self) -> None:
        w = self.watcher()
        dark = paint(self.ref, self.masks[0], scale=0.5)
        events = run(w, self.feed, [(dark, 1.0, 30.0)])
        self.assertEqual(len(events), 1)
        fixed = paint(self.ref, self.masks[0])
        events = run(w, self.feed, [(fixed, 2.0, 10.0)], t0=40.0)
        self.assertEqual([e.kind for _, e in events], ["advanced"])  # response to the change

    def test_hand_moving_outside_the_region_does_not_reset_the_settle_timer(self) -> None:
        w = self.watcher()
        done = paint(self.ref, self.masks[0])
        run(w, self.feed, [(done, 1.0, 0.5)])                    # works in the region, then leaves
        self.feed.hand_box = (0.3, 0.7, 0.9, 1.0)                # keeps moving, low on step 2's area
        events = run(w, self.feed, [(done, 6.0, 0.0)], t0=1.5)   # still moving there the whole time
        self.assertEqual([e.kind for _, e in events], ["advanced"])
        self.assertLessEqual(events[0][0], 1.5 + w.cfg.settle_s + 1.0)  # ~settle_s after leaving the region

    def test_hand_moving_inside_the_region_keeps_resetting_it(self) -> None:
        w = self.watcher()
        self.feed.hand_box = (0.3, 0.7, 0.1, 0.3)                # inside step 1's region
        events = run(w, self.feed, [(paint(self.ref, self.masks[0]), 20.0, 0.0)])
        self.assertEqual(events, [])

    def test_three_strikes_keep_correcting(self) -> None:
        w = self.watcher()
        t = 0.0
        kinds = []
        for scale in (0.5, 0.45, 0.4):  # each repaint is still too dark: a new canvas, a new check
            ev = run(w, self.feed, [(paint(self.ref, self.masks[0], scale=scale), 1.0, 12.0)], t0=t)
            kinds += [e.kind for _, e in ev]
            t += 20.0
        self.assertEqual(kinds, ["correction", "correction", "correction"])
        self.assertEqual((w.machine.status, w.machine.state["tries"]), ("active", 3))

    def test_steady_coverage_progress_is_not_a_strike(self) -> None:
        w = self.watcher()
        cats, t = [], 0.0
        for frac in (0.1, 0.2, 0.3, 0.4):                    # 20%, 40%, 60%, 80% of the sky, pausing in between
            ev = run(w, self.feed, [(paint(self.ref, self.masks[0], rows_frac=frac), 1.0, 12.0)], t0=t)
            cats += [e.verdict.category for _, e in ev]
            t += 20.0
        self.assertEqual(cats, ["coverage"] * 4)
        self.assertEqual(w.machine.state["tries"], 0)                 # every nudge followed real progress
        self.assertAlmostEqual(w.coverage, 0.8, delta=0.05)           # the live readout
        ev = run(w, self.feed, [(paint(self.ref, self.masks[0], rows_frac=0.41), 1.0, 12.0)], t0=t)
        self.assertEqual([e.verdict.category for _, e in ev], ["coverage"])
        self.assertEqual(w.machine.state["tries"], 1)                 # barely any progress: that one counts

    def test_small_progress_in_a_small_region_still_triggers_a_check(self) -> None:
        box = np.zeros((H, W), np.uint8)
        box[:20, :30] = 255                                           # about 6% of the canvas
        Image.fromarray(box).save(self.dir / "layers" / "1.png")
        self.ref = np.full_like(self.ref, DEFAULT_BARE_RGB)
        self.ref[:20, :30] = (200, 215, 235)                          # a pale blue: little brightness change
        w = self.watcher()
        half = np.full_like(self.ref, DEFAULT_BARE_RGB)
        half[:20, :15] = self.ref[:20, :15]
        full = half.copy()
        full[:20, :30] = self.ref[:20, :30]
        a, b = (_small_gray(cv2.cvtColor(c, cv2.COLOR_RGB2BGR)) for c in (half, full))
        self.assertLess(float(np.abs(a - b).mean()), w.cfg.change_t)  # a whole-frame test misses this progress
        first = run(w, self.feed, [(half, 1.0, 12.0)])
        self.assertEqual([e.verdict.category for _, e in first], ["coverage"])
        done = run(w, self.feed, [(full, 1.0, 6.0)], t0=20.0)
        self.assertEqual((done[0][1].kind, done[0][1].step_index), ("advanced", 1))

    def test_full_two_step_session_completes(self) -> None:
        w = self.watcher()
        step1 = paint(self.ref, self.masks[0])
        step2 = paint(self.ref, self.masks[1], base=step1)
        events = run(w, self.feed, [(step1, 2.0, 8.0), (step2, 2.0, 8.0)])
        self.assertEqual([e.kind for _, e in events], ["advanced", "complete"])


class LayerStackTest(unittest.TestCase):
    """A layers.stack scene: a wash over everything, then a band painted over the wash."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        for sub in ("masks", "steps"):
            (self.dir / sub).mkdir()
        wash = np.full((H, W, 3), (150, 140, 160), np.uint8)            # step 1: mauve wash, whole sheet
        band = wash.copy()
        band[:H // 2] = (60, 110, 200)                                   # step 2: blue band, top half
        self.frames = [wash, band]
        visible = [np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8)]
        visible[0][H // 2:], visible[1][:H // 2] = 255, 255
        report = {"steps": []}
        for i, (name, frame, vis) in enumerate(zip(("wash", "band"), self.frames, visible), 1):
            Image.fromarray(vis).save(self.dir / "masks" / f"0{i}_{name}.png")
            Image.fromarray(frame).save(self.dir / "steps" / f"0{i}_{name}.png")
            report["steps"].append({"mask_path": f"masks/0{i}_{name}.png", "step_path": f"steps/0{i}_{name}.png"})
        (self.dir / "report.json").write_text(json.dumps(report))
        self.steps = [make_step(1, "wash").model_copy(update={"mask_path": "masks/01_wash.png"}),
                      make_step(2, "band").model_copy(update={"mask_path": "masks/02_band.png"})]
        self.feed = SimFeed((W, H))
        self.feed.set(np.full((H, W, 3), 245, np.uint8))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def watcher(self) -> Watcher:
        w = Watcher(StepMachine(self.steps), self.frames[1], self.dir, self.feed.capture)
        w.calibrate(self.feed.capture())
        return w

    def test_later_step_is_judged_from_the_canvas_it_started_on(self) -> None:
        w = self.watcher()
        events = run(w, self.feed, [(self.frames[0], 2.0, 8.0)])          # wash done
        self.assertEqual([e.kind for _, e in events], ["advanced"])
        events = run(w, self.feed, [(self.frames[0], 2.0, 3.0)], t0=20.0)  # hand loads the brush, pauses
        self.assertEqual(events, [])        # band not painted yet: the wash under it is not the band
        half = self.frames[0].copy()
        half[:H // 4] = self.frames[1][:H // 4]
        events = run(w, self.feed, [(half, 2.0, 4.0)], t0=40.0)
        self.assertEqual(events, [])        # half the band: progress, not done
        events = run(w, self.feed, [(self.frames[1], 2.0, 8.0)], t0=50.0)
        self.assertEqual([e.kind for _, e in events], ["complete"])

    def test_too_dark_band_over_the_wash_is_a_value_correction(self) -> None:
        w = self.watcher()
        run(w, self.feed, [(self.frames[0], 2.0, 8.0)])
        dark = self.frames[0].copy()
        dark[:H // 2] = (25, 45, 90)
        events = run(w, self.feed, [(dark, 2.0, 10.0)], t0=20.0)
        self.assertEqual([(e.kind, e.verdict.category) for _, e in events], [("correction", "value")])


if __name__ == "__main__":
    unittest.main()
