from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lesson import planner
from lesson.machine import StepMachine
from lesson.schema import Step, Verdict
from layers.palette import PIGMENT_NAMES

READY = Verdict(verdict="READY", category="none", adjustment="")


def adjust(cat: str, text: str = "Do the thing.") -> Verdict:
    return Verdict(verdict="ADJUST", category=cat, adjustment=text)


def make_step(i: int = 1) -> Step:
    return Step(index=i, name="s", mask_path="layers/x.png", target_rgb=(1, 2, 3),
                mix=[{"pigment": "titanium white", "parts": 1}], brush="1in flat",
                technique="t", stroke_dir_deg=0, success="ok")


class StepMachineTest(unittest.TestCase):
    def test_ready_advances_then_completes(self) -> None:
        m = StepMachine([make_step(1), make_step(2)])
        self.assertEqual(m.submit(READY), "advanced")
        self.assertEqual(m.current.index, 2)
        self.assertEqual(m.submit(READY), "complete")
        self.assertIsNone(m.current)

    def test_struggling_after_three_strikes_never_locks_the_step(self) -> None:
        m = StepMachine([make_step(1), make_step(2)])
        self.assertEqual(m.submit(adjust("value")), "retry")
        self.assertEqual(m.submit(adjust("value")), "retry")
        self.assertEqual(m.submit(adjust("value")), "struggling")
        self.assertEqual(m.status, "active")
        self.assertEqual(m.submit(READY), "advanced")       # still judged, can still pass
        self.assertEqual((m.current.index, m.state["tries"]), (2, 0))

    def test_a_nudge_that_is_not_a_strike_does_not_count(self) -> None:
        m = StepMachine([make_step(1), make_step(2)])
        for _ in range(5):
            self.assertEqual(m.submit(adjust("coverage"), strike=False), "retry")
        self.assertEqual(m.state["tries"], 0)

    def test_skip_works_any_time_and_is_recorded(self) -> None:
        m = StepMachine([make_step(1), make_step(2)])
        self.assertEqual(m.skip(), "advanced")
        self.assertEqual(m.state["history"][-1], {"step": 1, "skipped": True})
        self.assertEqual(m.skip(), "complete")
        with self.assertRaises(RuntimeError):
            m.skip()

    def test_ready_resets_tries_and_state_round_trips(self) -> None:
        steps = [make_step(1), make_step(2)]
        m = StepMachine(steps)
        m.submit(adjust("coverage"))
        m.submit(READY)
        m2 = StepMachine.from_dict(steps, dict(m.to_dict()))
        self.assertEqual((m2.current.index, m2.state["tries"]), (2, 0))


class PlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.scene = Path(self.tmp.name)
        mix = [{"pigment": "titanium white", "parts": 8}, {"pigment": "phthalo blue", "parts": 1}]
        rows = [("01_ground-wash", "base", 0.0), ("02_sky", "sky", 178.0), ("03_hill", "near", 88.0),
                ("04_odd", "mystery", 45.0)]
        steps = [{"index": i, "name": stem[3:].title(), "role": role, "mask_path": f"masks/{stem}.png",
                  "target_rgb": [10 * i, 20, 30], "stroke_dir_deg": axis, "mix": mix}
                 for i, (stem, role, axis) in enumerate(rows, 1)]
        (self.scene / "report.json").write_text(json.dumps({"steps": steps}))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_keeps_layers_order_mix_and_role(self) -> None:
        steps = planner.plan(self.scene)
        self.assertEqual([s.index for s in steps], [1, 2, 3, 4])
        self.assertEqual([s.mask_path for s in steps],
                         ["masks/01_ground-wash.png", "masks/02_sky.png", "masks/03_hill.png", "masks/04_odd.png"])
        self.assertEqual(steps[1].target_rgb, (20, 20, 30))
        self.assertEqual(steps[1].mix_description, "8 parts titanium white to 1 part phthalo blue")
        self.assertTrue(all(m.pigment in PIGMENT_NAMES for s in steps for m in s.mix))
        self.assertEqual(steps[1].brush, planner.ROLES["sky"][0])
        self.assertIn("horizontal", steps[1].technique)          # 178 degrees is a horizontal axis
        self.assertIn("vertical", steps[2].technique)
        self.assertEqual(steps[1].stroke_dir_deg, 178)

    def test_unknown_role_gets_the_generic_template(self) -> None:
        odd = planner.plan(self.scene)[3]
        self.assertEqual((odd.brush, odd.success), (planner.GENERIC[0], planner.GENERIC[2]))
        self.assertIn("diagonal", odd.technique)

    def test_direction_words(self) -> None:
        self.assertEqual([planner.direction(d) for d in (0, 170, 90, 45, 135, 360)],
                         ["horizontal", "horizontal", "vertical", "diagonal", "diagonal", "horizontal"])


if __name__ == "__main__":
    unittest.main()
