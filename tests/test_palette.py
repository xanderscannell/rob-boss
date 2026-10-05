from __future__ import annotations

import unittest

import numpy as np

from layers.stack import Stage, delta_e76, paint, quantize_to_mixes, rgb_to_lab
from layers.palette import PIGMENT_NAMES, describe_mix, mix_rgb, nearest_mix


def _delta_e(first, second) -> float:
    return float(
        delta_e76(
            rgb_to_lab(np.array(first, dtype=np.float64)),
            rgb_to_lab(np.array(second, dtype=np.float64)),
        )
    )


class MixingTest(unittest.TestCase):
    def test_single_pigment_mixes_to_itself(self) -> None:
        for index, name in enumerate(PIGMENT_NAMES):
            mix, colour = nearest_mix(mix_rgb([(index, 1)]))
            self.assertEqual([entry["pigment"] for entry in mix], [name])
            self.assertLess(_delta_e(colour, mix_rgb([(index, 1)])), 1.0)

    def test_blue_and_yellow_make_green_not_grey(self) -> None:
        """Subtractive mixing is the whole reason this is not a linear average."""
        blue = PIGMENT_NAMES.index("phthalo blue")
        yellow = PIGMENT_NAMES.index("cadmium yellow")
        mixed = mix_rgb([(blue, 1), (yellow, 1)])
        averaged = (np.array(mix_rgb([(blue, 1)])) + np.array(mix_rgb([(yellow, 1)]))) / 2.0
        self.assertGreater(mixed[1], mixed[0])
        self.assertGreater(mixed[1], mixed[2])
        self.assertGreater(mixed[1] - mixed[0], averaged[1] - averaged[0])

    def test_white_tints_strongly(self) -> None:
        white = PIGMENT_NAMES.index("titanium white")
        blue = PIGMENT_NAMES.index("phthalo blue")
        self.assertGreater(mix_rgb([(white, 8), (blue, 1)])[2], mix_rgb([(white, 1), (blue, 1)])[2])

    def test_nearest_mix_stays_close(self) -> None:
        for target in [(94, 142, 183), (200, 210, 230), (30, 40, 25), (120, 120, 120)]:
            _, colour = nearest_mix(target)
            self.assertLess(_delta_e(target, colour), 8.0, msg=str(target))

    def test_unmeasurable_ratios_are_described_as_a_touch(self) -> None:
        self.assertEqual(
            describe_mix([{"pigment": "titanium white", "parts": 1}]), "titanium white"
        )
        self.assertIn(
            "a speck of",
            describe_mix(
                [
                    {"pigment": "titanium white", "parts": 384},
                    {"pigment": "phthalo blue", "parts": 1},
                ]
            ),
        )
        self.assertIn(
            "parts",
            describe_mix(
                [
                    {"pigment": "phthalo blue", "parts": 3},
                    {"pigment": "titanium white", "parts": 1},
                ]
            ),
        )


class QuantizeTest(unittest.TestCase):
    def test_layer_is_reduced_to_at_most_the_limit(self) -> None:
        field = np.random.default_rng(0).uniform(0, 255, size=(40, 40, 3))
        region = np.ones((40, 40), dtype=bool)
        _, mixes = quantize_to_mixes(field, region, 3)
        self.assertLessEqual(len(mixes), 3)
        self.assertGreaterEqual(len(mixes), 1)

    def test_nothing_outside_the_chosen_mixes_is_laid_down(self) -> None:
        """A pass may blend its mixes together; it may not invent a new colour."""
        field = np.full((20, 20, 3), 90.0)
        field[:, 10:] = 210.0
        region = np.ones((20, 20), dtype=bool)
        colour, mixes = quantize_to_mixes(field, region, 2)
        palette = np.array(
            [
                mix_rgb([(PIGMENT_NAMES.index(e["pigment"]), e["parts"]) for e in entry["mix"]])
                for entry in mixes
            ]
        )
        laid = colour.reshape(-1, 3)
        self.assertTrue(np.all(laid >= palette.min(axis=0) - 1e-6))
        self.assertTrue(np.all(laid <= palette.max(axis=0) + 1e-6))

    def test_outside_the_region_nothing_is_painted(self) -> None:
        field = np.full((16, 16, 3), 120.0)
        region = np.zeros((16, 16), dtype=bool)
        region[4:12, 4:12] = True
        colour, _ = quantize_to_mixes(field, region, 2)
        self.assertTrue(np.all(colour[~region] == 0.0))


class SequentialPaintTest(unittest.TestCase):
    """The point of the rewrite: a pass depends on what is already on the canvas."""

    @staticmethod
    def _stack(ground: float) -> list[Stage]:
        shape = (24, 24)
        full = np.ones(shape, dtype=bool)
        base = Stage(
            index=1,
            name="Ground wash",
            role="base",
            paint_mask=full,
            visible_mask=np.zeros(shape, dtype=bool),
            feather=0,
            opacity=0.55,
            depth=1.0,
            target=np.full((*shape, 3), ground),
            alpha=np.full(shape, 0.55),
        )
        over = Stage(
            index=2,
            name="Sky",
            role="sky",
            paint_mask=full,
            visible_mask=full,
            feather=0,
            opacity=0.8,
            depth=0.5,
            target=np.full((*shape, 3), 150.0),
            alpha=np.full(shape, 0.8),
        )
        return [base, over]

    def test_the_same_target_over_a_different_ground_calls_for_a_different_mix(self) -> None:
        dark = self._stack(40.0)
        light = self._stack(220.0)
        paint(dark, (24, 24))
        paint(light, (24, 24))
        self.assertNotEqual(dark[1].mixes[0]["mix"], light[1].mixes[0]["mix"])

    def test_a_thin_pass_thickens_when_it_cannot_reach(self) -> None:
        stages = self._stack(240.0)
        stages[1].target = np.zeros((24, 24, 3))
        paint(stages, (24, 24))
        self.assertGreater(stages[1].opacity, 0.8)


if __name__ == "__main__":
    unittest.main()
