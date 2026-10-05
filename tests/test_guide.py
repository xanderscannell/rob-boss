from __future__ import annotations

import unittest

import cv2
import numpy as np

from app import overlay


class MaskedStepTest(unittest.TestCase):
    def test_step_is_compiled_frame_masked_by_its_own_layer(self):
        """canvas = canvas * (1 - alpha) + colour * alpha builds the main image; each step
        then lights only its own layer's area of it, with opacity divided out of the mask."""
        import tempfile
        from pathlib import Path
        from app.scene import compiled_frames, masked_step
        with tempfile.TemporaryDirectory() as d:
            under = np.zeros((10, 10, 4), np.uint8)
            under[...] = (255, 0, 0, 255)                                 # layer 1: opaque blue (BGRA)
            over = np.zeros((10, 10, 4), np.uint8)
            over[...] = (0, 0, 255, 0)                                    # layer 2: red at opacity 0.5 ...
            over[:, :5, 3] = 128                                          # ... on the left half only
            paths = [Path(d) / "1.png", Path(d) / "2.png"]
            cv2.imwrite(str(paths[0]), under)
            cv2.imwrite(str(paths[1]), over)
            frames = compiled_frames(paths, bare=245)
            step2 = masked_step(frames[1], paths[1], opacity=0.5)
        a = 128 / 255
        # inside layer 2: the compiled colour (red over blue at 50%), at full brightness
        np.testing.assert_allclose(step2[0, 0], np.round([255 * (1 - a), 0, 255 * a]), atol=1)
        self.assertEqual(step2[0, 9].sum(), 0)                            # outside layer 2: dark


class OverlayTest(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((400, 600), np.uint8)
        self.mask[100:300, 100:400] = 255

    def test_fill_only_inside_region(self):
        img = overlay.render(self.mask, {"fill_rgb": (0, 0, 255), "fill_alpha": 1.0, "outline": False})
        self.assertEqual(tuple(img[200, 250]), (255, 0, 0))            # BGR blue inside
        self.assertEqual(img[50, 50].sum(), 0)                          # black outside

    def test_arrows_stay_inside_region_and_point_along_direction(self):
        img = overlay.render(self.mask, {"fill": False, "outline": False, "arrows": True,
                                         "stroke_dir_deg": 0, "arrow_spacing_px": 80, "arrow_px": 2})
        lit = img.sum(2) > 0
        self.assertTrue(lit.any())
        self.assertFalse(lit[~(cv2.dilate(self.mask, np.ones((7, 7), np.uint8)) > 0)].any())
        ys, xs = np.nonzero(lit)
        self.assertGreater(np.ptp(xs), np.ptp(ys))                      # horizontal strokes

    def test_fill_image_copies_layer_colours_inside_region(self):
        layer = np.zeros((400, 600, 4), np.uint8)
        layer[:, :300] = (255, 0, 0, 255)                              # BGRA: blue left half
        layer[:, 300:] = (0, 0, 255, 128)                              # red right half, alpha ignored
        img = overlay.render(self.mask, {"fill_image": layer, "fill_alpha": 1.0, "outline": False})
        self.assertEqual(tuple(img[200, 150]), (255, 0, 0))
        self.assertEqual(tuple(img[200, 350]), (0, 0, 255))
        self.assertEqual(img[50, 50].sum(), 0)                          # outside the mask stays dark

    def test_outline_mask_outlines_visible_part_while_fill_covers_whole_layer(self):
        visible = np.zeros_like(self.mask)
        visible[100:300, 250:400] = 255                                 # right part of the filled region
        img = overlay.render(self.mask, {"fill_rgb": (0, 0, 255), "fill_alpha": 1.0,
                                         "outline_rgb": (0, 255, 0), "outline_px": 3,
                                         "outline_mask": visible})
        self.assertEqual(tuple(img[200, 150]), (255, 0, 0))            # buried part is still filled
        self.assertEqual(tuple(img[200, 250]), (0, 255, 0))            # outline runs along the visible part
        self.assertNotEqual(tuple(img[200, 100]), (0, 255, 0))         # not along the filled region's edge

    def test_render_resizes_mask(self):
        img = overlay.render(self.mask, size=(300, 200))
        self.assertEqual(img.shape, (200, 300, 3))


if __name__ == "__main__":
    unittest.main()
