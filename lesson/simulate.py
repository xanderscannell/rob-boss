from __future__ import annotations

import cv2
import numpy as np

from lesson.cv import DEFAULT_BARE_RGB


class SimFeed:
    def __init__(self, size: tuple[int, int] = (768, 512)):
        w, h = size
        self.canvas = np.full((h, w, 3), DEFAULT_BARE_RGB, np.uint8)
        self._n = 0
        self.hand_box = (0.2, 0.8, 0.2, 0.8)  # where the moving hand roams: x0, x1, y0, y1 fractions

    def set(self, canvas_rgb: np.ndarray) -> None:
        self.canvas = canvas_rgb.astype(np.uint8)

    def peek(self, hand: bool = False) -> np.ndarray:
        frame = cv2.cvtColor(self.canvas, cv2.COLOR_RGB2BGR).copy()
        if hand:
            self._n += 1
            h, w = frame.shape[:2]
            x0, x1, y0, y1 = self.hand_box
            cx = int(w * (x0 + (x1 - x0) * ((self._n * 0.37) % 1.0)))
            cy = int(h * (y0 + (y1 - y0) * ((self._n * 0.61) % 1.0)))
            cv2.ellipse(frame, (cx, cy), (w // 8, h // 6), 0, 0, 360, (40, 60, 90), -1)
        return frame

    def capture(self) -> np.ndarray:
        return cv2.cvtColor(self.canvas, cv2.COLOR_RGB2BGR).copy()


def run(watcher, feed: SimFeed, script, dt: float = 0.5, t0: float = 0.0):
    """Drive `watcher` through `script`, a list of (canvas_rgb, hand_s, still_s).

    For each entry: set the canvas, show the hand for `hand_s`, then a still canvas for
    `still_s`, ticking every `dt`. Returns [(now, Event), ...].
    """
    now, events = t0, []
    for canvas, hand_s, still_s in script:
        feed.set(canvas)
        for hand, dur in ((True, hand_s), (False, still_s)):
            for _ in range(int(round(dur / dt))):
                now += dt
                ev = watcher.tick(feed.peek(hand=hand), now)
                if ev:
                    events.append((now, ev))
    return events
