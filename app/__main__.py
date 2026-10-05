from __future__ import annotations

import argparse
import json
import re
import time
import webbrowser
from pathlib import Path

import cv2
import numpy as np

from .panel import Panel
from .scene import scene_steps, step_images, step_overlay
from .screen import Screen

ROOT = Path(__file__).resolve().parent.parent
DEMO_IMAGE = ROOT / "demo" / "bobross-sunset.jpg"
DEMO_SCENE = ROOT / "demo" / "bobross-sunset-layers"        # its layer stack, made ahead of time
UPLOADS = ROOT / "scenes" / "uploads"
ADVANCED = "Beautiful. That one's finished - let's move right along."
COMPLETE = "And there you have it. Your painting's finished - happy painting, friend."


def decompose(image: Path) -> Path:
    """The picture's layer stack (layers.stack), made once and reused from scenes/."""
    if image == DEMO_IMAGE:
        return DEMO_SCENE
    # ponytail: ASCII-only folder name, because cv2.imread/imwrite cannot open non-ASCII paths on
    # Windows (they return None). A non-ASCII repo path would still break; switch to
    # cv2.imdecode(np.fromfile(...)) everywhere if that ever matters.
    scene = ROOT / "scenes" / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', image.stem)}-layers"
    if (scene / "report.json").exists():
        print(f"using existing layers in {scene}")
        return scene
    from layers.stack import decompose as layers_decompose
    print(f"decomposing {image.name} with layers.stack (about 40 s) ...")
    t = time.monotonic()
    report = layers_decompose(image.resolve(), scene)
    print(f"  {report['stage_count']} steps in {time.monotonic() - t:.0f} s -> {scene}")
    return scene


def choose(panel: Panel) -> Path | None:
    """Start screen: the picture to paint (the demo, or the upload saved to scenes/uploads/); None on quit."""
    got = panel.wait_choice()
    if got is None:
        return None
    name, data = got
    if data is None:
        return DEMO_IMAGE
    UPLOADS.mkdir(parents=True, exist_ok=True)
    saved = UPLOADS / f"{time.strftime('%Y%m%d-%H%M%S')}_{Path(name).stem[:40]}{Path(name).suffix.lower() or '.jpg'}"
    saved.write_bytes(data)
    print(f"upload saved to {saved}")
    return saved


def show_breakdown(panel: Panel, i: int, scene: Path) -> None:
    """Put the scene's step contact sheet (layers.stack) under stage i, with the step count."""
    report = json.loads((scene / "report.json").read_text())
    sheet = cv2.imread(str(scene / "steps-contact-sheet.png"))
    panel.stage(i, "done", f"{report['stage_count']} happy little steps, from the back to the front", image=sheet)


def run_stage(panel: Panel, i: int, fn, running: str = ""):
    """Run one pipeline stage, showing it on the page. fn() returns its result and a detail line."""
    panel.stage(i, "running", running)
    try:
        result, detail = fn()
    except Exception as e:
        panel.stage(i, "failed", f"{type(e).__name__}: {e}")
        raise
    panel.stage(i, "done", detail)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765, help="page port")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to open the page from another device")
    ap.add_argument("--no-browser", action="store_true", help="don't open the page automatically")
    args = ap.parse_args()

    panel, screen = Panel(args.host, args.port, demo=DEMO_IMAGE).start(), None
    print(f"RobBoss: {panel.url}")
    if not args.no_browser:
        webbrowser.open(panel.url)
    try:
        image = choose(panel)
        if image is None:
            return
        panel.stages(["Pick your painting", "Finding the layers in your painting", "Planning our lesson together"])
        panel.stage(0, "done", "the demo" if image == DEMO_IMAGE else image.name.split("_", 1)[-1])

        scene = run_stage(panel, 1, lambda: (lambda s: (s, s.name))(decompose(image)),
                          "looking at your picture and splitting it into happy little layers")
        show_breakdown(panel, 1, scene)
        steps = scene_steps(scene)
        images = step_images(scene, steps)
        from lesson.planner import plan
        lesson = run_stage(panel, 2, lambda: (lambda p: (p, f"{len(p)} steps, ready and waiting"))(plan(scene)),
                           "mixing up the colours and working out every stroke")

        from layers.stack import BARE_CANVAS
        h, w = images[0].shape[:2]
        screen = Screen(panel, (w, h), int(BARE_CANVAS))
        paint(screen, panel, image, scene, steps, images, lesson)
    except KeyboardInterrupt:
        pass
    except Exception as e:                      # leave the failed stage on the page until quit
        panel.fail(f"{type(e).__name__}: {e}")
        print(f"failed: {type(e).__name__}: {e}\n(the page shows where; Cancel or Ctrl+C to exit)")
        try:
            while panel.key(0.5) != "quit":
                pass
        except KeyboardInterrupt:
            pass
        raise
    finally:
        if screen:
            # ponytail: assumes the default Downloads folder; a relocated one (OneDrive) still gets a ~/Downloads
            out = Path.home() / "Downloads" / f"RobBoss {time.strftime('%Y-%m-%d %H.%M.%S')}.png"
            if screen.save(out):
                print(f"your painting is saved to {out}")
                panel.saved(str(out))
                time.sleep(0.8)                 # let the page pick the path up before the server stops
        panel.close()


def paint(screen: Screen, panel: Panel, image: Path, scene: Path, steps, images, lesson) -> None:
    """The lesson: the page's canvas -> the watcher -> the next step's guide."""
    from PIL import Image
    from lesson.machine import StepMachine
    from lesson.schema import Verdict
    from lesson.watcher import Watcher

    ref = np.asarray(Image.open(image).convert("RGB"))
    machine = StepMachine(lesson)
    peek_size = (384, round(384 * ref.shape[0] / ref.shape[1]))
    w = Watcher(machine, ref, scene, screen.canvas)
    w.calibrate(screen.canvas())

    outline, shown, tone = False, None, None
    status = "Go ahead and paint this area. Whenever your brush leaves it, I'll take a little peek after about 2 seconds."
    t0 = time.monotonic()
    while machine.status != "complete":
        i = machine.state["current"]
        painted = None if w.coverage is None else round(w.coverage * 100)
        if shown != (i, outline, status, painted):
            if shown is None or shown[:2] != (i, outline):
                screen.show_step(*step_overlay(steps[i], images[i], fill_only=not outline))
            panel.update(index=i + 1, count=len(steps), outline=outline,
                         lesson=machine.steps[i].model_dump() | {"tones": w.expected_tones(machine.steps[i])},
                         lessons=[s.model_dump() for s in machine.steps], status=status, tone=tone,
                         painted=painted)
            shown = (i, outline, status, painted)

        now = time.monotonic() - t0
        ev = w.tick(cv2.resize(screen.canvas(), peek_size, interpolation=cv2.INTER_AREA), now)
        if ev is not None:
            text = ev.verdict.adjustment if ev.verdict and ev.verdict.adjustment else ""
            print(f"[{now:6.1f}s] step {ev.step_index}: {ev.kind} {text}", flush=True)
            if ev.kind in ("advanced", "complete"):
                status, tone = ADVANCED if ev.kind == "advanced" else COMPLETE, "done"
            else:
                status, tone = text, None

        key = panel.key(0.2)
        if key == "quit":
            break
        if key == "next":                                           # painter says it's done
            machine.submit(Verdict(verdict="READY", category="none", adjustment=""))
            w.rebase(screen.canvas())   # the next step is judged from the canvas as it is now
            w._reset_step()
            status, tone = "You're the boss on this canvas. On to the next one.", "done"
        elif key == "skip":
            machine.skip()
            w.rebase(screen.canvas())
            w._reset_step()
            status, tone = "That's fine, we'll let that one be. On to the next one.", "done"
        elif key == "outline":
            outline = not outline
    if machine.status == "complete":
        panel.update(index=len(steps), count=len(steps), outline=outline,
                     lessons=[s.model_dump() for s in machine.steps], status=COMPLETE, tone="done",
                     lesson=machine.steps[-1].model_dump())
        time.sleep(1.0)     # let the page pick up the final state
    print(f"session: {machine.status}, step {machine.state['current'] + 1} of {len(steps)}")


if __name__ == "__main__":
    main()
