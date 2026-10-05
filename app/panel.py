from __future__ import annotations

import json
import queue
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import cv2
import numpy as np

from layers.palette import PIGMENTS

KEYS = ("next", "outline", "skip", "quit")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_CANVAS_PX = 4096 * 4096

# The thirteen tubes the layer stack mixes from (swatches on the page).
PIGMENT_RGB = dict(PIGMENTS)


class Panel:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, demo: Path | None = None):
        self.host, self.port, self.demo = host, port, demo
        self._lock = threading.Lock()
        self._state: dict = {"version": 0, "phase": "working", "stages": []}
        self._stage_jpegs: dict[int, bytes] = {}
        self._keys: queue.Queue[str] = queue.Queue()
        self._choices: queue.Queue[tuple[str, bytes | None]] = queue.Queue()
        self._canvas: np.ndarray | None = None      # the painting (BGR)
        self._guide = b""                           # the step drawn over it (PNG)
        self._server: ThreadingHTTPServer | None = None

    @property
    def url(self) -> str:
        return f"http://{'localhost' if self.host in ('127.0.0.1', '0.0.0.0') else self.host}:{self.port}/"

    def start(self) -> "Panel":
        panel = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):          # keep the terminal for watcher events
                pass

            def _send(self, body: bytes, ctype: str, status=HTTPStatus.OK):
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, value, status=HTTPStatus.OK):
                self._send(json.dumps(value).encode(), "application/json", status)

            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/":
                    self._send(PAGE.encode(), "text/html; charset=utf-8")
                elif path == "/state.json":
                    with panel._lock:
                        body = json.dumps(panel._state).encode()
                    self._send(body, "application/json")
                elif path.startswith("/stage/") and path.endswith(".jpg"):
                    with panel._lock:
                        body = panel._stage_jpegs.get(int(path[7:-4]) if path[7:-4].isdigit() else -1, b"")
                    self._send(body, "image/jpeg") if body else self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
                elif path == "/demo.jpg" and panel.demo is not None:
                    self._send(panel.demo.read_bytes(), "image/jpeg")
                elif path == "/canvas.png" and panel._canvas is not None:
                    self._send(cv2.imencode(".png", panel.canvas())[1].tobytes(), "image/png")
                elif path == "/guide.png":
                    with panel._lock:
                        body = panel._guide
                    self._send(body, "image/png") if body else self._send(b"", "text/plain", HTTPStatus.NO_CONTENT)
                else:
                    self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                if self.path in ("/upload", "/demo"):
                    if panel._state.get("phase") != "choose":
                        return self._json({"ok": False, "error": "hang on, I'm not quite ready for a picture yet"}, HTTPStatus.CONFLICT)
                    if self.path == "/demo":
                        panel._choices.put(("demo", None))
                        return self._json({"ok": True})
                    if not 0 < n <= MAX_UPLOAD_BYTES:
                        return self._json({"ok": False, "error": "that picture's a little big - keep it under 25 MB"},
                                          HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                    data = self.rfile.read(n)
                    if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is None:
                        return self._json({"ok": False, "error": "I can't quite read that picture - try a JPG or PNG"}, HTTPStatus.BAD_REQUEST)
                    panel._choices.put((unquote(self.headers.get("X-Filename") or "upload.jpg"), data))
                    return self._json({"ok": True})
                if self.path == "/canvas":
                    if panel._canvas is None or not 0 < n <= MAX_CANVAS_PX * 4:
                        return self._json({"ok": False}, HTTPStatus.BAD_REQUEST)
                    img = cv2.imdecode(np.frombuffer(self.rfile.read(n), np.uint8), cv2.IMREAD_COLOR)
                    if img is None or img.shape != panel._canvas.shape:
                        return self._json({"ok": False}, HTTPStatus.BAD_REQUEST)
                    with panel._lock:
                        panel._canvas = img
                    return self._json({"ok": True})
                if self.path != "/key":
                    return self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
                try:
                    key = json.loads(self.rfile.read(n) or b"{}").get("key")
                except (ValueError, AttributeError):
                    key = None
                if key not in KEYS:
                    return self._json({"ok": False}, HTTPStatus.BAD_REQUEST)
                panel._keys.put(key)
                self._json({"ok": True})

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def _set(self, **fields) -> None:
        with self._lock:
            self._state.update(fields, version=self._state["version"] + 1)

    # ---- start screen ----------------------------------------------------------------
    def wait_choice(self) -> tuple[str, bytes | None] | None:
        """Show the start screen until the painter picks: ("demo", None), (filename, bytes) for an
        upload, or None if the page quit."""
        self._set(phase="choose", demo=self.demo is not None)
        while True:
            try:
                return self._choices.get(timeout=0.1)
            except queue.Empty:
                if self.key(0) == "quit":
                    return None

    # ---- progress screen -------------------------------------------------------------
    def stages(self, names: list[str]) -> None:
        self._set(phase="working", stages=[{"name": n, "state": "pending", "detail": ""} for n in names])

    def stage(self, i: int, state: str, detail: str = "", image: np.ndarray | None = None) -> None:
        """state: pending | running | done | failed. `image` (BGR) is shown under the stage."""
        jpeg = _jpeg(image, 1440) if image is not None else None
        with self._lock:
            self._state["stages"][i].update(state=state, detail=detail)
            if jpeg is not None:
                self._stage_jpegs[i] = jpeg
                self._state["stages"][i]["image"] = f"/stage/{i}.jpg?v={self._state['version'] + 1}"
            self._state["version"] += 1

    def saved(self, path: str) -> None:
        """Where the painting was saved; the page keeps showing it after the server stops."""
        self._set(saved=path)

    def fail(self, detail: str) -> None:
        """Mark the first unfinished stage failed (a no-op if one already is), so a crash between
        stages shows on the page instead of leaving it waiting."""
        with self._lock:
            stages = self._state.get("stages", [])
            if not any(g["state"] == "failed" for g in stages):
                todo = [g for g in stages if g["state"] in ("pending", "running")]
                if todo:
                    todo[0].update(state="failed", detail=detail)
            self._state["version"] += 1

    # ---- lesson screen ---------------------------------------------------------------
    def canvas_setup(self, size: tuple[int, int], bare: int) -> None:
        """A blank (w, h) canvas of grey `bare` to paint on, kept across page reloads."""
        w, h = size
        with self._lock:
            self._canvas = np.full((h, w, 3), bare, np.uint8)
        self._set(canvas={"w": w, "h": h, "bare": bare},
                  tubes=[{"pigment": k, "rgb": v} for k, v in PIGMENT_RGB.items()])

    def canvas(self) -> np.ndarray:
        with self._lock:
            return self._canvas.copy()

    def guide(self, bgra: np.ndarray | None) -> None:
        """What to draw over the canvas (BGRA, canvas size); None clears it."""
        png = cv2.imencode(".png", bgra)[1].tobytes() if bgra is not None else b""
        with self._lock:
            self._guide = png
        self._set(guide_version=self._state["version"] + 1)

    def update(self, *, index: int, count: int, outline: bool, lesson: dict | None = None,
               lessons: list[dict] | None = None, status: str | None = None, tone: str | None = None,
               painted: int | None = None) -> None:
        """Publish the lesson screen. `tone`: "done" shows the status as good news. `painted`: % of
        the step's area painted, if measured."""
        mix = [{"pigment": m["pigment"], "parts": m["parts"], "rgb": PIGMENT_RGB.get(m["pigment"])}
               for m in (lesson or {}).get("mix", [])]
        self._set(phase="lesson", index=index, count=count, outline=outline, status=status, tone=tone,
                  painted=painted, lesson=lesson | {"mix": mix} if lesson else None,
                  lessons=[{"index": l["index"], "name": l["name"]} for l in (lessons or [])])

    def key(self, timeout_s: float = 0.05) -> str | None:
        try:
            return self._keys.get(timeout=timeout_s) if timeout_s > 0 else self._keys.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()


def _jpeg(img: np.ndarray, longest: int) -> bytes:
    h, w = img.shape[:2]
    if max(h, w) > longest:
        img = cv2.resize(img, (round(w * longest / max(h, w)), round(h * longest / max(h, w))))
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RobBoss</title>
<style>
:root { --bg:#f6f4ef; --card:#fff; --ink:#1d1d1f; --muted:#6b6b70; --line:#e2ded6;
        --accent:#c2410c; --ok:#15803d; --warn:#b45309; --bad:#b91c1c; }
@media (prefers-color-scheme: dark) { :root { --bg:#141416; --card:#1e1e22; --ink:#f2f2f2;
        --muted:#9a9aa2; --line:#2e2e34; --accent:#fb923c; --ok:#4ade80; --warn:#fbbf24; --bad:#f87171; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.45 system-ui, sans-serif; }
[hidden] { display:none !important; }
main { max-width:1500px; margin:0 auto; padding:16px; display:grid; gap:16px;
       grid-template-columns: minmax(0,1.7fr) minmax(0,1fr) 200px; }
@media (max-width: 1000px) { main { grid-template-columns: 1fr; } }
.narrow { max-width:640px; margin:0 auto; padding:16px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; }
h1 { font-size:26px; margin:2px 0 10px; }
.kicker { color:var(--muted); font-size:13px; text-transform:uppercase; letter-spacing:.06em; }
img { width:100%; border-radius:8px; display:block; }
.demo { display:flex; gap:14px; align-items:center; margin:12px 0 4px; }
.demo img { width:160px; flex:none; }
.or { text-align:center; color:var(--muted); font-size:14px; margin:14px 0 0; }
.drop { border:2px dashed var(--line); border-radius:12px; padding:32px 16px; text-align:center;
        cursor:pointer; color:var(--muted); margin:12px 0; display:block; }
.drop.over { border-color:var(--accent); color:var(--ink); }
.drop img { max-height:320px; object-fit:contain; margin:0 auto 10px; }
.error { color:var(--bad); font-weight:600; margin-top:8px; }
.stages { list-style:none; padding:0; margin:12px 0 0; }
.stages li { display:flex; gap:12px; padding:10px 0; border-top:1px solid var(--line); color:var(--ink); }
.stages li:first-child { border-top:0; }
.icon { width:22px; flex:none; text-align:center; font-weight:700; }
.stages .pending { color:var(--muted); }
.stages .done .icon { color:var(--ok); } .stages .failed .icon { color:var(--bad); }
.stages .running .icon { color:var(--accent); animation:pulse 1s infinite alternate; }
@keyframes pulse { from { opacity:.3 } to { opacity:1 } }
.detail { color:var(--muted); font-size:14px; }
.mix { display:flex; flex-wrap:wrap; gap:8px; margin:8px 0 14px; }
.chip { display:flex; align-items:center; gap:8px; border:1px solid var(--line); border-radius:999px;
        padding:4px 12px 4px 4px; font-size:15px; }
.dot { width:26px; height:26px; border-radius:50%; border:1px solid rgba(0,0,0,.2); flex:none; }
.target { display:flex; align-items:center; gap:6px; color:var(--muted); font-size:14px; }
.target .dot { width:30px; height:30px; border-radius:8px; }
.label { font-weight:600; margin-top:12px; }
.status { margin-top:14px; padding:12px; border-radius:8px; border-left:4px solid var(--warn);
          background:color-mix(in srgb, var(--warn) 12%, transparent); font-weight:600; }
.status.done { border-color:var(--ok); background:color-mix(in srgb, var(--ok) 12%, transparent); }
.progress { margin-top:14px; font-size:14px; color:var(--muted); }
.bar { height:8px; border-radius:999px; background:var(--line); overflow:hidden; margin-top:6px; }
.bar > div { height:100%; background:var(--ok); transition:width .4s; }
.buttons { display:flex; flex-wrap:wrap; gap:8px; margin-top:16px; }
button { font:inherit; padding:10px 16px; border-radius:8px; border:1px solid var(--line);
         background:var(--card); color:var(--ink); cursor:pointer; }
button.primary { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }
button:disabled { opacity:.5; cursor:default; }
ol { margin:0; padding-left:22px; }
#steps li { padding:3px 0; color:var(--muted); }
#steps li.done { text-decoration:line-through; }
#steps li.now { color:var(--ink); font-weight:700; }
kbd { border:1px solid var(--line); border-radius:4px; padding:0 5px; font-size:12px; }
.hint { color:var(--muted); font-size:13px; margin-top:10px; }
.stage-img { margin-top:8px; max-width:100%; }
#work-view { max-width:960px; }
#sheet-link { display:block; margin-top:14px; }
.tools { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin:8px 0 12px; font-size:14px; }
.swatch { width:30px; height:30px; padding:0; border-radius:6px; border:2px solid var(--line); }
.swatch.on { border-color:var(--ink); box-shadow:0 0 0 2px var(--card) inset; }
.tools input[type=color] { width:38px; height:32px; padding:0; border:0; background:none; }
.tools button { padding:6px 12px; }
.easel { position:relative; line-height:0; }
.easel canvas { width:100%; border-radius:8px; touch-action:none; cursor:crosshair; box-shadow:0 0 0 1px var(--line); }
.easel img { position:absolute; inset:0; height:100%; pointer-events:none; }
</style></head>
<body>
<div id="choose-view" class="narrow" hidden><section class="card">
  <div class="kicker">RobBoss</div><h1>What would you like to paint today?</h1>
  <div class="demo" id="demo-row" hidden>
    <img src="/demo.jpg" alt="A Bob Ross style sunset over a lake">
    <div><div>A Bob Ross style sunset, ten steps from sky to highlights.</div>
      <div class="buttons"><button id="demo" class="primary">Paint the demo</button></div></div>
  </div>
  <div class="or" id="or" hidden>or paint your own picture (a landscape works best)</div>
  <label class="drop" id="drop">
    <img id="upload-preview" alt="" hidden>
    <div id="drop-text">Drop a picture right here, or click to pick one (JPG or PNG)</div>
    <input type="file" id="file" accept="image/*" hidden>
  </label>
  <div class="buttons"><button id="start" class="primary" disabled>Paint my picture</button>
    <button data-key="quit">Quit</button></div>
  <div id="upload-error" class="error" hidden></div>
</section></div>

<div id="work-view" class="narrow" hidden><section class="card">
  <div class="kicker">Getting ready</div><h1>Let's get our paints ready</h1>
  <ul class="stages" id="stages"></ul>
  <div class="buttons"><button data-key="quit">Cancel</button></div>
</section></div>

<main id="lesson-view" hidden>
  <section class="card"><div class="kicker">Your canvas</div>
    <div class="tools">
      <span id="swatches" class="tools" style="margin:0"></span>
      <input type="color" id="colour" value="#3c5a8c" title="Any colour">
      <label>Brush <input type="range" id="size" min="2" max="120" value="24"></label>
      <button id="undo" title="Ctrl+Z">Undo</button>
      <label><input type="checkbox" id="guide-on" checked> Guide</label>
    </div>
    <div class="easel"><canvas id="paint"></canvas><img id="guide" alt="" hidden></div>
  </section>
  <section class="card">
    <div class="kicker" id="count">Step</div>
    <h1 id="name"></h1>
    <div id="lesson"></div>
    <div id="progress" class="progress" hidden><span id="progress-text"></span>
      <div class="bar"><div id="progress-bar"></div></div></div>
    <div id="status" class="status" hidden></div>
    <div class="buttons">
      <button class="primary" data-key="next">I'm happy with it →</button>
      <button data-key="skip">Let this one be</button>
      <button id="outline" data-key="outline">Guide lines</button>
      <button data-key="quit">Quit</button>
    </div>
    <div class="hint">I'll take a peek about 2 s after your brush leaves the area · <kbd>Space</kbd> I'm happy with it · <kbd>K</kbd> let it be · <kbd>O</kbd> guide lines · <kbd>Esc</kbd> quit</div>
  </section>
  <aside class="card"><div class="kicker">Today's lesson</div><ol id="steps"></ol>
    <a id="sheet-link" target="_blank" hidden><div class="kicker">All the happy little layers</div><img id="sheet" alt="Every step"></a></aside>
</main>
<div id="gone" class="narrow" hidden><section class="card"><h1>That's all for today</h1>
  <p id="saved" class="status done" hidden></p>
  <p class="detail">RobBoss has stopped. Run it again whenever you're ready to paint some more.</p></section></div>
<script>
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
let version = -1, phase = null, chosen = null;

function send(key) {
  fetch("/key", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({key})})
    .then(poll).catch(() => {});
}
document.querySelectorAll("button[data-key]").forEach(b => b.onclick = () => send(b.dataset.key));
document.addEventListener("keydown", e => {
  if (phase !== "lesson") { if (e.key === "Escape") send("quit"); return; }
  if ((e.ctrlKey || e.metaKey) && e.key === "z") { e.preventDefault(); return undoStroke(); }
  if (e.target.closest("input")) return;     // arrows on the brush slider are not "next"
  const k = {" ":"next","ArrowRight":"next","n":"next","N":"next",
             "o":"outline","O":"outline","k":"skip","K":"skip","Escape":"quit","q":"quit","Q":"quit"}[e.key];
  if (k) { e.preventDefault(); send(k); }
});

// ---- start screen: the demo, or an upload
function choose(file) {
  if (!file || !file.type.startsWith("image/")) return showError("Hmm, that doesn't look like a picture. Try a JPG or PNG.");
  chosen = file; $("upload-error").hidden = true;
  $("upload-preview").src = URL.createObjectURL(file); $("upload-preview").hidden = false;
  $("drop-text").textContent = file.name + " - click to pick a different one";
  $("start").disabled = false;
}
function showError(msg) { $("upload-error").textContent = msg; $("upload-error").hidden = false; }
$("file").onchange = e => choose(e.target.files[0]);
const drop = $("drop");
drop.ondragover = e => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = e => { e.preventDefault(); drop.classList.remove("over"); choose(e.dataTransfer.files[0]); };
function submit(url, init) {
  return fetch(url, init).then(r => r.json()).then(j => { if (!j.ok) throw new Error(j.error); poll(); });
}
$("demo").onclick = () => {
  $("demo").disabled = true;
  submit("/demo", {method:"POST"}).catch(err => { showError(err.message); $("demo").disabled = false; });
};
$("start").onclick = () => {
  if (!chosen) return;
  $("start").disabled = true; $("start").textContent = "Sending it over…";
  submit("/upload", {method:"POST", headers:{"Content-Type":chosen.type, "X-Filename":encodeURIComponent(chosen.name)},
                     body:chosen})
    .catch(err => { showError("That one didn't quite make it: " + err.message); $("start").disabled = false; })
    .finally(() => { $("start").textContent = "Paint my picture"; });
};

// ---- progress screen
const ICON = {pending:"○", running:"●", done:"✓", failed:"✕"};
function renderStages(s) {
  $("stages").innerHTML = s.stages.map(g =>
    `<li class="${g.state}"><span class="icon">${ICON[g.state] || ""}</span><div><div>${esc(g.name)}</div>` +
    (g.detail ? `<div class="detail">${esc(g.detail)}</div>` : "") +
    (g.image ? `<img class="stage-img" src="${esc(g.image)}" alt="">` : "") + `</div></li>`).join("");
}

// ---- lesson screen
const rgbCss = rgb => `rgb(${rgb.join(",")})`;
function renderLesson(s) {
  $("count").textContent = `Step ${s.index} of ${s.count} · I'm watching`;
  const L = s.lesson;
  $("name").textContent = L ? L.name : "";
  $("lesson").innerHTML = !L ? "" :
    `<div class="label">Let's mix up</div><div>${esc(L.mix_description)}</div><div class="mix">` +
    L.mix.map(m => `<span class="chip"><span class="dot" style="background:${m.rgb ? rgbCss(m.rgb) : "#888"}"></span>${esc(m.pigment)}</span>`).join("") +
    `</div><div class="target">` + (L.tones || [L.target_rgb]).map(t => `<span class="dot" style="background:${rgbCss(t)}"></span>`).join("") +
    `&nbsp;the colours we're after · brush: <b>${esc(L.brush)}</b></div>` +
    `<div class="label">How we'll do it</div><div>${esc(L.technique)}</div>` +
    `<div class="label">You'll know it's done when</div><div>${esc(L.success)}</div>`;
  $("status").hidden = !s.status;
  $("status").textContent = s.status || "";
  $("status").classList.toggle("done", s.tone === "done");
  $("progress").hidden = s.painted == null;
  if (s.painted != null) {
    $("progress-text").textContent = `About ${s.painted}% of this area painted so far`;
    $("progress-bar").style.width = `${Math.min(100, s.painted)}%`;
  }
  $("outline").textContent = `Guide lines: ${s.outline ? "on" : "off"}`;
  $("steps").innerHTML = (s.lessons || []).map(l =>
    `<li class="${l.index < s.index ? "done" : l.index === s.index ? "now" : ""}">${esc(l.name)}</li>`).join("");
  const sheet = (s.stages || []).map(g => g.image).filter(Boolean).pop();
  $("sheet-link").hidden = !sheet;
  if (sheet && $("sheet-link").getAttribute("href") !== sheet) { $("sheet-link").href = sheet; $("sheet").src = sheet; }
  renderPaint(s);
}

// ---- the canvas (app/screen.py)
const cv = $("paint"), ctx = cv.getContext("2d");
let canvasKey = null, guideVersion = -1, last = null, history = [], paintIndex = null;
const hex = rgb => "#" + rgb.map(v => v.toString(16).padStart(2, "0")).join("");
function postCanvas() { cv.toBlob(b => fetch("/canvas", {method:"POST", body:b}).catch(() => {}), "image/png"); }
function undoStroke() { if (history.length) { ctx.putImageData(history.pop(), 0, 0); postCanvas(); } }
function pos(e) {
  const r = cv.getBoundingClientRect();
  return [(e.clientX - r.left) * cv.width / r.width, (e.clientY - r.top) * cv.height / r.height];
}
function stroke(a, b) {
  ctx.strokeStyle = $("colour").value;
  ctx.lineWidth = $("size").value * cv.width / cv.getBoundingClientRect().width;
  ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
}
cv.onpointerdown = e => {
  cv.setPointerCapture(e.pointerId);
  history.push(ctx.getImageData(0, 0, cv.width, cv.height));
  if (history.length > 30) history.shift();
  last = pos(e); stroke(last, last);
};
cv.onpointermove = e => { if (last) { const p = pos(e); stroke(last, p); last = p; } };
cv.onpointerup = cv.onpointercancel = () => { if (last) { last = null; postCanvas(); } };
$("undo").onclick = undoStroke;
$("guide-on").onchange = () => $("guide").hidden = !$("guide-on").checked || guideVersion < 0;
function pick(h) {                               // h: "#rrggbb"
  $("colour").value = h;
  document.querySelectorAll(".swatch").forEach(b => b.classList.toggle("on", b.dataset.hex === h));
}
$("colour").oninput = () => pick($("colour").value);
function renderPaint(s) {
  if (!s.canvas) return;
  const key = `${s.canvas.w}x${s.canvas.h}`;
  if (canvasKey !== key) {                       // first time (or a reload): pick up the server's canvas
    canvasKey = key; cv.width = s.canvas.w; cv.height = s.canvas.h; history = [];
    ctx.lineCap = ctx.lineJoin = "round";
    const img = new Image();
    img.onload = () => ctx.drawImage(img, 0, 0);
    img.src = `/canvas.png?t=${Date.now()}`;
  }
  const L = s.lesson, b = s.canvas.bare;
  // this step's own colours first (darkest to lightest, from the watcher), else the layer's target colour
  const tones = (L && L.tones && L.tones.length ? L.tones : L && L.target_rgb ? [L.target_rgb] : [])
    .map((rgb, i, all) => ({pigment: all.length > 1 ? `this step, ${["darker", "middle", "lighter"][Math.round(2 * i / (all.length - 1))]}` : "this step", rgb}));
  const swatches = [...tones, ...(s.tubes || []), {pigment:"bare canvas (eraser)", rgb:[b, b, b]}];
  $("swatches").innerHTML = swatches.map(w =>
    `<button class="swatch" title="${esc(w.pigment)}" data-hex="${hex(w.rgb)}" style="background:${hex(w.rgb)}"></button>`).join("");
  document.querySelectorAll(".swatch").forEach(el => el.onclick = () => pick(el.dataset.hex));
  const start = tones[Math.floor(tones.length / 2)];
  pick(paintIndex !== s.index && start ? hex(start.rgb) : $("colour").value);  // a new step starts on its colour
  paintIndex = s.index;
  if (s.guide_version !== guideVersion) {
    guideVersion = s.guide_version ?? -1;
    $("guide").src = `/guide.png?v=${guideVersion}`;
  }
  $("guide").hidden = !$("guide-on").checked || guideVersion < 0;
}

function show(view) {
  for (const v of ["choose-view", "work-view", "lesson-view", "gone"]) $(v).hidden = v !== view;
}
function poll() {
  return fetch("/state.json", {cache:"no-store"}).then(r => r.json()).then(s => {
    if (s.version === version) return;
    version = s.version; phase = s.phase;
    if (s.saved) { $("saved").textContent = `Your painting is saved to ${s.saved}`; $("saved").hidden = false; }
    if (s.phase === "choose") { show("choose-view"); $("demo-row").hidden = $("or").hidden = !s.demo; }
    else if (s.phase === "working") { show("work-view"); renderStages(s); }
    else if (s.phase === "lesson") { show("lesson-view"); renderLesson(s); }
  }).catch(() => { show("gone"); version = -1; });
}
setInterval(poll, 300); poll();
</script></body></html>
"""
