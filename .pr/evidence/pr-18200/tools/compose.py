"""Compose the before/after video for OpenHands PR #18200.

Inputs: clips/<run>-<scene>.mp4 and .jsonl from record_scene.sh (control-openhands
browser record + logged click boxes). Output: out/pr-18200-tablet-navigation.mp4
"""

import json
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from changes import changes  # noqa: E402

S = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1920, 1080, 30
SRC_W, SRC_H, SRC_FPS = 820, 1180, 10
SCALE = 0.83
PANE_W, PANE_H = round(SRC_W * SCALE), round(SRC_H * SCALE)
PANE_X, PANE_Y = 140, (H - PANE_H) // 2
PANEL_X = PANE_X + PANE_W + 110
PANEL_R = W - 90

BG = (13, 14, 17)
TEXT = (244, 244, 245)
SUB = (170, 172, 180)
MUTED = (113, 116, 125)
CHIP_BG = (28, 30, 35)
CHIP_BORDER = (52, 55, 63)
RED = (242, 87, 87)
BLUE = (84, 145, 255)

INTER = "/usr/share/fonts/opentype/inter/"
_fonts = {}


def font(weight, size):
    key = (weight, size)
    if key not in _fonts:
        name = {"regular": "Inter-Regular.otf", "medium": "Inter-Medium.otf",
                "semibold": "Inter-SemiBold.otf", "bold": "Inter-Bold.otf",
                "mono": None}[weight]
        path = INTER + name if name else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
        _fonts[key] = ImageFont.truetype(path, size)
    return _fonts[key]


def ease(x):
    x = min(1.0, max(0.0, x))
    return 4 * x**3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ramp(t, start, dur):
    return min(1.0, max(0.0, (t - start) / dur)) if dur > 0 else float(t >= start)


def with_alpha(color, a):
    return (*color, int(round(255 * max(0.0, min(1.0, a)))))


# ---------------------------------------------------------------- sprites

def supersampled(size, draw_fn, factor=4):
    """Draw at `factor`x and downsample: anti-aliased shapes from PIL."""
    big = Image.new("RGBA", (size[0] * factor, size[1] * factor), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(big), factor)
    return big.resize(size, Image.LANCZOS)


def cursor_sprite(scale=1.55):
    pts = [(0, 0), (0, 17), (4.2, 13.2), (7.2, 20), (10, 18.8), (7.1, 12.3), (12.6, 12.3)]
    pad = 4
    size = (int(14 * scale) + 2 * pad, int(22 * scale) + 2 * pad)

    def draw(d, f):
        poly = [((x * scale + pad) * f, (y * scale + pad) * f) for x, y in pts]
        shadow = [(x + 1.5 * f, y + 2 * f) for x, y in poly]
        d.polygon(shadow, fill=(0, 0, 0, 90))
        d.polygon(poly, fill=(255, 255, 255, 255), outline=(10, 10, 10, 255))
        d.line(poly + [poly[0]], fill=(10, 10, 10, 255), width=int(1.6 * f), joint="curve")

    return supersampled(size, draw), pad


CURSOR, CURSOR_PAD = cursor_sprite()


def paste_ring(layer, cx, cy, radius, width, color, alpha, fill_alpha=0.0):
    r = int(math.ceil(radius + width + 2))
    size = (2 * r, 2 * r)

    def draw(d, f):
        box = [(r - radius) * f, (r - radius) * f, (r + radius) * f, (r + radius) * f]
        if fill_alpha > 0:
            d.ellipse(box, fill=with_alpha(color, fill_alpha))
        d.ellipse(box, outline=with_alpha(color, alpha), width=max(1, int(width * f)))

    tile = supersampled(size, draw)
    layer.alpha_composite(tile, (int(cx - r), int(cy - r)))


def paste_round_rect(layer, box, radius, color, alpha, width=3, fill_alpha=0.0):
    x0, y0, x1, y1 = box
    pad = width + 2
    size = (int(x1 - x0) + 2 * pad, int(y1 - y0) + 2 * pad)

    def draw(d, f):
        rect = [pad * f, pad * f, (pad + x1 - x0) * f, (pad + y1 - y0) * f]
        if fill_alpha > 0:
            d.rounded_rectangle(rect, radius * f, fill=with_alpha(color, fill_alpha))
        d.rounded_rectangle(rect, radius * f, outline=with_alpha(color, alpha), width=int(width * f))

    tile = supersampled(size, draw, factor=3)
    layer.alpha_composite(tile, (int(x0 - pad), int(y0 - pad)))


# ---------------------------------------------------------------- clips

def to_canvas(x, y):
    return PANE_X + x * SCALE, PANE_Y + y * SCALE


def load_frames(path):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True).stdout
    arr = np.frombuffer(raw, np.uint8).reshape(-1, SRC_H, SRC_W, 3)
    return [Image.fromarray(a).resize((PANE_W, PANE_H), Image.LANCZOS) for a in arr]


def load_clip(name):
    path = f"{S}/clips/{name}.mp4"
    events = [json.loads(line) for line in open(f"{S}/clips/{name}.jsonl")]
    rest = next(e for e in events if e["kind"] == "rest")
    clicks = [e for e in events if e["kind"] == "click"]
    big = [c["t"] for c in changes(path) if c["mean"] > 1.0]
    # The first click's effect is the first big change; later clicks keep their
    # logged spacing, snapped to the change they caused.
    first = big[0]
    times = []
    for c in clicks:
        pred = first + (c["t0"] - clicks[0]["t0"]) / 1000
        near = [b for b in big if pred - 0.25 <= b <= pred + 0.45]
        times.append((near[0] if near else pred) - 0.08)
    frames = load_frames(path)
    return {
        "frames": frames,
        "duration": len(frames) / SRC_FPS,
        "rest": (rest["x"] + rest["w"] / 2, rest["y"] + rest["h"] / 2),
        "clicks": [dict(c, vt=t) for c, t in zip(clicks, times)],
    }


# ---------------------------------------------------------------- scenes

SCENES = {
    "before-settings": {
        "accent": RED, "badge": "BEFORE", "ref": "main · 4a8f77e",
        "title": "Settings: Application → LLM → Secrets",
        "steps": ["Find the gear at the bottom of the rail",
                  "Settings list opens: choose LLM",
                  "Back down to the gear",
                  "Settings list again: choose Secrets"],
        "start": "Application",
        "pages": [("Settings list", True), ("LLM", False), ("Settings list", True), ("Secrets", False)],
    },
    "before-customize": {
        "accent": RED, "badge": "BEFORE", "ref": "main · 4a8f77e",
        "title": "Customize: MCP Servers → Skills",
        "steps": ["Customize in the rail", "Customize list opens: choose Skills"],
        "start": "MCP Servers",
        "pages": [("Customize list", True), ("Skills", False)],
    },
    "after-settings": {
        "accent": BLUE, "badge": "AFTER", "ref": "PR #18200 · d84357b",
        "title": "Settings: Application → LLM → Secrets",
        "steps": ["Open the Settings selector on the page",
                  "Choose LLM",
                  "Open it again",
                  "Choose Secrets"],
        "start": "Application",
        "pages": [None, ("LLM", False), None, ("Secrets", False)],
    },
    "after-customize": {
        "accent": BLUE, "badge": "AFTER", "ref": "PR #18200 · d84357b",
        "title": "Customize: MCP Servers → Skills",
        "steps": ["Open the Customize selector", "Choose Skills"],
        "start": "MCP Servers",
        "pages": [None, ("Skills", False)],
    },
}
# Before clicks navigate on every step; after clicks navigate on every second.
SCENES["before-customize"]["pages"] = [("Customize list", True), ("Skills", False)]

LEAD, END_HOLD = 0.9, 1.3


def cursor_plan(clip):
    """Waypoints: (move_start, move_end, from, to) per click, in clip time."""
    plan = []
    pos = to_canvas(*clip["rest"])
    prev_click = -1.0
    for c in clip["clicks"]:
        target = to_canvas(c["x"] + c["w"] / 2, c["y"] + c["h"] / 2)
        dist = math.dist(pos, target)
        dur = min(0.95, 0.38 + dist / 1500)
        end = c["vt"] - 0.14
        start = max(prev_click + 0.3, end - dur)
        plan.append((start, end, pos, target))
        pos, prev_click = target, c["vt"]
    return plan


def cursor_at(plan, rest, t):
    pos = rest
    for start, end, a, b in plan:
        if t < start:
            return pos
        if t <= end:
            k = ease((t - start) / (end - start))
            return a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k
        pos = b
    return pos


def draw_badge(d, x, y, label, color, size=30):
    f = font("bold", size)
    tw = d.textlength(label, font=f)
    padx, h = 18, size + 22
    d.rounded_rectangle([x, y, x + tw + 2 * padx, y + h], h // 2, fill=with_alpha(color, 0.16),
                        outline=with_alpha(color, 0.9), width=2)
    d.text((x + padx, y + h / 2), label, font=f, fill=color, anchor="lm")
    return x + tw + 2 * padx


def draw_chips(d, x0, y0, chips, max_x):
    """chips: [(label, detour, alpha)] laid out with arrows, wrapping."""
    f = font("medium", 23)
    x, y = x0, y0
    h, gap = 46, 8
    for i, (label, detour, a) in enumerate(chips):
        if a <= 0:
            continue
        tw = d.textlength(label, font=f)
        w = tw + 30
        arrow_w = 32 if i else 0
        if x + arrow_w + w > max_x:
            x, y = x0, y + h + 18
        if arrow_w:
            d.text((x + 4, y + h / 2), "→", font=font("regular", 24), fill=with_alpha(MUTED, a), anchor="lm")
            x += arrow_w
        border = RED if detour else CHIP_BORDER
        fg = (255, 175, 175) if detour else (228, 228, 231)
        d.rounded_rectangle([x, y, x + w, y + h], 12, fill=with_alpha(RED if detour else CHIP_BG, 0.14 * a if detour else a),
                            outline=with_alpha(border, a), width=2)
        d.text((x + 15, y + h / 2), label, font=f, fill=with_alpha(fg, a), anchor="lm")
        x += w + gap


def render_clip_frame(bg, clip, scene, t, part):
    cfg = SCENES[scene]
    accent = cfg["accent"]
    frame = bg.copy()
    ct = t - LEAD
    idx = int(max(0.0, min(clip["duration"] - 1e-6, ct)) * SRC_FPS)
    idx = min(idx, len(clip["frames"]) - 1)
    frame.paste(clip["frames"][idx], (PANE_X, PANE_Y))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    # Click targets, rings and the cursor (all in clip time).
    plan = cursor_plan(clip)
    for i, c in enumerate(clip["clicks"]):
        vt = c["vt"]
        start = plan[i][0]
        a = ramp(ct, start + 0.1, 0.2) * (1 - ramp(ct, vt + 0.12, 0.22))
        if a > 0:
            x0, y0 = to_canvas(c["x"], c["y"])
            x1, y1 = to_canvas(c["x"] + c["w"], c["y"] + c["h"])
            paste_round_rect(layer, (x0 - 5, y0 - 5, x1 + 5, y1 + 5), 9, accent, 0.95 * a, width=3,
                             fill_alpha=0.10 * a)
        k = (ct - vt) / 0.55
        if 0 <= k <= 1:
            cx, cy = to_canvas(c["x"] + c["w"] / 2, c["y"] + c["h"] / 2)
            paste_ring(layer, cx, cy, 10 + 34 * ease(k), 4, accent, 1 - k)
            if k < 0.45:
                paste_ring(layer, cx, cy, 9, 0, accent, 0, fill_alpha=0.55 * (1 - k / 0.45))
    rest = to_canvas(*clip["rest"])
    cx, cy = cursor_at(plan, rest, ct)
    ca = ramp(t, 0.15, 0.35)
    press = any(0 <= ct - c["vt"] <= 0.12 for c in clip["clicks"])
    sprite = CURSOR
    if press:
        sprite = CURSOR.resize((int(CURSOR.width * 0.88), int(CURSOR.height * 0.88)), Image.LANCZOS)
    if ca < 1:
        sprite = sprite.copy()
        sprite.putalpha(sprite.getchannel("A").point(lambda v: int(v * ca)))
    layer.alpha_composite(sprite, (int(cx - CURSOR_PAD), int(cy - CURSOR_PAD)))

    # Device frame around the pane.
    d.rounded_rectangle([PANE_X - 12, PANE_Y - 12, PANE_X + PANE_W + 12, PANE_Y + PANE_H + 12], 22,
                        outline=(48, 51, 58, 255), width=2)

    # Caption panel.
    x = PANEL_X
    d.text((x, 92), "OpenHands  ·  tablet, 820 × 1180", font=font("medium", 24), fill=MUTED)
    bx = draw_badge(d, x, 140, cfg["badge"], accent)
    d.text((bx + 22, 140 + 26), cfg["ref"], font=font("mono", 24), fill=SUB, anchor="lm")
    d.text((PANEL_R, 140 + 26), part, font=font("medium", 24), fill=MUTED, anchor="rm")
    d.text((x, 238), cfg["title"], font=font("semibold", 40), fill=TEXT)

    # Steps: each appears as its move begins; the active one is bright.
    y = 322
    for i, label in enumerate(cfg["steps"]):
        c = clip["clicks"][i]
        shown = ramp(ct, plan[i][0] - 0.15, 0.25)
        if shown <= 0:
            continue
        nxt = plan[i + 1][0] if i + 1 < len(plan) else 1e9
        active = ct < nxt - 0.15
        col = TEXT if active else SUB
        num_col = accent if active else MUTED
        d.ellipse([x, y + 4, x + 40, y + 44], fill=with_alpha(num_col, 0.18 * shown),
                  outline=with_alpha(num_col, shown), width=2)
        d.text((x + 20, y + 24), str(i + 1), font=font("bold", 22), fill=with_alpha(num_col, shown), anchor="mm")
        d.text((x + 60, y + 24), label, font=font("medium", 31), fill=with_alpha(col, shown), anchor="lm")
        y += 62

    # Pages visited, growing with each navigation.
    py = 640
    d.text((x, py), "Pages visited", font=font("semibold", 24), fill=MUTED)
    chips = [(cfg["start"], False, 1.0)]
    for i, page in enumerate(cfg["pages"]):
        if page:
            chips.append((page[0], page[1], ramp(ct, clip["clicks"][i]["vt"] + 0.08, 0.2)))
    draw_chips(d, x, py + 44, chips, PANEL_R)
    detours = sum(1 for (lbl, det, a) in chips if det and a > 0.5)
    total = sum(1 for p in cfg["pages"] if p and p[1])
    note = (f"Detours through a list page: {detours}" if total else "Detours through a list page: 0")
    d.text((x, 900), note, font=font("semibold", 30), fill=accent if (detours or not total) else SUB)

    frame = frame.convert("RGBA")
    frame.alpha_composite(layer)
    return frame.convert("RGB")


# ---------------------------------------------------------------- cards

def make_bg():
    g = np.linspace(0, 1, H)[:, None]
    top, bottom = np.array([17, 18, 22]), np.array(BG)
    arr = (top * (1 - g) + bottom * g)[:, None, :].repeat(W, axis=1)
    return Image.fromarray(arr.astype(np.uint8)[:, :, :3].reshape(H, W, 3))


def title_card(bg):
    im = bg.copy().convert("RGBA")
    d = ImageDraw.Draw(im)
    cx = W // 2
    d.text((cx, 330), "OpenHands  ·  PR #18200", font=font("medium", 30), fill=MUTED, anchor="mm")
    d.text((cx, 430), "Switching sections on a tablet", font=font("bold", 76), fill=TEXT, anchor="mm")
    d.text((cx, 525), "Settings and Customize at 768–1023 px, before and after the compact section selector",
           font=font("regular", 32), fill=SUB, anchor="mm")
    lx = cx - 330
    paste_ring(im, lx, 650, 13, 4, RED, 1.0)
    d.text((lx + 30, 650), "click, before", font=font("medium", 26), fill=SUB, anchor="lm")
    paste_ring(im, lx + 300, 650, 13, 4, BLUE, 1.0)
    d.text((lx + 330, 650), "click, after", font=font("medium", 26), fill=SUB, anchor="lm")
    d.text((cx, 760), "Same steps and fresh state in both builds  ·  recorded with verify-openhands",
           font=font("regular", 26), fill=MUTED, anchor="mm")
    return im.convert("RGB")


def section_card(bg, word, color, line):
    im = bg.copy()
    d = ImageDraw.Draw(im)
    d.text((W // 2, 480), word, font=font("bold", 120), fill=color, anchor="mm")
    d.text((W // 2, 590), line, font=font("regular", 34), fill=SUB, anchor="mm")
    return im


def end_card(bg, before_still, after_still):
    im = bg.copy().convert("RGBA")
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    sh = 600
    sw = round(SRC_W * sh / SRC_H)
    gap = 200
    lx = W // 2 - gap // 2 - sw
    rx = W // 2 + gap // 2
    top = 150
    k = sh / SRC_H
    for still, x, color, label, caption, target in (
        (before_still, lx, RED, "BEFORE", "Leave the page, pick from a list", (253, 1134, 36, 36)),
        (after_still, rx, BLUE, "AFTER", "Switch from the page you are on", (316, 32, 478, 44)),
    ):
        im.paste(still.resize((sw, sh), Image.LANCZOS), (x, top))
        tx, ty, tw_, th_ = target
        paste_round_rect(layer, (x + tx * k - 4, top + ty * k - 4, x + (tx + tw_) * k + 4, top + (ty + th_) * k + 4),
                         7, color, 0.95, width=3, fill_alpha=0.12)
        d.rounded_rectangle([x - 8, top - 8, x + sw + 8, top + sh + 8], 16, outline=(48, 51, 58, 255), width=2)
        f = font("bold", 26)
        tw = d.textlength(label, font=f)
        bx = x + (sw - tw - 36) / 2
        d.rounded_rectangle([bx, top - 78, bx + tw + 36, top - 30], 24, fill=with_alpha(color, 0.16),
                            outline=with_alpha(color, 0.9), width=2)
        d.text((bx + 18, top - 54), label, font=f, fill=color, anchor="lm")
        d.text((x + sw / 2, top + sh + 50), caption, font=font("semibold", 32), fill=TEXT, anchor="mm")
    d.text((W // 2, top + sh + 128), "Same two taps per switch: no detour through a list, and the current section stays in view.",
           font=font("regular", 30), fill=SUB, anchor="mm")
    d.text((W // 2, H - 52),
           "Recorded with verify-openhands (control-openhands browser record) on a real local stack  ·  "
           "Agent Server 1.53.0  ·  music synthesized for this video",
           font=font("regular", 22), fill=MUTED, anchor="mm")
    im.alpha_composite(layer)
    return im.convert("RGB")


# ---------------------------------------------------------------- timeline

def main(out):
    bg = make_bg()
    clips = {name: load_clip(name) for name in SCENES}
    for name, clip in clips.items():
        print(name, round(clip["duration"], 2), [round(c["vt"], 2) for c in clip["clicks"]], file=sys.stderr)

    before_hub = clips["before-settings"]["frames"][int((clips["before-settings"]["clicks"][0]["vt"] + 0.6) * SRC_FPS)]
    after_open = clips["after-settings"]["frames"][int((clips["after-settings"]["clicks"][0]["vt"] + 0.6) * SRC_FPS)]
    # End-card stills at full resolution.
    full = {}
    for key, name, k in (("before", "before-settings", 0), ("after", "after-settings", 0)):
        t = clips[name]["clicks"][k]["vt"] + 0.6
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", f"{S}/clips/{name}.mp4",
                              "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True, check=True).stdout
        full[key] = Image.fromarray(np.frombuffer(raw, np.uint8).reshape(SRC_H, SRC_W, 3))
    del before_hub, after_open

    segments = [
        ("still", 3.8, title_card(bg)),
        ("still", 1.9, section_card(bg, "Before", RED, "main  ·  no section navigation on the page")),
        ("clip", "before-settings", "1 / 2"),
        ("clip", "before-customize", "2 / 2"),
        ("still", 1.9, section_card(bg, "After", BLUE, "PR #18200  ·  a section selector above the page")),
        ("clip", "after-settings", "1 / 2"),
        ("clip", "after-customize", "2 / 2"),
        ("still", 6.5, end_card(bg, full["before"], full["after"])),
    ]
    plan = []
    for seg in segments:
        if seg[0] == "still":
            plan.append((seg[1], lambda t, im=seg[2]: im))
        else:
            name, part = seg[1], seg[2]
            dur = LEAD + clips[name]["duration"] + END_HOLD
            plan.append((dur, lambda t, n=name, p=part: render_clip_frame(bg, clips[n], n, t, p)))
    total = sum(p[0] for p in plan)
    print("total seconds", round(total, 2), file=sys.stderr)

    music = f"{S}/out/music.wav"
    subprocess.run([sys.executable, "-I", f"{S}/make_music.py", music, f"{total + 0.5:.2f}"], check=True)
    meas = subprocess.run(["ffmpeg", "-hide_banner", "-i", music, "-af", "ebur128=framelog=quiet", "-f", "null", "-"],
                          capture_output=True, text=True).stderr
    lufs = float(meas.split("Integrated loudness:")[1].split("I:")[1].split("LUFS")[0])
    gain = -25.0 - lufs  # quiet background level
    print("music LUFS", lufs, "gain", round(gain, 1), file=sys.stderr)

    enc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-i", music,
         "-filter:a", f"volume={gain:.2f}dB,afade=t=out:st={total - 3:.2f}:d=3",
         "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", out],
        stdin=subprocess.PIPE)
    fade = 0.35
    n = 0
    for dur, render in plan:
        frames = int(round(dur * FPS))
        for i in range(frames):
            t = i / FPS
            im = render(t)
            a = min(1.0, t / fade, (dur - t) / fade)
            if a < 1:
                im = Image.blend(bg, im, max(0.0, a))
            enc.stdin.write(im.tobytes())
            n += 1
    enc.stdin.close()
    enc.wait()
    print("frames", n, "->", out, file=sys.stderr)


if __name__ == "__main__":
    os.makedirs(f"{S}/out", exist_ok=True)
    main(sys.argv[1] if len(sys.argv) > 1 else f"{S}/out/pr-18200-tablet-navigation.mp4")
