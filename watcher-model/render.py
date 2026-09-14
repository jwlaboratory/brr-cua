"""
render.py — synthetic renderer for 5 website types, each in a LOADING and a
LOADED (ready) state, drawn with PIL. This is the training-data source of truth
for the tiny watcher model.

Design note: real apps tend to show loading on a dark/desaturated screen and
content on a light one, but we DON'T want the model to become a mere brightness
detector. So background lightness is randomized INDEPENDENTLY of the phase
(50/50 light vs dark for both classes) and neutral colors adapt for contrast.
That forces the model to learn structure — spinner/skeleton/progress/dots vs
check/chart/play/product — which is the signal that actually generalizes to
real screenshots.

Each render is a square RGB image. Label convention: 0 = loading, 1 = ready.
"""

import math
import numpy as np
from PIL import Image, ImageDraw

SITES = ["booking", "feed", "dashboard", "video", "shop"]
S = 256  # render size (square); downscaled later for the model


def _rrect(d, box, r, fill):
    d.rounded_rectangle(box, radius=r, fill=fill)


def _jit(rng, v, amt):
    return v + rng.uniform(-amt, amt)


def _bg(d, rng, light):
    if light:
        base = rng.randint(232, 246)
        top = (base, base, min(255, base + rng.randint(2, 8)))
        bot = (base - rng.randint(2, 10), base - rng.randint(0, 6), base)
    else:
        base = rng.randint(16, 34)
        top = (base, base, base + rng.randint(6, 20))
        bot = (base + rng.randint(4, 16), base, base + rng.randint(10, 30))
    for y in range(S):
        t = y / S
        c = tuple(int(top[i] * (1 - t) + bot[i] * t) for i in range(3))
        d.line([(0, y), (S, y)], fill=c)


def _ink(rng, light):
    # Text/foreground bar color that contrasts with the background.
    return (rng.randint(28, 55),) * 3 if light else (rng.randint(150, 195),) * 3


def _placeholder(rng, light):
    # Skeleton/shimmer block color: a neutral gray offset from the bg.
    return (rng.randint(198, 214),) * 3 if light else (rng.randint(60, 82),) * 3


def _bars(d, x, y, n, wmax, color, rng, gap=16):
    for i in range(n):
        w = int(rng.uniform(0.4, 1.0) * wmax)
        _rrect(d, [x, y + i * gap, x + w, y + i * gap + 8], 4, color)


def _shimmer(img, rng):
    band = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    bd = ImageDraw.Draw(band)
    off = rng.uniform(-S * 0.3, S * 0.9)
    w = rng.uniform(28, 55)
    bd.polygon([(off, 0), (off + w, 0), (off + w + S, S), (off + S, S)],
               fill=(255, 255, 255, 45))
    img.alpha_composite(band)


# ----------------------------------------------------------------------------
# Per-site renderers: (img, draw, rng, loading, light)
# ----------------------------------------------------------------------------

def _booking(img, d, rng, loading, light):
    cx, cy = S / 2, S / 2 - 10
    if loading:
        r = _jit(rng, 34, 4)
        a0 = rng.uniform(0, 360)
        track = (200, 203, 212) if light else (60, 60, 80)
        d.arc([cx - r, cy - r, cx + r, cy + r], 0, 360, fill=track, width=5)
        d.arc([cx - r, cy - r, cx + r, cy + r], a0, a0 + 90, fill=(129, 140, 248), width=5)
        _bars(d, cx - 60, cy + 60, 1, 120, _placeholder(rng, light), rng)
    else:
        r = 30
        d.ellipse([cx - r, cy - r - 20, cx + r, cy + r - 20], fill=(76, 195, 108))
        d.line([(cx - 14, cy - 20), (cx - 3, cy - 8), (cx + 16, cy - 34)], fill="white", width=6, joint="curve")
        _bars(d, cx - 70, cy + 30, 2, 140, _placeholder(rng, light), rng)
        _rrect(d, [cx - 65, cy + 75, cx + 65, cy + 100], 10, (124, 108, 246))


def _feed(img, d, rng, loading, light):
    x = S * 0.18
    if loading:
        g = _placeholder(rng, light)
        d.ellipse([x, 50, x + 34, 84], fill=g)
        _bars(d, x + 46, 54, 2, 120, g, rng)
        _rrect(d, [x, 100, S - x, 175], 10, g)
        _bars(d, x, 190, 2, 150, g, rng)
    else:
        col = tuple(int(c) for c in rng.randint(80, 220, 3))
        d.ellipse([x, 50, x + 34, 84], fill=col)
        _bars(d, x + 46, 54, 2, 120, _ink(rng, light), rng)
        img2 = tuple(int(c) for c in rng.randint(40, 210, 3))
        _rrect(d, [x, 100, S - x, 175], 10, img2)
        _bars(d, x, 190, 3, 150, _ink(rng, light), rng)


def _dashboard(img, d, rng, loading, light):
    if loading:
        y = S / 2
        track = (210, 213, 220) if light else (43, 45, 52)
        _rrect(d, [40, y - 6, S - 40, y + 6], 6, track)
        frac = rng.uniform(0.15, 0.9)
        _rrect(d, [40, y - 6, 40 + (S - 80) * frac, y + 6], 6, (59, 130, 246))
        _bars(d, 40, y + 26, 1, 120, _placeholder(rng, light), rng)
    else:
        base = S - 60
        n = rng.randint(5, 7)
        bw = (S - 100) / n
        for i in range(n):
            h = rng.uniform(30, 130)
            col = tuple(int(c) for c in rng.randint(80, 230, 3))
            _rrect(d, [50 + i * bw, base - h, 50 + i * bw + bw * 0.6, base], 4, col)
        _bars(d, 50, 40, 2, 110, _ink(rng, light), rng)


def _video(img, d, rng, loading, light):
    # Player pane (always dark, like a real video surface) sits in the upper
    # area; playback controls live BELOW it on the page background.
    pane_bottom = 170
    _rrect(d, [34, 44, S - 34, pane_bottom], 12, (12, 12, 18))
    cx = S / 2
    cy = (44 + pane_bottom) / 2
    if loading:
        ph = rng.uniform(0, math.pi * 2)
        sp = rng.uniform(20, 30)
        for i in range(3):
            rr = (6 + 4 * math.sin(ph + i * 1.1)) * rng.uniform(0.8, 1.2)
            xx = cx - sp + i * sp
            d.ellipse([xx - rr, cy - rr, xx + rr, cy + rr], fill=(200, 200, 220))
    else:
        # Play triangle in the pane + a scrubber/time bar BELOW it: the timeline
        # is the cue that separates "playing/ready" from "buffering". The
        # triangle size and scrubber length vary so the model doesn't assume a
        # fixed triangle-to-pane ratio (real players use small controls).
        tf = rng.uniform(0.5, 1.1)
        d.polygon([(cx - 18 * tf, cy - 24 * tf), (cx - 18 * tf, cy + 24 * tf), (cx + 26 * tf, cy)], fill="white")
        ty = pane_bottom + rng.uniform(18, 34)
        rightx = S - 40 - rng.uniform(0, 90)   # scrubber may span full or partial width
        _rrect(d, [40, ty - 4, rightx, ty + 4], 4, (176, 180, 190) if light else (90, 92, 104))
        kx = 40 + (rightx - 40) * rng.uniform(0.1, 0.6)
        _rrect(d, [40, ty - 4, kx, ty + 4], 4, (239, 90, 90))
        d.ellipse([kx - 7, ty - 7, kx + 7, ty + 7], fill=(239, 90, 90))
        _bars(d, 40, ty + 22, 1, 70, _ink(rng, light), rng)


def _shop(img, d, rng, loading, light):
    if loading:
        g = _placeholder(rng, light)
        _rrect(d, [40, 45, 150, 155], 10, g)
        _bars(d, 165, 55, 4, 60, g, rng)
        _rrect(d, [40, 175, S - 40, 205], 10, g)
    else:
        prod = tuple(int(c) for c in rng.randint(90, 235, 3))
        _rrect(d, [40, 45, 150, 155], 10, prod)
        _bars(d, 165, 60, 2, 70, _ink(rng, light), rng)
        d.text((165, 108), "$" + str(rng.randint(9, 99)), fill=(30, 30, 40) if light else (230, 230, 240))
        _rrect(d, [40, 180, S - 40, 210], 10, (46, 180, 110))


_RENDER = {
    "booking": _booking, "feed": _feed, "dashboard": _dashboard,
    "video": _video, "shop": _shop,
}


def render(site, loading, rng):
    """Return a PIL RGB image (S x S) for the given site and phase.

    The site content is drawn on a transparent overlay, then scaled to a random
    fraction of the frame and placed near center over a full-frame background.
    This mimics real screenshots, where the app's content occupies only a
    centered sub-region of the whole viewport, and teaches the model to be
    scale-invariant instead of assuming content fills the frame.
    """
    light = rng.rand() < 0.5   # background lightness is independent of phase
    img = Image.new("RGB", (S, S))
    _bg(ImageDraw.Draw(img), rng, light)

    overlay = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    _RENDER[site](overlay, od, rng, loading, light)
    if loading and site in ("feed", "shop"):
        _shimmer(overlay, rng)

    sc = rng.uniform(0.5, 0.95)
    ns = max(8, int(S * sc))
    ov = overlay.resize((ns, ns), Image.BILINEAR)
    slack = S - ns
    ox = int(slack / 2 + rng.uniform(-0.18, 0.18) * slack)
    oy = int(slack / 2 + rng.uniform(-0.18, 0.18) * slack)
    base = img.convert("RGBA")
    base.alpha_composite(ov, (max(0, min(slack, ox)), max(0, min(slack, oy))))

    arr = np.asarray(base.convert("RGB")).astype(np.float32)
    arr += rng.normal(0, 3, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    cols = len(SITES)
    sheet = Image.new("RGB", (cols * (S + 8), 2 * (S + 8)), (0, 0, 0))
    for c, site in enumerate(SITES):
        for r, loading in enumerate([True, False]):
            sheet.paste(render(site, loading, rng), (c * (S + 8), r * (S + 8)))
    sheet.save("data/preview.png")
    print("wrote data/preview.png")
