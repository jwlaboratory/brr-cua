"""
make_dataset.py — build the training set for the watcher model.

Two sources are combined:
  1. SYNTHETIC  — render.py draws unlimited randomized variants of the 5 sites.
                  Gives volume and pose/scale/color variety.
  2. REAL       — actual browser screenshots captured from the running sites
                  (data/realtrain/*.png), each expanded with heavy augmentation
                  (random square crop/scale, jitter, brightness/contrast, noise,
                  small rotation). This is what closes the synthetic->real gap.

The held-out real frames in data/real/ are NOT used here — they are the test set.

Writes data/dataset.npz with X (N,48,48,3) float32, y (0=loading,1=ready), sites.
Usage: python3 make_dataset.py [synthetic_per_class_per_site] [aug_per_real_frame]
"""

import sys
import glob
import os
import numpy as np
from PIL import Image
from render import render, SITES

IMG = 64  # model input size


def build_synth(per, rng):
    X, y, sid = [], [], []
    for si, site in enumerate(SITES):
        for loading in (True, False):
            for _ in range(per):
                im = render(site, loading, rng).resize((IMG, IMG))
                X.append(np.asarray(im, dtype=np.float32) / 255.0)
                y.append(0 if loading else 1)
                sid.append(si)
    return X, y, sid


def aug_real_frame(im, n, rng):
    """Yield n augmented IMGxIMG views of one large real screenshot.

    The content is zoomed BOTH in (crop) and out (pad with the page's own
    background colour), so the model sees the app occupying anywhere from ~50%
    to >100% of the frame — real screenshots vary a lot in how much margin
    surrounds the content depending on window size.
    """
    W, H = im.size
    m = min(W, H)
    base = im.crop(((W - m) // 2, (H - m) // 2, (W - m) // 2 + m, (H - m) // 2 + m))
    corner = np.asarray(im)[:8, :8].reshape(-1, 3).mean(0)   # background colour
    out = []
    for _ in range(n):
        z = rng.uniform(0.5, 1.2)
        cs = max(8, int(IMG * z))
        b = base.resize((cs, cs))
        bg = tuple(int(np.clip(c + rng.normal(0, 4), 0, 255)) for c in corner)
        if cs <= IMG:                       # zoom out: content smaller, pad with bg
            canvas = Image.new("RGB", (IMG, IMG), bg)
            canvas.paste(b, (rng.randint(0, IMG - cs + 1), rng.randint(0, IMG - cs + 1)))
            c = canvas
        else:                               # zoom in: content larger, crop
            ox, oy = rng.randint(0, cs - IMG + 1), rng.randint(0, cs - IMG + 1)
            c = b.crop((ox, oy, ox + IMG, oy + IMG))
        if rng.rand() < 0.4:
            c = c.rotate(rng.uniform(-6, 6), fillcolor=bg)
        a = np.asarray(c, dtype=np.float32)
        a *= rng.uniform(0.8, 1.2)                     # brightness
        a = (a - 128) * rng.uniform(0.85, 1.15) + 128  # contrast
        a += rng.normal(0, 4, a.shape)                 # noise
        out.append(np.clip(a, 0, 255) / 255.0)
    return out


def build_real(n, rng):
    X, y, sid = [], [], []
    files = sorted(glob.glob("data/realtrain/*.png"))
    for f in files:
        name = os.path.basename(f)
        label = 1 if "loaded" in name else 0
        site = next((i for i, s in enumerate(SITES) if s in name), -1)
        im = Image.open(f).convert("RGB")
        for a in aug_real_frame(im, n, rng):
            X.append(a); y.append(label); sid.append(site)
    return X, y, sid, len(files)


if __name__ == "__main__":
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    naug = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    rng = np.random.RandomState(1)

    Xs, ys, ss = build_synth(per, rng)
    Xr, yr, sr, nfiles = build_real(naug, rng)

    X = np.stack(Xs + Xr)
    y = np.array(ys + yr, dtype=np.int64)
    sid = np.array(ss + sr, dtype=np.int64)
    p = rng.permutation(len(y))
    X, y, sid = X[p], y[p], sid[p]

    np.savez_compressed("data/dataset.npz", X=X, y=y, sites=sid)
    print(f"dataset: {len(y)} samples  (synthetic={len(ys)}, real={len(yr)} from {nfiles} frames x {naug} aug)")
    print(f"  loading={int((y==0).sum())}  ready={int((y==1).sum())}  shape={X.shape}")
