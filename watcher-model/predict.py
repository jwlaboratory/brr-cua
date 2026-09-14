"""
predict.py — run the trained watcher on real images (e.g. browser screenshots).

  python3 predict.py <image_or_glob> ...

Center-crops each input to a square (matching the square training renders),
resizes to 32x32, and prints the predicted class + confidence.
"""

import sys
import glob
import numpy as np
from PIL import Image
from model import TinyCNN, to_nchw, CLASSES

IMG = 64


def load(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    im = im.resize((IMG, IMG))
    return np.asarray(im, dtype=np.float32) / 255.0


def main(paths):
    files = []
    for p in paths:
        files.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
    if not files:
        print("no files matched")
        return
    net = TinyCNN.load("model.npz")
    X = to_nchw(np.stack([load(f) for f in files]))
    prob = net.predict(X)
    print(f"{'file':40s}  {'prediction':10s}  confidence")
    print("-" * 66)
    for f, pr in zip(files, prob):
        k = int(pr.argmax())
        name = f if len(f) <= 40 else "…" + f[-39:]
        print(f"{name:40s}  {CLASSES[k]:10s}  {pr[k] * 100:5.1f}%   (loading={pr[0]*100:.1f}, ready={pr[1]*100:.1f})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 predict.py <image_or_glob> ...")
    else:
        main(sys.argv[1:])
