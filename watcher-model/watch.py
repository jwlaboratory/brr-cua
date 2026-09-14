"""
watch.py — the "watcher" loop: a tiny model that watches the screen and fires
the moment the page is READY, so a big/expensive agent can wait for an interrupt
instead of burning turns polling. (This is the concrete realization of the
proposal in ../Agent/REPORT.md.)

It grabs a screen region every ~200ms with macOS `screencapture`, runs the
NumPy CNN (a few ms), and prints READY when it flips loading -> ready.

Usage:
  python3 watch.py                 # watch the full main display
  python3 watch.py X Y W H         # watch a screen region (points)
  python3 watch.py --image f.png   # one-shot classify an image (no screen grab)

Requires only numpy + PIL + the macOS `screencapture` CLI.
"""

import sys
import time
import subprocess
import tempfile
import os
import numpy as np
from PIL import Image
from model import TinyCNN, to_nchw, CLASSES

IMG = 64
POLL_S = 0.2
STABLE_N = 2   # require this many consecutive "ready" reads before firing


def preprocess(im):
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((IMG, IMG))
    return to_nchw(np.asarray(im, dtype=np.float32)[None] / 255.0)


def grab(region):
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    cmd = ["screencapture", "-x"]
    if region:
        cmd += ["-R" + ",".join(str(int(v)) for v in region)]
    cmd += [path]
    subprocess.run(cmd, check=True)
    im = Image.open(path).convert("RGB")
    os.unlink(path)
    return im


def main():
    net = TinyCNN.load("model.npz")

    if len(sys.argv) >= 3 and sys.argv[1] == "--image":
        p = net.predict(preprocess(Image.open(sys.argv[2]).convert("RGB")))[0]
        k = int(p.argmax())
        print(f"{CLASSES[k]}  (loading={p[0]*100:.1f}%, ready={p[1]*100:.1f}%)")
        return

    region = [float(a) for a in sys.argv[1:5]] if len(sys.argv) >= 5 else None
    print("watching… (Ctrl-C to stop)")
    t0 = time.time()
    ready_streak = 0
    while True:
        p = net.predict(preprocess(grab(region)))[0]
        k = int(p.argmax())
        elapsed = time.time() - t0
        print(f"  [{elapsed:5.1f}s] {CLASSES[k]:8s} ready={p[1]*100:5.1f}%", flush=True)
        ready_streak = ready_streak + 1 if k == 1 else 0
        if ready_streak >= STABLE_N:
            print(f"\n>>> READY after {elapsed:.1f}s — waking the main agent.")
            return
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
