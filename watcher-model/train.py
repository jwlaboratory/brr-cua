"""
train.py — train the tiny NumPy CNN watcher on the rendered dataset.

  python3 make_dataset.py      # writes data/dataset.npz
  python3 train.py             # writes model.npz, prints metrics

Reports overall accuracy plus a per-site breakdown so we can see the model
generalizes the "loading vs ready" concept across all 5 website types.
"""

import numpy as np
from model import TinyCNN, to_nchw, softmax_ce
from render import SITES


def main(epochs=14, batch=64, lr=1.5e-3, seed=0):
    d = np.load("data/dataset.npz")
    X, y, sid = d["X"], d["y"], d["sites"]
    Xn = to_nchw(X)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y))
    Xn, y, sid = Xn[perm], y[perm], sid[perm]
    ntr = int(0.85 * len(y))
    Xtr, ytr = Xn[:ntr], y[:ntr]
    Xte, yte, ste = Xn[ntr:], y[ntr:], sid[ntr:]

    net = TinyCNN(seed=seed)
    nb = int(np.ceil(ntr / batch))
    for ep in range(epochs):
        order = rng.permutation(ntr)
        tot = 0.0
        for b in range(nb):
            idx = order[b * batch:(b + 1) * batch]
            xb, yb = Xtr[idx], ytr[idx]
            logits, cache = net.forward(xb)
            loss, dlogits, _ = softmax_ce(logits, yb)
            net.step(net.backward(dlogits, cache), lr=lr)
            tot += loss * len(idx)
        # eval
        pr = net.predict(Xte).argmax(1)
        acc = (pr == yte).mean()
        print(f"epoch {ep + 1:2d}/{epochs}  train_loss={tot / ntr:.4f}  test_acc={acc:.4f}")

    # final metrics
    prob = net.predict(Xte)
    pred = prob.argmax(1)
    acc = (pred == yte).mean()
    print(f"\nFINAL test accuracy: {acc:.4f}  ({int((pred==yte).sum())}/{len(yte)})")

    # confusion matrix
    cm = np.zeros((2, 2), int)
    for t, p in zip(yte, pred):
        cm[t, p] += 1
    print("confusion [rows=true loading/ready, cols=pred]:")
    print(f"  loading: {cm[0]}")
    print(f"  ready  : {cm[1]}")

    # per-site accuracy
    print("per-site test accuracy:")
    for i, s in enumerate(SITES):
        m = ste == i
        if m.any():
            print(f"  {s:10s}: {(pred[m] == yte[m]).mean():.3f}  (n={int(m.sum())})")

    net.save("model.npz")
    print("\nsaved model.npz")


if __name__ == "__main__":
    main()
