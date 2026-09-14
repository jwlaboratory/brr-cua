"""
model.py — a tiny CNN implemented from scratch in NumPy (no torch/sklearn).

Architecture (input 3x48x48):
  conv1 3->8, 3x3, pad1   -> ReLU -> maxpool 2   (8x24x24)
  conv2 8->16, 3x3, pad1  -> ReLU -> maxpool 2   (16x12x12)
  flatten (2304) -> fc -> 2 logits -> softmax

~6k parameters. Trains in seconds on CPU. Conv uses an im2col + single
BLAS matmul, so it's fast despite being pure NumPy.
"""

import numpy as np

CLASSES = ["loading", "ready"]


# ---- conv via im2col (stride 1, pad 1, 3x3) --------------------------------

def _im2col(Xp, kh, kw, oh, ow):
    # Xp: (N,C,Hp,Wp) -> cols: (C*kh*kw, N*oh*ow)
    N, C, Hp, Wp = Xp.shape
    patches = np.empty((N, C, kh, kw, oh, ow), dtype=Xp.dtype)
    for i in range(kh):
        for j in range(kw):
            patches[:, :, i, j] = Xp[:, :, i:i + oh, j:j + ow]
    return patches.transpose(1, 2, 3, 0, 4, 5).reshape(C * kh * kw, N * oh * ow)


def conv_forward(X, W, b):
    # X:(N,C,H,W) W:(F,C,kh,kw) pad1 stride1
    N, C, H, Wd = X.shape
    F, _, kh, kw = W.shape
    Xp = np.pad(X, ((0, 0), (0, 0), (1, 1), (1, 1)))
    oh, ow = H, Wd
    cols = _im2col(Xp, kh, kw, oh, ow)              # (C*kh*kw, N*oh*ow)
    Wc = W.reshape(F, -1)                            # (F, C*kh*kw)
    out = (Wc @ cols + b[:, None]).reshape(F, N, oh, ow).transpose(1, 0, 2, 3)
    return out, (X.shape, cols, Wc, kh, kw)


def conv_backward(dout, cache, W):
    Xshape, cols, Wc, kh, kw = cache
    N, C, H, Wd = Xshape
    F = W.shape[0]
    oh, ow = H, Wd
    dcol_out = dout.transpose(1, 0, 2, 3).reshape(F, -1)   # (F, N*oh*ow)
    dWc = dcol_out @ cols.T                                # (F, C*kh*kw)
    db = dcol_out.sum(axis=1)
    dcols = Wc.T @ dcol_out                                # (C*kh*kw, N*oh*ow)
    dpatches = dcols.reshape(C, kh, kw, N, oh, ow).transpose(3, 0, 1, 2, 4, 5)
    dXp = np.zeros((N, C, H + 2, Wd + 2), dtype=dout.dtype)
    for i in range(kh):
        for j in range(kw):
            dXp[:, :, i:i + oh, j:j + ow] += dpatches[:, :, i, j]
    dX = dXp[:, :, 1:-1, 1:-1]
    return dX, dWc.reshape(W.shape), db


# ---- other layers ----------------------------------------------------------

def relu_forward(x):
    return np.maximum(0, x), x


def relu_backward(dout, cache):
    return dout * (cache > 0)


def maxpool_forward(x):
    N, C, H, W = x.shape
    xr = x.reshape(N, C, H // 2, 2, W // 2, 2)
    out = xr.max(axis=(3, 5))
    return out, (x.shape, xr, out)


def maxpool_backward(dout, cache):
    xshape, xr, out = cache
    N, C, H, W = xshape
    mask = (xr == out[:, :, :, None, :, None])
    counts = mask.sum(axis=(3, 5), keepdims=True)
    dxr = mask * (dout[:, :, :, None, :, None] / counts)
    return dxr.reshape(xshape)


def softmax_ce(logits, y):
    z = logits - logits.max(axis=1, keepdims=True)
    ez = np.exp(z)
    p = ez / ez.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    loss = -np.log(p[np.arange(n), y] + 1e-9).mean()
    dlogits = p.copy()
    dlogits[np.arange(n), y] -= 1
    dlogits /= n
    return loss, dlogits, p


# ---- model -----------------------------------------------------------------

class TinyCNN:
    def __init__(self, seed=0):
        r = np.random.RandomState(seed)
        def he(shape, fan_in):
            return (r.randn(*shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)
        self.p = {
            "W1": he((8, 3, 3, 3), 3 * 9), "b1": np.zeros(8, np.float32),
            "W2": he((16, 8, 3, 3), 8 * 9), "b2": np.zeros(16, np.float32),
            "W3": he((16 * 16 * 16, 2), 16 * 16 * 16), "b3": np.zeros(2, np.float32),
        }
        self._init_adam()

    def _init_adam(self):
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, X):
        c1, cc1 = conv_forward(X, self.p["W1"], self.p["b1"])
        r1, cr1 = relu_forward(c1)
        p1, cp1 = maxpool_forward(r1)
        c2, cc2 = conv_forward(p1, self.p["W2"], self.p["b2"])
        r2, cr2 = relu_forward(c2)
        p2, cp2 = maxpool_forward(r2)
        flat = p2.reshape(p2.shape[0], -1)
        logits = flat @ self.p["W3"] + self.p["b3"]
        cache = (cc1, cr1, cp1, cc2, cr2, cp2, p2.shape, flat)
        return logits, cache

    def backward(self, dlogits, cache):
        cc1, cr1, cp1, cc2, cr2, cp2, p2shape, flat = cache
        g = {}
        g["W3"] = flat.T @ dlogits
        g["b3"] = dlogits.sum(axis=0)
        dflat = dlogits @ self.p["W3"].T
        dp2 = dflat.reshape(p2shape)
        dr2 = maxpool_backward(dp2, cp2)
        dc2 = relu_backward(dr2, cr2)
        dp1, g["W2"], g["b2"] = conv_backward(dc2, cc2, self.p["W2"])
        dr1 = maxpool_backward(dp1, cp1)
        dc1 = relu_backward(dr1, cr1)
        _, g["W1"], g["b1"] = conv_backward(dc1, cc1, self.p["W1"])
        return g

    def step(self, g, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * (g[k] ** 2)
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + eps)

    def predict(self, X):
        logits, _ = self.forward(X)
        z = logits - logits.max(axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / ez.sum(axis=1, keepdims=True)

    def save(self, path):
        np.savez(path, **self.p)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        obj = cls()
        for k in obj.p:
            obj.p[k] = d[k]
        return obj


def to_nchw(X):
    """(N,H,W,3) [0,1] -> (N,3,H,W) float32."""
    return np.ascontiguousarray(X.transpose(0, 3, 1, 2)).astype(np.float32)
