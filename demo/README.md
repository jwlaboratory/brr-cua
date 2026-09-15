# demo — the handoff, live in the browser

A self-contained page showing the whole idea in motion: instead of the agent
polling with screenshots, it **hands off to the tiny watcher**, which runs in a
fast loop and wakes the agent the instant the page is ready.

**Everything is real** — the exact weights trained in [`../watcher-model`](../watcher-model)
are exported to JSON and the CNN forward pass is re-implemented in JavaScript, so
the model literally runs in your browser (~60 inferences/second), classifying the
mock page's pixels every frame.

## Files
- `demo.html` — the page (template; `{{WEIGHTS}}` is filled at build time).
- `build.py` — exports `model.npz` weights to JSON and writes `index.html`.
- `index.html` — the built, self-contained page (open it directly, or publish it).

## Rebuild
```bash
cd demo && python3 build.py    # reads ../watcher-model/model.npz -> index.html
```

## What you're seeing
1. **Agent** needs the data on a page that's still loading.
2. Instead of polling, it **hands off** — the watcher arms.
3. The watcher classifies the page canvas every frame (`P(ready)`), holding at
   *loading* (~0%) and counting inferences.
4. The moment the page settles, `P(ready)` jumps to ~100%, the watcher **fires**,
   and the agent is woken — having spent **zero** screenshot-turns waiting.
