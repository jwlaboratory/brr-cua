# Agent — computer-use page-load waiting experiment

An investigation into how computer-use agents (Claude in Chrome and similar) handle
**waiting for a page to load**, and whether that behavior is efficient.

- **[`REPORT.md`](./REPORT.md)** — the full write-up: method, measurements, verdict, and fixes
  (including the proposed *tiny always-on watcher model* that interrupts the main model when the page is ready).
- **`booking-canvas.html`** — the test fixture. A ticket-booking flow (booking → 1–4s loading → confirmation)
  rendered entirely to `<canvas>`, so page state is **invisible to the DOM**. This forces an agent onto the
  pure-vision path and exposes the real cost of waiting.

## Run it

```
cd Agent && python3 -m http.server 8778
# open http://localhost:8778/booking-canvas.html
```

Then drive it with a computer-use agent and watch how it detects the loading→confirmed transition:
`read_page` returns nothing on this page; only a screenshot reveals state.

## Headline finding

There is no interrupt/"notify me when ready" primitive — an agent can only blind-sleep or poll, and each
poll is a full ~4s model turn, *longer than the ~2s load it's waiting on*. So for visual readiness, the
current loop is structurally inefficient. See `REPORT.md`.
