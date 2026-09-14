# How computer-use agents handle "wait for the page to load" — and whether it's inefficient

**Date:** 2026-09-14
**Setup:** Claude (Opus 4.8) driving Chrome via the `claude-in-chrome` computer-use tools, against a purpose-built canvas fixture (`booking-canvas.html`) that hides all UI state from the DOM.

## TL;DR

Yes — for anything whose "ready" signal is **visual**, the current computer-use loop is structurally inefficient. Not because the model picks a bad strategy, but because the toolset gives it only two ways to wait, and both are bad:

1. **Blind sleep** (`computer` `wait` action) — guess a duration, hope the page is ready.
2. **Poll** — re-perceive (screenshot / read_page) in a loop until it looks ready.

There is **no interrupt / event / "notify me when ready" primitive**. The agent cannot subscribe to readiness; it can only guess or busy-wait. And each poll costs a *full agent turn*, which in this session averaged **~4 seconds** and thousands of vision tokens — *longer than the thing being waited on*.

## Why a canvas fixture

If a page signals readiness in the DOM (text appears, a spinner node is removed, an `aria-busy` flips), an agent skips vision entirely and polls with `read_page` — cheap, exact, ~instant page-side. That path is fine, so it's not where the inefficiency lives.

To study the hard case, `booking-canvas.html` paints the **entire** flow (booking → loading spinner → confirmation) onto a `<canvas>`. State lives only in a JS closure; nothing is mirrored to the DOM. This forces the pure-computer-use path: to know whether loading finished, the agent *must look at pixels*.

Confirmed empirically — `read_page` (accessibility tree, the cheap path) on this page returns nothing but:

```
Viewport: 1200x728
```

No text, no elements, no state. Screenshot is the only perception channel.

## What the toolset actually offers for waiting

| Mechanism | Tool | Nature | Cost per check |
|---|---|---|---|
| Blind sleep | `computer` action `wait` (≤ 10s) | Fire-and-forget, no feedback | 1 turn |
| Visual poll | `computer` action `screenshot` | Full raster + vision tokens | 1 turn + image tokens |
| DOM poll | `read_page` / `javascript_tool` | Cheap, exact — **but empty on canvas** | 1 turn |
| Event / interrupt | *(none exists)* | — | — |
| Batched sequence | `browser_batch` | Fixed list, stops on first error, coords frozen to pre-batch screenshot; **no conditional/await-state** | 1 turn |

The absence of the "Event / interrupt" row is the whole story.

## Measurements

**Ground-truth loading delay** (page's real 1–4s randomized processing, measured page-side over 8 real buy cycles):

```
trials (ms): 2264, 1635, 1210, 3580, 1700, 1230, 3228, 1862
min 1210 · avg 2089 · max 3580
```

**Agent observe-cycle latency** — wall-clock between two *consecutive, trivial* tool calls (a one-line JS eval, the cheapest possible perception):

```
~4212 ms
```

This is dominated by model inference, not the tool. It is the floor on how fast the agent can poll anything.

**Live pure-vision runs:** I did the real booking flow twice, perceiving only via screenshots. In **both** runs I clicked "Buy" and the very next screenshot already showed the *confirmation* screen — I **never once captured the loading spinner**. The 1–4s load completed entirely *inside a single observe cycle*.

**Fixture sanity-check (all three states render):** because the load window fits inside one observe cycle, the loader is invisible to the agent under normal timing. To confirm the app itself is correct, the loading state was pinned open (deadline pushed 60s out) and screenshotted: the animated spinner + "Processing your payment…" render as designed. Booking and confirmation states were also verified live. So the fixture is behaving correctly — the missing spinner in the live runs is the *finding*, not a bug.

## Interpretation: the loop is coarser than what it observes

Put the two numbers side by side:

- Thing being waited on: **~2.1s average**, up to 3.6s.
- One agent observe→decide→act cycle: **~4.2s**.

The agent's clock ticks slower than the event it's watching. Consequences:

1. **Tight polling is impossible.** The minimum poll interval is one full LLM turn (~seconds). You cannot "check every 100ms."
2. **Transient states are missed.** A spinner shorter than a turn is invisible to the agent (exactly what happened — 0/2 spinners seen).
3. **Every poll is a full turn** — model inference + (for vision) a whole screenshot's worth of image tokens. Polling a slow, variable load N times costs N expensive turns.
4. **Blind sleep is a gamble.** Too short → the agent acts on a stale/loading screen (clicks the wrong thing, misreads state). Too long → wasted wall-clock on every run, sized for the worst case.
5. **No backpressure.** The environment knows the instant it's ready; the agent has no way to receive that. All the timing information is on the wrong side of the boundary.

## So — is it inefficient?

**Yes, structurally, for visual-readiness — and the cost scales the wrong way.**

- **Fast/instant loads:** the agent still burns a full turn "checking," or a blind `wait` it didn't need. Overhead is a fixed tax per navigation.
- **Slow/variable loads (the realistic case):** the agent either over-sleeps (waste on every run) or polls — and each poll is a costly, coarse turn that may still miss the transition.
- **DOM-visible readiness:** *not* meaningfully inefficient — `read_page` polling is cheap and exact. The problem is specifically **canvas / `<video>` / WebGL / image-diff / pixel-only** readiness, which is a growing share of real apps (maps, editors, games, charts, video).

The inefficiency is **not** a bad model policy. Given these tools, "screenshot, and if it still looks like it's loading, screenshot again" is the *correct* strategy. The ceiling is set by the interface: **no readiness event + a multi-second, token-heavy perception step.**

## What would fix it (roughly in order of leverage)

### 0. A tiny always-on "watcher" model that interrupts the main model when the page is ready *(proposed approach)*

The idea: run a **small, cheap vision model in a tight loop on the screen** (or on a live screenshot/video stream of the tab), doing exactly one job — decide "still loading?" vs "ready / changed". When it detects the page has settled, it **fires an interrupt to the main model**, which was parked and spending nothing in the meantime.

This directly supplies the primitive the toolset is missing (the empty "Event / interrupt" row above). Why it fits the measured problem:

- **It decouples the poll clock from the main model's inference clock.** The bottleneck we measured is the ~4.2s main-model turn. A tiny model can tick far faster (tens of ms) and *cheaply*, so effective poll resolution goes from "seconds" to "sub-second" — fine-grained enough to catch a 1–4s load and even the transient spinner the big model kept missing.
- **The expensive model spends ~0 turns waiting.** Instead of N costly screenshot-turns (image tokens + Opus inference each), the big model issues one action, yields, and is woken once. Cost stops scaling with wait length.
- **It's the visual analog of `network-idle`,** but works on the hostile case this fixture targets: canvas / `<video>` / WebGL, where DOM/network signals say nothing. The watcher judges *pixels*, which is the only channel that carries state here.

Design notes / open questions worth prototyping:
- **What the watcher emits:** a cheap boolean/edge ("changed since baseline", "no motion for K frames", "spinner gone"), not a full description — keep its per-tick cost near zero.
- **Baseline + debounce:** snapshot the frame at action time; fire only after the scene both *changes* and then *stabilizes* for K frames, to avoid firing on the spinner animation itself.
- **Timeout / fallback:** if no settle within a budget, wake the main model anyway so it can decide (retry, error, escalate).
- **Where it runs:** ideally in-process next to the browser (frame access, no MCP round-trip per tick). A local small multimodal model or even a non-ML pixel-diff/perceptual-hash pass could cover most "did it stop changing?" cases; use the small model only to classify ambiguous frames.
- **Generalizes beyond load:** the same watcher handles "video finished", "spinner appeared", "modal popped", "content scrolled into view" — any *visual* edge the main model currently has to burn a turn to check for.

This is essentially rows 1–4 below implemented as an out-of-band service rather than waiting on the interface to grow the feature.

> **Built as a proof of concept** in [`../watcher-model`](../watcher-model): a
> ~40 KB from-scratch NumPy CNN that classifies a screenshot as `loading` vs
> `ready` across 5 site types, scoring **10/10 on held-out real browser
> screenshots** and running in a few ms per frame. `watcher-model/watch.py` is
> the loop — it polls the screen and emits a single "READY" edge, the interrupt
> the computer-use toolset is missing.

### Interface-level fixes (if building the primitives directly)

1. **A readiness/settle signal from the environment.** Let the tool block until "page settled" and return control on that edge — e.g. `network-idle`, `requestAnimationFrame` quiescence, no DOM/canvas mutation for K ms, or a page-declared `window.__ready` flag. Turn busy-wait into an interrupt. This is the single biggest win.
2. **A cheap `wait_for(predicate)` primitive** that runs the poll loop *inside the browser* (page-side, ms resolution) and returns once satisfied or timed out — so the agent spends **one** turn, not N.
3. **Conditional steps in `browser_batch`** (await-selector / await-stable-pixels between actions) so a click→wait→verify sequence is one round-trip instead of three.
4. **Cheap perceptual deltas** — a "did the visible region change since screenshot X?" check that returns a bool/hash instead of a full image, so polling doesn't pay vision-token cost every tick.
5. **App-side cooperation** (when you control the page): expose readiness in the DOM/`aria-busy`/a global flag even for canvas apps, so agents can use the cheap path. The canvas fixture here is the deliberately *hostile* case that removes this.

## Files

- `booking-canvas.html` — the hostile fixture: full booking flow rendered to `<canvas>`, state hidden from the DOM, with a randomized 1–4s "processing" delay.
- `REPORT.md` — this document.

## Reproduce

```
cd Agent && python3 -m http.server 8778
# open http://localhost:8778/booking-canvas.html
# Drive it with a computer-use agent: read_page (returns nothing) vs screenshot (shows state).
# Click "Buy Ticket"; observe how the agent detects the loading→confirmed transition.
```
