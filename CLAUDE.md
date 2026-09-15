# brr-cua

## Waiting for a page to load — call the watcher, don't poll

When you're driving a browser and a page is loading — right after a navigation
or a click that kicks off a load, or any time you would otherwise take repeated
screenshots just to check "is it done yet?" — **do not poll with screenshots.**
Call the `watcher` MCP tool instead:

- **`mcp__watcher__wait_for_ready`** — blocks until the page on screen settles,
  then hands control back and reports how long it waited. Right after a click
  that starts a load, pass `require_loading_first: true` so it waits for the
  loading→ready edge rather than returning on the pre-load state.
- **`mcp__watcher__classify_screen`** — one-shot "loading vs ready" check.

Why: a screenshot-poll costs a full model turn (seconds + vision tokens) each
time, and the page often finishes between checks. The watcher runs a ~40 KB
local vision model in a fast loop and collapses the whole wait into one tool
call.

It watches the **visible screen**, so the page must be the frontmost window
(it is when you're the one driving the browser). Pass `region: [x, y, w, h]`
(points) to target a specific area.
