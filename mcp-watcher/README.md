# mcp-watcher — the page-load watcher as an MCP tool

Exposes the tiny [`../watcher-model`](../watcher-model) CNN over the Model
Context Protocol, so an agent (Claude Code, or any MCP client) can **wait for a
loading page in one tool call instead of polling with screenshots**.

The `wait_for_ready` tool **blocks** — it watches the screen in a fast local loop
and returns the instant the page settles (or on timeout). That's the whole point:
Claude calls it when it's about to wait, and gets control back when the page is
done.

## Tools
| tool | what it does |
|---|---|
| `wait_for_ready(timeout_s?, region?, require_loading_first?, poll_interval_s?)` | Block until the page is ready; return `{ready, waited_ms, inferences, p_ready}`. |
| `classify_screen(region?)` | One-shot: `{state: "loading"\|"ready", p_ready}`. |

## Requirements
- **numpy + Pillow** (already used by `watcher-model`) — no MCP SDK needed; the
  server hand-rolls JSON-RPC over stdio.
- macOS `screencapture` (built in). The watcher reads the **visible screen**, so
  the page you're waiting on must be the frontmost window (it is when an agent is
  driving the browser). Use `region: [x, y, w, h]` to target a specific area.

## Install (Claude Code)
This repo already ships a project-scoped registration in [`../.mcp.json`](../.mcp.json):

```json
{ "mcpServers": { "watcher": {
  "command": "/opt/homebrew/opt/python@3.13/bin/python3.13",
  "args": ["/ABS/PATH/brr-cua/mcp-watcher/server.py"] } } }
```

1. Restart Claude Code in this project (MCP servers load at startup).
2. Approve the `watcher` server when prompted (project-scoped servers ask once).
3. The tools appear as `mcp__watcher__wait_for_ready` / `mcp__watcher__classify_screen`.

`../CLAUDE.md` tells Claude to prefer `wait_for_ready` over screenshot-polling.

Or register from any project with:
```bash
claude mcp add watcher -- /opt/homebrew/opt/python@3.13/bin/python3.13 \
  /ABS/PATH/brr-cua/mcp-watcher/server.py
```

## Test it standalone
```bash
printf '%s\n' \
 '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
 '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"classify_screen","arguments":{}}}' \
 | python3 server.py
```

## Notes
- The model path is validated correct on real page frames (loaded → P(ready)≈1,
  loading → P(ready)≈0). Accuracy in the wild depends only on the watcher
  capturing the right pixels — keep the target page visible, or pass `region`.
- A Claude Code **hook** (`PostToolUse` on navigate) can enforce this
  automatically instead of relying on the model to call the tool — ask if you
  want that variant.
