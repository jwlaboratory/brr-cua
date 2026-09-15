#!/usr/bin/env python3
"""
mcp-watcher — a Model Context Protocol server that exposes the tiny page-load
watcher as a tool.

The idea: instead of an agent polling a loading page with expensive screenshot
turns, it calls `wait_for_ready` ONCE. This tool watches the screen in a fast
local loop (the ~40 KB NumPy CNN from ../watcher-model), and BLOCKS until the
page settles — then hands control back with how long it waited. One tool call
replaces the whole poll-or-sleep loop.

Zero third-party MCP deps: hand-rolled JSON-RPC 2.0 over stdio (newline-framed).
Requires only numpy + Pillow + the macOS `screencapture` CLI.

Tools:
  wait_for_ready(timeout_s?, region?, require_loading_first?, poll_interval_s?)
  classify_screen(region?)               # one-shot: loading vs ready
"""

import sys, os, json, time, tempfile, subprocess, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
WM = os.path.join(HERE, "..", "watcher-model")
sys.path.insert(0, WM)

import numpy as np
from PIL import Image
from model import TinyCNN, to_nchw   # the trained CNN

NET = TinyCNN.load(os.path.join(WM, "model.npz"))
IMG = 64

def log(*a):
    print("[watcher]", *a, file=sys.stderr, flush=True)

# ---- perception -------------------------------------------------------------

def grab(region):
    """Screenshot the whole display (or a [x,y,w,h] region) -> PIL RGB."""
    fd, path = tempfile.mkstemp(suffix=".png"); os.close(fd)
    cmd = ["screencapture", "-x"]
    if region and len(region) == 4:
        cmd.append("-R" + ",".join(str(int(v)) for v in region))
    cmd.append(path)
    subprocess.run(cmd, check=True, capture_output=True)
    im = Image.open(path).convert("RGB"); im.load()
    os.unlink(path)
    return im

def p_ready(im):
    w, h = im.size; s = min(w, h)
    im = im.crop(((w-s)//2, (h-s)//2, (w-s)//2+s, (h-s)//2+s)).resize((IMG, IMG))
    x = to_nchw(np.asarray(im, dtype=np.float32)[None] / 255.0)
    return float(NET.predict(x)[0, 1])

# ---- tools ------------------------------------------------------------------

def tool_wait_for_ready(args):
    timeout = float(args.get("timeout_s", 30))
    region = args.get("region")
    edge = bool(args.get("require_loading_first", False))
    interval = float(args.get("poll_interval_s", 0.15))
    t0 = time.time()
    seen_loading = not edge
    streak = 0; n = 0; last = 0.0
    while time.time() - t0 < timeout:
        last = p_ready(grab(region)); n += 1
        if last < 0.5:
            seen_loading = True; streak = 0
        else:
            streak += 1
        if seen_loading and streak >= 2:
            waited = int((time.time() - t0) * 1000)
            return {"ready": True, "waited_ms": waited, "inferences": n,
                    "p_ready": round(last, 3),
                    "note": f"Page settled after {waited/1000:.1f}s ({n} checks). Safe to proceed."}
        time.sleep(interval)
    return {"ready": False, "timed_out": True, "waited_ms": int(timeout*1000),
            "inferences": n, "p_ready": round(last, 3),
            "note": "Still loading at timeout — take a screenshot to inspect."}

def tool_classify_screen(args):
    p = p_ready(grab(args.get("region")))
    return {"state": "ready" if p >= 0.5 else "loading", "p_ready": round(p, 3)}

TOOLS = {
    "wait_for_ready": {
        "fn": tool_wait_for_ready,
        "description": ("Block until the page on screen finishes loading, then return. "
            "Call this INSTEAD of polling a loading page with repeated screenshots: it "
            "runs a tiny vision model in a fast local loop and hands control back the "
            "instant the page is ready (or on timeout). Returns how long it waited."),
        "schema": {"type": "object", "properties": {
            "timeout_s": {"type": "number", "description": "Max seconds to wait (default 30)."},
            "region": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
                       "description": "Optional screen region [x,y,w,h] in points. Default: whole display, center-cropped."},
            "require_loading_first": {"type": "boolean",
                       "description": "If true, only fire after first seeing a loading state (edge trigger). Default false — returns immediately if already ready."},
            "poll_interval_s": {"type": "number", "description": "Seconds between checks (default 0.15)."}
        }}
    },
    "classify_screen": {
        "fn": tool_classify_screen,
        "description": "One-shot check: is the page on screen currently 'loading' or 'ready'? Returns the state and P(ready).",
        "schema": {"type": "object", "properties": {
            "region": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
                       "description": "Optional screen region [x,y,w,h]. Default: whole display."}
        }}
    },
}

# ---- minimal MCP (JSON-RPC 2.0 over stdio, newline-delimited) ---------------

def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n"); sys.stdout.flush()

def reply(id, result):  send({"jsonrpc": "2.0", "id": id, "result": result})
def error(id, code, m): send({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": m}})

def handle(msg):
    method = msg.get("method"); mid = msg.get("id")
    if method == "initialize":
        reply(mid, {"protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "watcher", "version": "0.1.0"}})
    elif method == "notifications/initialized":
        pass
    elif method == "ping":
        reply(mid, {})
    elif method == "tools/list":
        reply(mid, {"tools": [
            {"name": n, "description": t["description"], "inputSchema": t["schema"]}
            for n, t in TOOLS.items()]})
    elif method in ("resources/list", "prompts/list"):
        reply(mid, {"resources": []} if "resources" in method else {"prompts": []})
    elif method == "tools/call":
        name = msg["params"]["name"]; args = msg["params"].get("arguments", {}) or {}
        if name not in TOOLS:
            return error(mid, -32602, f"unknown tool: {name}")
        try:
            out = TOOLS[name]["fn"](args)
            reply(mid, {"content": [{"type": "text", "text": json.dumps(out)}], "isError": False})
        except Exception as e:
            log("tool error:", traceback.format_exc())
            reply(mid, {"content": [{"type": "text", "text": f"error: {e}"}], "isError": True})
    elif mid is not None:
        error(mid, -32601, f"method not found: {method}")

def main():
    log("ready · model loaded ·", ", ".join(TOOLS))
    for line in iter(sys.stdin.readline, ""):
        line = line.strip()
        if not line:
            continue
        try:
            handle(json.loads(line))
        except Exception:
            log("bad message:", traceback.format_exc())

if __name__ == "__main__":
    main()
