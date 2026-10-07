"""Local page that follows output/events.jsonl as each pipeline step is written."""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Standardized pipeline</title>
<style>
  :root {
    color-scheme: light;
    --ink: #1c1917;
    --muted: #57534e;
    --line: #e7e5e4;
    --bg: #fafaf9;
    --card: #ffffff;
    --model: #166534;
    --model-bg: #dcfce7;
    --human: #9a3412;
    --human-bg: #ffedd5;
    --active: #1d4ed8;
    --active-bg: #dbeafe;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font: 15px/1.45 "Segoe UI", system-ui, sans-serif;
    color: var(--ink);
    background: var(--bg);
  }
  header, main { padding: 20px 24px; }
  header { border-bottom: 1px solid var(--line); background: var(--card); }
  h1 { font-size: 20px; margin: 0 0 4px; }
  p { margin: 0; color: var(--muted); }
  .rail { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
  .stage {
    border: 1px solid var(--line);
    background: var(--card);
    border-radius: 999px;
    padding: 6px 12px;
    color: var(--muted);
  }
  .stage.seen { color: var(--ink); }
  .stage.active { background: var(--active-bg); color: var(--active); border-color: #93c5fd; }
  .counts { display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }
  .count {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 12px 14px;
    min-width: 140px;
  }
  .count b { display: block; font-size: 22px; }
  table { width: 100%; border-collapse: collapse; background: var(--card); }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
  th { font-size: 12px; letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); }
  tr.model td:first-child { box-shadow: inset 3px 0 var(--model); }
  tr.human td:first-child { box-shadow: inset 3px 0 var(--human); }
  .tag {
    display: inline-block;
    border-radius: 999px;
    padding: 2px 8px;
    font-size: 12px;
  }
  .tag.model { background: var(--model-bg); color: var(--model); }
  .tag.human { background: var(--human-bg); color: var(--human); }
  .layout { display: grid; grid-template-columns: 1.4fr 0.8fr; gap: 16px; }
  @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }
  .panel { background: var(--card); border: 1px solid var(--line); border-radius: 12px; overflow: auto; max-height: 70vh; }
  .panel h2 { font-size: 14px; margin: 0; padding: 12px 14px; border-bottom: 1px solid var(--line); }
</style>
</head>
<body>
<header>
  <h1>Standardized pipeline</h1>
  <p id="summary">Waiting for the first step.</p>
  <div class="rail" id="rail"></div>
</header>
<main>
  <div class="counts" id="counts"></div>
  <div class="layout">
    <section class="panel">
      <h2>Decisions</h2>
      <table>
        <thead>
          <tr><th>Step</th><th>Raw</th><th>Proposed</th><th>Confidence</th><th>Route</th></tr>
        </thead>
        <tbody id="decisions"></tbody>
      </table>
    </section>
    <section class="panel">
      <h2>Rows</h2>
      <table>
        <thead><tr><th>Row</th><th>Status</th><th>Waiting on</th></tr></thead>
        <tbody id="rows"></tbody>
      </table>
    </section>
  </div>
</main>
<script>
const STAGES = ["schema", "ingest", "map_column", "standardize_value", "assemble", "export", "human"];
const rail = document.getElementById("rail");
const decisionsBody = document.getElementById("decisions");
const rowsBody = document.getElementById("rows");
const counts = document.getElementById("counts");
const summary = document.getElementById("summary");
const seen = new Set();
let active = "";
const decisions = new Map();
const rows = new Map();
let threshold = 0.8;

STAGES.forEach((stage) => {
  const node = document.createElement("div");
  node.className = "stage";
  node.id = "stage-" + stage;
  node.textContent = stage.replace(/_/g, " ");
  rail.appendChild(node);
});

function paintRail() {
  STAGES.forEach((stage) => {
    const node = document.getElementById("stage-" + stage);
    node.className = "stage" + (seen.has(stage) ? " seen" : "") + (stage === active ? " active" : "");
  });
}

function paintCounts() {
  let model = 0;
  let waiting = 0;
  let people = 0;
  decisions.forEach((item) => {
    if (item.status === "completed_by_model") model += 1;
    else if (item.status === "needs_human") waiting += 1;
    else if (item.status === "completed_by_human") people += 1;
  });
  let ready = 0;
  let held = 0;
  rows.forEach((item) => {
    if (item.status === "ready") ready += 1;
    else held += 1;
  });
  const cards = [
    ["Model completed", model],
    ["Waiting for a person", waiting],
    ["Completed by a person", people],
    ["Rows ready", ready],
    ["Rows held", held],
  ];
  counts.innerHTML = cards.map(([label, value]) =>
    `<div class="count"><b>${value}</b>${label}</div>`
  ).join("");
}

function paintDecisions() {
  const items = Array.from(decisions.values()).reverse();
  decisionsBody.innerHTML = items.map((item) => {
    const kind = item.status === "needs_human" ? "human" : "model";
    const score = Number(item.confidence);
    const shown = Number.isFinite(score) ? score.toFixed(2) : "";
    return `<tr class="${kind}">
      <td>${item.stage || ""}</td>
      <td>${escapeText(item.field)}: ${escapeText(item.raw_value)}</td>
      <td>${escapeText(item.accepted_value || item.proposed_value)}</td>
      <td>${shown}</td>
      <td><span class="tag ${kind}">${escapeText(item.status)}</span></td>
    </tr>`;
  }).join("");
}

function paintRows() {
  const items = Array.from(rows.values());
  rowsBody.innerHTML = items.map((item) => `<tr>
    <td>${escapeText(item.row_id)}</td>
    <td>${escapeText(item.status)}</td>
    <td>${escapeText(item.pending_fields || "")}</td>
  </tr>`).join("");
}

function escapeText(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[ch]));
}

function applyEvent(event) {
  active = event.stage || active;
  if (event.stage) seen.add(event.stage);
  if (event.stage === "schema") {
    threshold = Number(event.threshold);
    summary.textContent = `${event.schema || "dataset"} · threshold ${event.threshold} · ${event.llm || ""}`;
  }
  if (event.stage === "map_column" || event.stage === "standardize_value") {
    decisions.set(event.decision_id, event);
  }
  if (event.stage === "human" && event.decision_id) {
    const current = decisions.get(event.decision_id) || {};
    decisions.set(event.decision_id, Object.assign({}, current, event, {
      stage: event.decision_stage || current.stage || "human",
      status: event.status,
      accepted_value: event.accepted_value,
    }));
  }
  if (event.stage === "assemble" && event.row_id) {
    rows.set(event.row_id, event);
  }
  if (event.stage === "export") {
    summary.textContent = `Export ready ${event.rows_ready}, held ${event.rows_held}. Threshold ${threshold}.`;
  }
  paintRail();
  paintCounts();
  paintDecisions();
  paintRows();
}

const source = new EventSource("/events");
source.onmessage = (message) => {
  applyEvent(JSON.parse(message.data));
};
source.onerror = () => {
  summary.textContent = "Reconnecting to the event stream.";
};
</script>
</body>
</html>
"""


def read_new_events(path: Path, position: int) -> Tuple[List[str], int]:
    """Return complete JSON lines written after ``position``.

    A trailing partial line is left unread so a flush boundary cannot split an event.
    If the file is replaced by a new run, reading starts again at the beginning.
    """
    if not path.exists():
        return [], 0
    size = path.stat().st_size
    if size < position:
        position = 0
    with open(path, "rb") as handle:
        handle.seek(position)
        data = handle.read()
    if not data:
        return [], position
    if not data.endswith(b"\n"):
        cutoff = data.rfind(b"\n")
        if cutoff < 0:
            return [], position
        data = data[: cutoff + 1]
    lines = [line.decode("utf-8") for line in data.splitlines() if line.strip()]
    return lines, position + len(data)


class _DaemonHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


class LiveView:
    def __init__(self, server: ThreadingHTTPServer, thread: threading.Thread) -> None:
        self.server = server
        self.thread = thread

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/"

    def wait_until_stopped(self) -> None:
        try:
            while self.thread.is_alive():
                time.sleep(0.4)
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)


def start_live_view(output_dir: Path, port: int = 8765) -> LiveView:
    """Serve the live page. The event file may not exist yet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"
    handler = _handler_factory(events_path)
    server = None
    last_error: Optional[Exception] = None
    candidates = [0] if port == 0 else list(range(port, port + 20))
    for candidate in candidates:
        try:
            server = _DaemonHTTPServer(("127.0.0.1", candidate), handler)
            break
        except OSError as exc:
            last_error = exc
    if server is None:
        raise RuntimeError(f"Could not bind a live-view port starting at {port}") from last_error
    thread = threading.Thread(target=server.serve_forever, name="pipeline-live-view", daemon=True)
    thread.start()
    return LiveView(server, thread)


def open_live_view(view: LiveView) -> None:
    print(f"Live view: {view.url}", flush=True)
    try:
        webbrowser.open(view.url)
    except Exception:
        return None


def _handler_factory(events_path: Path):
    class LiveHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in {"/", "/index.html"}:
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/events":
                self._stream_events()
                return
            self.send_error(404)

        def _stream_events(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            position = 0
            try:
                while True:
                    lines, position = read_new_events(events_path, position)
                    for line in lines:
                        try:
                            json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        self.wfile.write(b"data: " + line.encode("utf-8") + b"\n\n")
                    self.wfile.flush()
                    time.sleep(0.2)
            except (BrokenPipeError, ConnectionResetError, json.JSONDecodeError, OSError):
                return

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    return LiveHandler
