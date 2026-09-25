"""Read-only browser for the review order (v0).

This version deliberately has no buttons. Its only job is to answer one question that no
measurement can: *are the piles at the top of the order actually people?* The ranking score
is a judgement -- size, readability, coherence -- and the gold set covers 1,499 faces out of
63,878, so nothing in the labelled data can grade it. A person looking at the first fifty
piles can, in about half an hour.

Nothing here writes. If the order turns out to be wrong, the score changes and no work is
lost, because no work has been built on top of it yet.
"""

from __future__ import annotations

import json
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from faceindex import review, store

PAGE_SIZE = 20
FACES_PER_PILE = 24


class ReviewStore:
    """Thin read-only accessor. One connection per call: handlers run on many threads."""

    def __init__(self, db_path: Path, run_id: str | None = None) -> None:
        self.db_path = db_path
        with self._open() as conn:
            chosen = None
            if run_id:
                found = [r for r in review.runs(conn) if r["run_id"] == run_id]
                if not found:
                    available = [str(r["run_id"]) for r in review.runs(conn)]
                    raise ValueError(f"No review run {run_id!r}. Available: {available or 'none'}")
                chosen = found[0]
            else:
                chosen = review.latest_run(conn)
            if chosen is None:
                raise ValueError(
                    "No review index has been built. Run scripts/build_review_index.py first."
                )
            self.run = chosen
            self.run_id = str(chosen["run_id"])
            self.n_people_known = len(review.known_people(conn))

    def _open(self) -> sqlite3.Connection:
        return store.connect(self.db_path, read_only=True)

    def summary(self) -> dict[str, Any]:
        return {
            "run": dict(self.run),
            "page_size": PAGE_SIZE,
            "people_already_named": self.n_people_known,
        }

    def page(self, offset: int, limit: int) -> dict[str, Any]:
        with self._open() as conn:
            piles = review.list_piles(conn, self.run_id, offset=offset, limit=limit)
            for rank, pile in enumerate(piles, start=offset + 1):
                pile["rank"] = rank
                pile["faces"] = review.pile_faces(
                    conn, self.run_id, int(pile["pile_id"]), limit=FACES_PER_PILE
                )
        return {"offset": offset, "piles": piles, "total": int(self.run["n_piles"])}

    def crop(self, face_id: int, *, context: bool) -> Path | None:
        with self._open() as conn:
            return review.crop_path(conn, face_id, context=context)


def make_handler(data: ReviewStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "faceindex-review"

        def log_message(self, fmt: str, *args: Any) -> None:
            # Never log file paths: they are photo paths (PLAN.md section 3, privacy).
            return

        def _send(self, status: int, body: bytes, content_type: str, cache: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, status: int = 200) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json", "no-store")

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            route = parsed.path
            query = parse_qs(parsed.query)
            try:
                if route == "/":
                    self._send(200, INDEX_HTML.encode(), "text/html; charset=utf-8", "no-store")
                elif route == "/api/summary":
                    self._json(data.summary())
                elif route == "/api/piles":
                    self._json(data.page(_int(query, "offset", 0), _int(query, "limit", PAGE_SIZE)))
                elif route.startswith("/crop/"):
                    self._serve_crop(route, query)
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as exc:  # a UI bug must not end the session
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

        def _serve_crop(self, route: str, query: dict[str, list[str]]) -> None:
            try:
                face_id = int(route.rsplit("/", 1)[1])
            except ValueError:
                self._json({"error": "bad face id"}, 400)
                return
            path = data.crop(face_id, context=query.get("kind", ["aligned"])[0] == "context")
            if path is None or not path.exists():
                self._json({"error": "crop missing"}, 404)
                return
            # Crops never change once written, so let the browser keep them. Paging through
            # twenty piles of twenty-four faces is 480 images; re-fetching them all on every
            # back-and-forth would make the order impossible to judge.
            self._send(200, path.read_bytes(), "image/jpeg", "private, max-age=86400")

    return Handler


def _int(query: dict[str, list[str]], key: str, default: int) -> int:
    raw = query.get(key, [""])[0]
    try:
        return max(int(raw), 0)
    except (TypeError, ValueError):
        return default


def serve(
    db_path: Path, *, host: str = "127.0.0.1", port: int = 8766, run_id: str | None = None
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(ReviewStore(db_path, run_id)))


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Review order</title>
<style>
  :root { --bg:#14161a; --card:#1d2027; --line:#2b2f38; --text:#e8eaed; --dim:#9aa1ad;
          --good:#5fd08a; --warn:#e8c468; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  header { position:sticky; top:0; z-index:5; background:var(--bg);
           border-bottom:1px solid var(--line); padding:12px 20px; }
  h1 { font-size:15px; margin:0 0 4px; font-weight:600; }
  .meta { color:var(--dim); font-size:12px; }
  .bar { display:flex; gap:8px; align-items:center; margin-top:10px; flex-wrap:wrap; }
  button, input { background:var(--card); color:var(--text); border:1px solid var(--line);
                  border-radius:6px; padding:5px 11px; font-size:13px; cursor:pointer; }
  button:hover:not(:disabled) { border-color:#4a5162; }
  button:disabled { opacity:.35; cursor:default; }
  input { width:80px; cursor:text; }
  kbd { background:var(--card); border:1px solid var(--line); border-radius:4px;
        padding:1px 5px; font-size:11px; color:var(--dim); }
  main { padding:16px 20px 60px; }
  .pile { background:var(--card); border:1px solid var(--line); border-radius:8px;
          padding:12px 14px; margin-bottom:14px; }
  .phead { display:flex; gap:14px; align-items:baseline; flex-wrap:wrap; margin-bottom:10px; }
  .rank { font-size:16px; font-weight:600; min-width:44px; }
  .stat { color:var(--dim); font-size:12px; }
  .stat b { color:var(--text); font-weight:600; }
  .faces { display:flex; flex-wrap:wrap; gap:6px; }
  .faces img { width:78px; height:78px; object-fit:cover; border-radius:5px;
               background:#000; border:1px solid var(--line); }
  .more { color:var(--dim); font-size:12px; align-self:center; padding-left:6px; }
  #status { padding:20px; color:var(--dim); }
  .err { color:#ff8b8b; }
</style>
</head>
<body>
<header>
  <h1>Review order — read only</h1>
  <div class="meta" id="meta">loading…</div>
  <div class="bar">
    <button id="first">&laquo; best</button>
    <button id="prev">&lsaquo; prev</button>
    <span class="stat" id="where"></span>
    <button id="next">next &rsaquo;</button>
    <button id="last">worst &raquo;</button>
    <input id="jump" type="number" min="1" placeholder="rank">
    <button id="go">go</button>
    <button id="kind">show wider crop</button>
    <span class="stat"><kbd>n</kbd> next <kbd>p</kbd> prev <kbd>t</kbd> crop</span>
  </div>
</header>
<main><div id="status">loading…</div><div id="list"></div></main>
<script>
let offset = 0, pageSize = 20, total = 0, context = false, busy = false;

const $ = (id) => document.getElementById(id);

async function boot() {
  try {
    const s = await (await fetch("/api/summary")).json();
    if (s.error) throw new Error(s.error);
    pageSize = s.page_size;
    total = s.run.n_piles;
    $("meta").textContent =
      `${s.run.model} · ${s.run.algorithm} @ ${s.run.threshold} · ` +
      `${s.run.n_piles.toLocaleString()} piles of 2+ faces · ` +
      `${s.run.n_lone.toLocaleString()} lone faces set aside · ` +
      `${s.people_already_named} people already named`;
    await load();
  } catch (e) {
    $("status").innerHTML = `<span class="err">${e.message}</span>`;
  }
}

async function load() {
  if (busy) return;
  busy = true;
  $("status").textContent = "loading…";
  $("list").innerHTML = "";
  try {
    const r = await (await fetch(`/api/piles?offset=${offset}&limit=${pageSize}`)).json();
    if (r.error) throw new Error(r.error);
    total = r.total;
    $("status").textContent = r.piles.length ? "" : "nothing here";
    for (const p of r.piles) $("list").appendChild(render(p));
  } catch (e) {
    $("status").innerHTML = `<span class="err">${e.message}</span>`;
  } finally {
    busy = false;
    paint();
  }
}

function render(p) {
  const el = document.createElement("div");
  el.className = "pile";
  const eye = p.median_eye == null ? "?" : Math.round(p.median_eye);
  const coh = p.coherence == null ? "?" : p.coherence.toFixed(2);
  const hidden = p.n_faces - p.faces.length;
  el.innerHTML =
    `<div class="phead">
       <span class="rank">#${p.rank}</span>
       <span class="stat"><b>${p.n_faces}</b> faces</span>
       <span class="stat">eye <b>${eye}px</b></span>
       <span class="stat">coherence <b>${coh}</b></span>
       <span class="stat">score <b>${p.score.toFixed(2)}</b></span>
       <span class="stat">pile ${p.pile_id}</span>
     </div>
     <div class="faces"></div>`;
  const box = el.querySelector(".faces");
  for (const f of p.faces) {
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = `/crop/${f.face_id}?kind=${context ? "context" : "aligned"}`;
    img.title = `face ${f.face_id} · eye ${Math.round(f.interocular_px || 0)}px`;
    box.appendChild(img);
  }
  if (hidden > 0) {
    const s = document.createElement("span");
    s.className = "more";
    s.textContent = `+${hidden} more`;
    box.appendChild(s);
  }
  return el;
}

function paint() {
  const from = total ? offset + 1 : 0;
  const to = Math.min(offset + pageSize, total);
  $("where").textContent = `${from}-${to} of ${total.toLocaleString()}`;
  $("prev").disabled = $("first").disabled = offset <= 0;
  $("next").disabled = $("last").disabled = offset + pageSize >= total;
  $("kind").textContent = context ? "show tight crop" : "show wider crop";
}

function goto(next) {
  const max = Math.max(0, total - pageSize);
  const clamped = Math.min(Math.max(next, 0), max);
  if (clamped === offset) return;
  offset = clamped;
  load();
}

$("next").onclick = () => goto(offset + pageSize);
$("prev").onclick = () => goto(offset - pageSize);
$("first").onclick = () => goto(0);
$("last").onclick = () => goto(total);
$("go").onclick = () => {
  const v = parseInt($("jump").value, 10);
  if (!isNaN(v)) goto(v - 1);
};
$("jump").onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); $("go").click(); } };
$("kind").onclick = () => { context = !context; load(); };

document.addEventListener("keydown", (e) => {
  // Never steal a keystroke aimed at the jump box -- the labelling tool's worst bug was a
  // key reaching the wrong target.
  if (e.target.tagName === "INPUT" || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "n") { e.preventDefault(); $("next").click(); }
  else if (e.key === "p") { e.preventDefault(); $("prev").click(); }
  else if (e.key === "t") { e.preventDefault(); $("kind").click(); }
});

boot();
</script>
</body>
</html>
"""
