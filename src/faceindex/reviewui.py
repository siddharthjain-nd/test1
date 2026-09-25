"""The review server: name piles, merge, repair, search.

v0 was read-only on purpose -- the ranking is a judgement no measurement can grade, so it
was put in front of a human before anything was built on it. That question is settled: the
first two hundred piles were real people, correctly grouped. Everything here builds on that.

Two rules shape the code. Decisions are recorded about FACES, never piles, so a better model
can renumber every cluster without costing a minute of human work. And every mutating action
is one batch, so undo is a delete rather than a reconstruction.

Writes go through a single connection behind a lock: SQLite takes one writer, and the
handler threads would otherwise collide on a busy database.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from faceindex import review, store

PAGE_SIZE = 20
FACES_PER_PILE = 24


class ReviewStore:
    """Reads on a fresh connection per call; writes serialised through one."""

    def __init__(self, db_path: Path, run_id: str | None = None) -> None:
        self.db_path = db_path
        self._write_lock = threading.Lock()
        self._writer: sqlite3.Connection | None = None
        with self._open() as conn:
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
            # An index built before merge suggestions existed has no centroids. The column
            # migrates in as NULL, so the merge screen would simply offer nothing and look
            # broken. Say so instead.
            missing = conn.execute(
                "SELECT COUNT(*) AS n FROM review_piles WHERE run_id = ? AND centroid IS NULL",
                (self.run_id,),
            ).fetchone()["n"]
            self.stale = int(missing) > 0

    def _open(self) -> sqlite3.Connection:
        return store.connect(self.db_path, read_only=True)

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """The single writer. SQLite takes one, and handler threads would otherwise collide."""
        with self._write_lock:
            if self._writer is None:
                # Shared across handler threads, which SQLite forbids unless told otherwise.
                # The lock above is what makes it safe; the flag alone would not.
                self._writer = store.connect(self.db_path, check_same_thread=False)
            yield self._writer

    def close(self) -> None:
        with self._write_lock:
            if self._writer is not None:
                self._writer.close()
                self._writer = None

    # -- reads ---------------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        with self._open() as conn:
            return {
                "run": dict(self.run),
                "page_size": PAGE_SIZE,
                "progress": review.progress(conn, self.run_id),
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

    def next_pile(self) -> dict[str, Any]:
        with self._open() as conn:
            pile = review.next_pile(conn, self.run_id)
            if pile is not None:
                pile["faces"] = review.pile_faces(
                    conn, self.run_id, int(pile["pile_id"]), limit=FACES_PER_PILE
                )
                pile_id = int(pile["pile_id"])
                pile["undecided"] = len(review.undecided_faces(conn, self.run_id, pile_id))
                pile["already"] = review.pile_attribution(conn, self.run_id, pile_id)
            return {"pile": pile, "progress": review.progress(conn, self.run_id)}

    def merge_next(self) -> dict[str, Any]:
        if self.stale:
            raise ValueError(
                "This index predates merge suggestions and has no pile centroids. "
                "Rebuild it: python scripts/build_review_index.py --model w600k_r50.onnx "
                "--similarity 0.51 (your naming is kept -- decisions are stored against "
                "faces, not piles)."
            )
        with self._open() as conn:
            found = review.merge_candidates(conn, self.run_id, limit=1)
            return {
                "suggestion": found[0] if found else None,
                "progress": review.progress(conn, self.run_id),
            }

    def people(self) -> dict[str, Any]:
        with self._open() as conn:
            return {"people": review.people(conn)}

    def person(self, person_id: str, offset: int, limit: int) -> dict[str, Any]:
        with self._open() as conn:
            return review.person_faces(conn, person_id, offset=offset, limit=limit)

    def bucket(self, kind: str, offset: int, limit: int) -> dict[str, Any]:
        with self._open() as conn:
            return review.bucket(conn, self.run_id, kind=kind, offset=offset, limit=limit)

    def search(self, query: str) -> dict[str, Any]:
        with self._open() as conn:
            return {"people": review.search_people(conn, query)}

    def crop(self, face_id: int, *, context: bool) -> Path | None:
        with self._open() as conn:
            return review.crop_path(conn, face_id, context=context)

    # -- writes --------------------------------------------------------------------------

    def assign(self, pile_id: int, person_id: str | None, name: str | None) -> dict[str, Any]:
        with self._write() as conn:
            result = review.assign_pile(
                conn, self.run_id, pile_id, person_id=person_id, name=name
            )
            result["progress"] = review.progress(conn, self.run_id)
            return result

    def junk(self, pile_id: int) -> dict[str, Any]:
        with self._write() as conn:
            result = review.junk_pile(conn, self.run_id, pile_id)
            result["progress"] = review.progress(conn, self.run_id)
            return result

    def skip(self, pile_id: int) -> dict[str, Any]:
        with self._write() as conn:
            result = review.skip_pile(conn, self.run_id, pile_id)
            result["progress"] = review.progress(conn, self.run_id)
            return result

    def undo(self) -> dict[str, Any]:
        with self._write() as conn:
            result = review.undo_last(conn, self.run_id)
            result["progress"] = review.progress(conn, self.run_id)
            return result

    def merge_accept(self, person_id: str, pile_id: int) -> dict[str, Any]:
        with self._write() as conn:
            result = review.assign_pile(
                conn, self.run_id, pile_id, person_id=person_id, source="merge"
            )
            result["progress"] = review.progress(conn, self.run_id)
            return result

    def merge_reject(self, person_id: str, pile_id: int) -> dict[str, Any]:
        with self._write() as conn:
            result = review.reject_merge(conn, self.run_id, person_id, pile_id)
            result["progress"] = review.progress(conn, self.run_id)
            return result

    def rename(self, person_id: str, name: str) -> dict[str, Any]:
        with self._write() as conn:
            review.rename_person(conn, person_id, name)
            return {"renamed": person_id, "progress": review.progress(conn, self.run_id)}

    def remove(self, person_id: str, face_ids: list[int]) -> dict[str, Any]:
        with self._write() as conn:
            result = review.remove_faces(conn, person_id, face_ids)
            result["progress"] = review.progress(conn, self.run_id)
            return result

    def split(self, person_id: str, face_ids: list[int], name: str) -> dict[str, Any]:
        with self._write() as conn:
            result = review.split_person(conn, person_id, face_ids, name)
            result["progress"] = review.progress(conn, self.run_id)
            return result

    def set_aside(self, face_ids: list[int]) -> dict[str, Any]:
        with self._write() as conn:
            result = review.set_aside(conn, face_ids)
            result["progress"] = review.progress(conn, self.run_id)
            return result

    def restore(self, face_ids: list[int]) -> dict[str, Any]:
        with self._write() as conn:
            result = review.restore_faces(conn, face_ids)
            result["progress"] = review.progress(conn, self.run_id)
            return result


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
                elif route == "/api/next":
                    self._json(data.next_pile())
                elif route == "/api/people":
                    self._json(data.people())
                elif route == "/api/merge":
                    self._json(data.merge_next())
                elif route == "/api/person":
                    self._json(
                        data.person(
                            query.get("person_id", [""])[0],
                            _int(query, "offset", 0),
                            _int(query, "limit", 60),
                        )
                    )
                elif route == "/api/bucket":
                    self._json(
                        data.bucket(
                            query.get("kind", ["junk"])[0],
                            _int(query, "offset", 0),
                            _int(query, "limit", 60),
                        )
                    )
                elif route == "/api/search":
                    self._json(data.search(query.get("q", [""])[0]))
                elif route.startswith("/crop/"):
                    self._serve_crop(route, query)
                else:
                    self._json({"error": "not found"}, 404)
            except ValueError as exc:
                self._json({"error": str(exc)}, 400)
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

        def do_POST(self) -> None:
            route = urlparse(self.path).path
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json({"error": "bad json"}, 400)
                return
            if not isinstance(body, dict):
                self._json({"error": "bad json"}, 400)
                return

            try:
                if route == "/api/assign":
                    self._json(
                        data.assign(
                            int(body["pile_id"]),
                            body.get("person_id") or None,
                            body.get("name") or None,
                        )
                    )
                elif route == "/api/junk":
                    self._json(data.junk(int(body["pile_id"])))
                elif route == "/api/skip":
                    self._json(data.skip(int(body["pile_id"])))
                elif route == "/api/merge/accept":
                    self._json(data.merge_accept(str(body["person_id"]), int(body["pile_id"])))
                elif route == "/api/merge/reject":
                    self._json(data.merge_reject(str(body["person_id"]), int(body["pile_id"])))
                elif route == "/api/person/remove":
                    self._json(data.remove(str(body["person_id"]), _ids(body)))
                elif route == "/api/person/split":
                    self._json(
                        data.split(str(body["person_id"]), _ids(body), str(body.get("name", "")))
                    )
                elif route == "/api/faces/aside":
                    self._json(data.set_aside(_ids(body)))
                elif route == "/api/faces/restore":
                    self._json(data.restore(_ids(body)))
                elif route == "/api/undo":
                    self._json(data.undo())
                elif route == "/api/rename":
                    self._json(data.rename(str(body["person_id"]), str(body.get("name", ""))))
                else:
                    self._json({"error": "not found"}, 404)
            except (KeyError, TypeError) as exc:
                self._json({"error": f"missing or malformed field: {exc}"}, 400)
            except ValueError as exc:
                self._json({"error": str(exc)}, 400)
            except Exception as exc:
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    return Handler


def _ids(body: dict[str, Any]) -> list[int]:
    raw = body.get("face_ids")
    if not isinstance(raw, list):
        raise TypeError("face_ids must be a list")
    return [int(value) for value in raw]


def _int(query: dict[str, list[str]], key: str, default: int) -> int:
    raw = query.get(key, [""])[0]
    try:
        return max(int(raw), 0)
    except (TypeError, ValueError):
        return default


def serve(
    db_path: Path, *, host: str = "127.0.0.1", port: int = 8766, run_id: str | None = None
) -> ThreadingHTTPServer:
    data = ReviewStore(db_path, run_id)
    server = ThreadingHTTPServer((host, port), make_handler(data))
    # The launcher warns about a stale index before the browser is even opened.
    server.review_store = data  # type: ignore[attr-defined]
    return server


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Face review</title>
<style>
  :root { --bg:#14161a; --card:#1d2027; --line:#2b2f38; --text:#e8eaed; --dim:#9aa1ad;
          --accent:#5b8dee; --good:#5fd08a; --bad:#ff8b8b; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  header { position:sticky; top:0; z-index:5; background:var(--bg);
           border-bottom:1px solid var(--line); padding:10px 20px; }
  .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  .grow { flex:1; }
  h1 { font-size:15px; margin:0; font-weight:600; }
  .dim { color:var(--dim); font-size:12px; }
  .tab { background:none; border:none; color:var(--dim); padding:5px 10px; cursor:pointer;
         border-radius:6px; font-size:13px; }
  .tab.on { background:var(--card); color:var(--text); }
  button, input { background:var(--card); color:var(--text); border:1px solid var(--line);
                  border-radius:6px; padding:6px 12px; font-size:13px; cursor:pointer; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }
  button:hover:not(:disabled) { filter:brightness(1.15); }
  button:disabled, input:disabled { opacity:.4; cursor:default; }
  input { cursor:text; }
  #name { min-width:260px; font-size:15px; padding:8px 12px; }
  kbd { background:var(--card); border:1px solid var(--line); border-radius:4px;
        padding:1px 5px; font-size:11px; color:var(--dim); }
  main { padding:16px 20px 80px; }
  .pile { background:var(--card); border:1px solid var(--line); border-radius:8px;
          padding:14px; margin-bottom:14px; }
  .phead { display:flex; gap:14px; align-items:baseline; flex-wrap:wrap; margin-bottom:10px; }
  .rank { font-size:16px; font-weight:600; }
  .stat b { color:var(--text); font-weight:600; }
  .stat { color:var(--dim); font-size:12px; }
  .faces { display:flex; flex-wrap:wrap; gap:6px; }
  .faces img { width:96px; height:96px; object-fit:cover; border-radius:5px; background:#000;
               border:1px solid var(--line); }
  .browse .faces img { width:78px; height:78px; }
  .more { color:var(--dim); font-size:12px; align-self:center; padding-left:6px; }
  #msg { min-height:20px; font-size:13px; margin-top:8px; }
  .err { color:var(--bad); }
  .ok { color:var(--good); }
  .done { text-align:center; padding:60px 20px; color:var(--dim); }
  .hide { display:none; }
  .faces img.sel { outline:3px solid var(--accent); outline-offset:-3px; }
  .faces img.pick { cursor:pointer; }
  .roster { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:14px; }
  .chip { background:var(--card); border:1px solid var(--line); border-radius:20px;
          padding:5px 12px; font-size:13px; cursor:pointer; }
  .chip.on { border-color:var(--accent); color:var(--accent); }
</style>
</head>
<body>
<header>
  <div class="row">
    <h1>Face review</h1>
    <button class="tab on" id="tab-review">Review</button>
    <button class="tab" id="tab-merge">Merge</button>
    <button class="tab" id="tab-people">People</button>
    <button class="tab" id="tab-bucket">Not filed</button>
    <button class="tab" id="tab-browse">Browse order</button>
    <span class="grow"></span>
    <span class="dim" id="progress">…</span>
  </div>
</header>

<main>
  <section id="view-review">
    <div id="current"></div>
    <div class="row">
      <input id="name" list="people" placeholder="Who is this?" autocomplete="off" disabled>
      <datalist id="people"></datalist>
      <button class="primary" id="save" disabled>Name it</button>
      <button id="skip" disabled>Skip <kbd>s</kbd></button>
      <button id="junk" disabled>Not a person <kbd>j</kbd></button>
      <button id="undo">Undo <kbd>u</kbd></button>
    </div>
    <div id="msg"></div>
  </section>

  <section id="view-merge" class="hide">
    <div id="suggestion"></div>
    <div class="row">
      <button class="primary" id="same" disabled>Same person <kbd>y</kbd></button>
      <button id="different" disabled>Different <kbd>n</kbd></button>
      <button id="undo2">Undo <kbd>u</kbd></button>
    </div>
    <div id="msg2"></div>
  </section>

  <section id="view-people" class="hide">
    <div class="row" style="margin-bottom:12px">
      <input id="q" placeholder="Search people" autocomplete="off" style="min-width:240px">
      <span class="stat" id="whoami"></span>
    </div>
    <div id="roster"></div>
    <div id="person"></div>
    <div class="row" id="person-actions" style="display:none">
      <span class="stat" id="picked">nothing selected</span>
      <button id="not-them" disabled>Not this person</button>
      <input id="split-name" placeholder="Split off as…" autocomplete="off" style="min-width:180px">
      <button id="do-split" disabled>Split off</button>
      <button id="do-rename">Rename</button>
      <button id="undo3">Undo <kbd>u</kbd></button>
    </div>
    <div id="msg3"></div>
  </section>

  <section id="view-bucket" class="hide">
    <div class="row" style="margin-bottom:12px">
      <button class="tab on" id="b-junk">Set aside by you</button>
      <button class="tab" id="b-lone">Never matched</button>
      <span class="stat" id="bucket-count"></span>
      <span class="grow"></span>
      <span class="stat" id="picked2">nothing selected</span>
      <button id="do-restore" disabled>Put back in the queue</button>
      <button id="b-prev">&lsaquo; prev</button>
      <button id="b-next">next &rsaquo;</button>
    </div>
    <div id="bucket"></div>
    <div id="msg4"></div>
  </section>

  <section id="view-browse" class="browse hide">
    <div class="row" style="margin-bottom:12px">
      <button id="first">&laquo; best</button>
      <button id="prev">&lsaquo; prev</button>
      <span class="stat" id="where"></span>
      <button id="next">next &rsaquo;</button>
      <button id="last">worst &raquo;</button>
      <input id="jump" type="number" min="1" placeholder="rank" style="width:90px">
      <button id="go">go</button>
      <button id="kind">wider crop <kbd>t</kbd></button>
    </div>
    <div id="list"></div>
  </section>
</main>

<script>
let busy = false, context = false, pile = null;
let offset = 0, pageSize = 20, total = 0, labels = new Map();

const $ = (id) => document.getElementById(id);
const say = (text, cls) => { $("msg").className = cls || ""; $("msg").textContent = text; };

async function api(path, body) {
  const options = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : undefined;
  const response = await fetch(path, options);
  const data = await response.json();
  if (data.error) throw new Error(data.error);
  return data;
}

function showProgress(p) {
  if (!p) return;
  const pct = p.faces_total ? Math.round(100 * p.faces_decided / p.faces_total) : 0;
  $("progress").textContent =
    `${p.faces_decided.toLocaleString()} / ${p.faces_total.toLocaleString()} faces (${pct}%) · ` +
    `${p.people_named} named of ${p.people_known} people · ${p.piles_skipped} skipped`;
}

function faceGrid(faces, n_faces) {
  const box = document.createElement("div");
  box.className = "faces";
  for (const f of faces) {
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = `/crop/${f.face_id}?kind=${context ? "context" : "aligned"}`;
    img.title = `face ${f.face_id} · eye ${Math.round(f.interocular_px || 0)}px`;
    box.appendChild(img);
  }
  const hidden = n_faces - faces.length;
  if (hidden > 0) {
    const s = document.createElement("span");
    s.className = "more";
    s.textContent = `+${hidden} more`;
    box.appendChild(s);
  }
  return box;
}

function pileCard(p, withRank) {
  const el = document.createElement("div");
  el.className = "pile";
  const eye = p.median_eye == null ? "?" : Math.round(p.median_eye);
  const coh = p.coherence == null ? "?" : p.coherence.toFixed(2);
  el.innerHTML =
    `<div class="phead">
       <span class="rank">${withRank ? "#" + p.rank : ""}</span>
       <span class="stat"><b>${p.n_faces}</b> faces</span>
       <span class="stat">eye <b>${eye}px</b></span>
       <span class="stat">coherence <b>${coh}</b></span>
       <span class="stat">score <b>${p.score.toFixed(2)}</b></span>
       <span class="stat">pile ${p.pile_id}</span>
     </div>`;
  el.appendChild(faceGrid(p.faces, p.n_faces));
  return el;
}

// A pile can arrive part-decided, from the gold seed or an earlier merge. Saying so turns
// the common case into confirming a suggestion instead of recalling a name -- and stops one
// person being split in two by someone who could not see they were already there.
function showAlready(p) {
  const named = (p.already || []).filter((a) => a.kind === "person" && a.person_id);
  if (!named.length) return;
  const box = document.createElement("div");
  box.className = "row";
  box.style.marginTop = "10px";
  const top = named[0];
  const label = top.display_name || top.person_id;
  const rest = named.length > 1 ? ` (and ${named.length - 1} other person in here)` : "";
  const text = document.createElement("span");
  text.className = "stat";
  text.innerHTML = `<b>${top.n_faces}</b> of these are already <b>${label}</b>${rest}. ` +
                   `${p.undecided} still undecided.`;
  const confirm = document.createElement("button");
  confirm.textContent = `Same person — ${label}`;
  confirm.onclick = () =>
    act("/api/assign", { pile_id: p.pile_id, person_id: top.person_id },
        (r) => `${r.n_faces} faces filed as ${label}.`);
  box.appendChild(text);
  box.appendChild(confirm);
  $("current").appendChild(box);
}

function setEnabled(on) {
  for (const id of ["name", "save", "skip", "junk"]) $(id).disabled = !on;
}

async function loadPeople() {
  const data = await api("/api/people");
  labels = new Map();
  const list = $("people");
  list.innerHTML = "";
  for (const person of data.people) {
    const label = person.display_name || person.person_id;
    labels.set(label.toLowerCase(), person.person_id);
    const option = document.createElement("option");
    option.value = label;
    option.label = `${person.n_faces} faces`;
    list.appendChild(option);
  }
}

async function loadNext() {
  if (busy) return;
  busy = true;
  setEnabled(false);
  try {
    const data = await api("/api/next");
    showProgress(data.progress);
    pile = data.pile;
    $("current").innerHTML = "";
    if (!pile) {
      $("current").innerHTML =
        `<div class="done">Every pile has been handled.<br>
         Undo still works, and Browse order shows all of them.</div>`;
      return;
    }
    $("current").appendChild(pileCard(pile, true));
    showAlready(pile);
    setEnabled(true);
    $("name").value = "";
    $("name").focus();
  } catch (e) {
    say(e.message, "err");
  } finally {
    busy = false;
  }
}

async function act(path, body, describe) {
  if (busy || !pile) return;
  busy = true;
  setEnabled(false);
  try {
    const result = await api(path, body);
    say(describe(result), "ok");
    await loadPeople();
    busy = false;
    await loadNext();
  } catch (e) {
    say(e.message, "err");
    busy = false;
    setEnabled(true);
  }
}

function save() {
  const typed = $("name").value.trim();
  if (!typed) { say("Type a name first, or press s to skip.", "err"); return; }
  const known = labels.get(typed.toLowerCase());
  const body = { pile_id: pile.pile_id };
  if (known) body.person_id = known; else body.name = typed;
  act("/api/assign", body, (r) => `${r.n_faces} faces filed.`);
}

$("save").onclick = save;
$("skip").onclick = () => act("/api/skip", { pile_id: pile.pile_id }, () => "Skipped.");
$("junk").onclick = () => act("/api/junk", { pile_id: pile.pile_id },
                              (r) => `${r.n_faces} faces set aside.`);
$("undo").onclick = async () => {
  if (busy) return;
  busy = true;
  try {
    const r = await api("/api/undo", {});
    say(r.undone ? `Undone (${r.undone}).` : "Nothing to undo.", r.undone ? "ok" : "");
    await loadPeople();
  } catch (e) {
    say(e.message, "err");
  } finally {
    busy = false;
    await loadNext();
  }
};

// Enter must not bubble: in the labelling tool a key that reached a second handler applied
// the action to the pile that had already moved on.
$("name").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); e.stopPropagation(); save(); }
});

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.metaKey || e.ctrlKey || e.altKey) return;
  const visible = (id) => !$(id).classList.contains("hide");
  // Deliberately no shortcut acts on a selection. A keystroke that reached faces the human
  // had not clicked mislabelled fifty of them in the labelling tool.
  const map = visible("view-merge")
    ? { y: "same", n: "different", u: "undo2" }
    : visible("view-browse")
      ? { t: "kind" }
      : visible("view-people")
        ? { u: "undo3" }
        : visible("view-bucket")
          ? {}
          : { s: "skip", j: "junk", u: "undo" };
  const hit = map[e.key];
  if (!hit) return;
  e.preventDefault();
  $(hit).click();
});

// ---- merge view -------------------------------------------------------------------------
// The measured case for this screen: recall 0.906 against precision 0.976. The machine
// splits people far more often than it mixes them, so confirming "these two are one person"
// is the highest-value thing a human can do here.

let suggestion = null;
const say2 = (t, c) => { $("msg2").className = c || ""; $("msg2").textContent = t; };

function sideBySide(s) {
  const el = document.createElement("div");
  el.className = "pile";
  const label = s.display_name || s.person_id;
  el.innerHTML =
    `<div class="phead">
       <span class="rank">Same person?</span>
       <span class="stat">similarity <b>${s.similarity.toFixed(2)}</b></span>
       <span class="stat"><b>${s.undecided}</b> faces would be filed</span>
     </div>
     <div class="stat" style="margin:6px 0 4px"><b>${label}</b> — already known</div>`;
  el.appendChild(faceGrid(s.person_faces, s.person_faces.length));
  const caption = document.createElement("div");
  caption.className = "stat";
  caption.style.margin = "12px 0 4px";
  caption.innerHTML = `<b>Pile ${s.pile.pile_id}</b> — ${s.pile.n_faces} faces, not yet named`;
  el.appendChild(caption);
  el.appendChild(faceGrid(s.pile_faces, s.pile.n_faces));
  return el;
}

async function loadMerge() {
  if (busy) return;
  busy = true;
  $("same").disabled = $("different").disabled = true;
  try {
    const data = await api("/api/merge");
    showProgress(data.progress);
    suggestion = data.suggestion;
    $("suggestion").innerHTML = "";
    if (!suggestion) {
      $("suggestion").innerHTML =
        `<div class="done">No suggestions left.<br>
         Name more piles in Review and new ones will appear.</div>`;
      return;
    }
    $("suggestion").appendChild(sideBySide(suggestion));
    $("same").disabled = $("different").disabled = false;
  } catch (e) {
    say2(e.message, "err");
  } finally {
    busy = false;
  }
}

async function answer(path, describe) {
  if (busy || !suggestion) return;
  busy = true;
  $("same").disabled = $("different").disabled = true;
  const body = { person_id: suggestion.person_id, pile_id: suggestion.pile.pile_id };
  try {
    const result = await api(path, body);
    say2(describe(result), "ok");
    await loadPeople();
  } catch (e) {
    say2(e.message, "err");
  } finally {
    busy = false;
    await loadMerge();
  }
}

$("same").onclick = () => answer("/api/merge/accept", (r) => `${r.n_faces} faces filed.`);
$("different").onclick = () => answer("/api/merge/reject", () => "Noted — not asked again.");
$("undo2").onclick = async () => {
  if (busy) return;
  busy = true;
  try {
    const r = await api("/api/undo", {});
    say2(r.undone ? `Undone (${r.undone}).` : "Nothing to undo.", r.undone ? "ok" : "");
    await loadPeople();
  } catch (e) {
    say2(e.message, "err");
  } finally {
    busy = false;
    await loadMerge();
  }
};

// ---- people view ------------------------------------------------------------------------
// Repairs are the rarer job (precision 0.976), so this screen is not the default. Selection
// is explicit and visible, and no keystroke ever acts on it: the labelling tool's worst bug
// was a key applying to faces the human had not clicked.

let person = null;
const picked = new Set();
const say3 = (t, c) => { $("msg3").className = c || ""; $("msg3").textContent = t; };

function selectableGrid(faces, onChange) {
  const box = document.createElement("div");
  box.className = "faces";
  for (const f of faces) {
    const img = document.createElement("img");
    img.className = "pick";
    img.loading = "lazy";
    img.src = `/crop/${f.face_id}?kind=${context ? "context" : "aligned"}`;
    img.title = `face ${f.face_id} · eye ${Math.round(f.interocular_px || 0)}px`;
    img.onclick = () => {
      if (picked.has(f.face_id)) picked.delete(f.face_id); else picked.add(f.face_id);
      img.classList.toggle("sel", picked.has(f.face_id));
      onChange();
    };
    box.appendChild(img);
  }
  return box;
}

function paintPicked() {
  const n = picked.size;
  const text = n ? `${n} face${n > 1 ? "s" : ""} selected` : "nothing selected";
  $("picked").textContent = text;
  $("picked2").textContent = text;
  $("not-them").disabled = $("do-split").disabled = n === 0;
  $("do-restore").disabled = n === 0;
}

async function loadRoster(query) {
  const data = await api(`/api/search?q=${encodeURIComponent(query || "")}`);
  const box = document.createElement("div");
  box.className = "roster";
  if (!data.people.length) box.innerHTML = `<span class="stat">nobody matches</span>`;
  for (const p of data.people) {
    const chip = document.createElement("button");
    chip.className = "chip" + (person && person.person_id === p.person_id ? " on" : "");
    chip.textContent = `${p.display_name || p.person_id} · ${p.n_faces}`;
    chip.onclick = () => openPerson(p.person_id);
    box.appendChild(chip);
  }
  $("roster").innerHTML = "";
  $("roster").appendChild(box);
}

async function openPerson(person_id) {
  picked.clear();
  try {
    person = await api(`/api/person?person_id=${encodeURIComponent(person_id)}&limit=120`);
    $("whoami").textContent =
      `${person.display_name || person.person_id} — ${person.n_faces} faces`;
    $("person").innerHTML = "";
    const card = document.createElement("div");
    card.className = "pile";
    card.appendChild(selectableGrid(person.faces, paintPicked));
    if (person.n_faces > person.faces.length) {
      const more = document.createElement("span");
      more.className = "more";
      more.textContent = `+${person.n_faces - person.faces.length} more not shown`;
      card.appendChild(more);
    }
    $("person").appendChild(card);
    $("person-actions").style.display = "flex";
    paintPicked();
    await loadRoster($("q").value);
  } catch (e) {
    say3(e.message, "err");
  }
}

async function personAction(path, body, describe) {
  if (busy || !person) return;
  busy = true;
  try {
    const r = await api(path, body);
    say3(describe(r), "ok");
    await loadPeople();
    await openPerson(person.person_id);
  } catch (e) {
    say3(e.message, "err");
  } finally {
    busy = false;
  }
}

$("q").addEventListener("input", () => loadRoster($("q").value));
$("not-them").onclick = () =>
  personAction("/api/person/remove",
               { person_id: person.person_id, face_ids: [...picked] },
               (r) => `${r.n_faces} faces taken off and back in the queue.`);
$("do-split").onclick = () => {
  const name = $("split-name").value.trim();
  if (!name) { say3("Type a name to split them off as.", "err"); return; }
  personAction("/api/person/split",
               { person_id: person.person_id, face_ids: [...picked], name },
               (r) => `${r.n_faces} faces moved to ${name}.`);
  $("split-name").value = "";
};
$("do-rename").onclick = async () => {
  if (!person) return;
  const name = prompt("New name", person.display_name || "");
  if (name === null) return;
  try {
    await api("/api/rename", { person_id: person.person_id, name });
    await loadPeople();
    await openPerson(person.person_id);
    say3("Renamed.", "ok");
  } catch (e) {
    say3(e.message, "err");
  }
};
$("undo3").onclick = async () => {
  try {
    const r = await api("/api/undo", {});
    say3(r.undone ? `Undone (${r.undone}).` : "Nothing to undo.", r.undone ? "ok" : "");
    await loadPeople();
    if (person) await openPerson(person.person_id);
  } catch (e) { say3(e.message, "err"); }
};

// ---- not-filed view ---------------------------------------------------------------------

let bucketKind = "junk", bucketOffset = 0, bucketTotal = 0;
const BUCKET_PAGE = 60;
const say4 = (t, c) => { $("msg4").className = c || ""; $("msg4").textContent = t; };

async function loadBucket() {
  picked.clear();
  try {
    const data = await api(`/api/bucket?kind=${bucketKind}&offset=${bucketOffset}&limit=${BUCKET_PAGE}`);
    bucketTotal = data.n_faces;
    $("bucket-count").textContent =
      bucketTotal
        ? `${bucketOffset + 1}-${Math.min(bucketOffset + BUCKET_PAGE, bucketTotal)} of ${bucketTotal.toLocaleString()}`
        : "empty";
    $("bucket").innerHTML = "";
    const card = document.createElement("div");
    card.className = "pile";
    card.appendChild(selectableGrid(data.faces, paintPicked));
    $("bucket").appendChild(card);
    $("do-restore").style.display = bucketKind === "junk" ? "" : "none";
    $("b-prev").disabled = bucketOffset <= 0;
    $("b-next").disabled = bucketOffset + BUCKET_PAGE >= bucketTotal;
    paintPicked();
  } catch (e) {
    say4(e.message, "err");
  }
}

function setBucket(kind) {
  bucketKind = kind;
  bucketOffset = 0;
  $("b-junk").classList.toggle("on", kind === "junk");
  $("b-lone").classList.toggle("on", kind === "lone");
  loadBucket();
}
$("b-junk").onclick = () => setBucket("junk");
$("b-lone").onclick = () => setBucket("lone");
$("b-prev").onclick = () => { bucketOffset = Math.max(0, bucketOffset - BUCKET_PAGE); loadBucket(); };
$("b-next").onclick = () => { bucketOffset += BUCKET_PAGE; loadBucket(); };
$("do-restore").onclick = async () => {
  if (!picked.size) return;
  try {
    const r = await api("/api/faces/restore", { face_ids: [...picked] });
    say4(`${r.n_faces} faces back in the queue.`, "ok");
    await loadBucket();
  } catch (e) { say4(e.message, "err"); }
};

// ---- browse view ----------------------------------------------------------------------

async function loadBrowse() {
  $("list").innerHTML = "";
  try {
    const r = await api(`/api/piles?offset=${offset}&limit=${pageSize}`);
    total = r.total;
    for (const p of r.piles) $("list").appendChild(pileCard(p, true));
  } catch (e) {
    $("list").innerHTML = `<span class="err">${e.message}</span>`;
  }
  const from = total ? offset + 1 : 0;
  $("where").textContent = `${from}-${Math.min(offset + pageSize, total)} of ${total.toLocaleString()}`;
  $("prev").disabled = $("first").disabled = offset <= 0;
  $("next").disabled = $("last").disabled = offset + pageSize >= total;
  $("kind").textContent = context ? "tight crop" : "wider crop";
}

function goto(next) {
  const clamped = Math.min(Math.max(next, 0), Math.max(0, total - pageSize));
  if (clamped === offset && $("list").children.length) return;
  offset = clamped;
  loadBrowse();
}

$("next").onclick = () => goto(offset + pageSize);
$("prev").onclick = () => goto(offset - pageSize);
$("first").onclick = () => goto(0);
$("last").onclick = () => goto(total);
$("go").onclick = () => { const v = parseInt($("jump").value, 10); if (!isNaN(v)) goto(v - 1); };
$("jump").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); e.stopPropagation(); $("go").click(); }
});
$("kind").onclick = () => { context = !context; loadBrowse(); };

function showView(which) {
  for (const name of ["review", "merge", "people", "bucket", "browse"]) {
    $("view-" + name).classList.toggle("hide", name !== which);
    $("tab-" + name).classList.toggle("on", name === which);
  }
  if (which === "review") $("name").focus();
  if (which === "merge") loadMerge();
  if (which === "people") { picked.clear(); loadRoster($("q").value); }
  if (which === "bucket") loadBucket();
  if (which === "browse" && !$("list").children.length) loadBrowse();
}
$("tab-review").onclick = () => showView("review");
$("tab-merge").onclick = () => showView("merge");
$("tab-people").onclick = () => showView("people");
$("tab-bucket").onclick = () => showView("bucket");
$("tab-browse").onclick = () => showView("browse");

(async function boot() {
  try {
    const s = await api("/api/summary");
    pageSize = s.page_size;
    total = s.run.n_piles;
    showProgress(s.progress);
    document.title = `Face review - ${s.run.run_id}`;
    await loadPeople();
    await loadNext();
  } catch (e) {
    say(e.message, "err");
  }
})();
</script>
</body>
</html>
"""
