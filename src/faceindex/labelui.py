"""Keyboard-driven grid labelling UI for the gold set.

The human only ever *confirms* -- selection was automated by the sampler. A clean
bootstrap cluster is accepted with one keystroke; the few wrong faces are clicked out
first and returned to the queue to be judged individually.

This is the Phase 7 review UI in embryo (PLAN.md: "build it once, here, and grow it").
It runs on the standard library's HTTP server rather than FastAPI because adding
dependencies to a lock file that must stay byte-identical across two machines is a real
cost, and this serves one user on localhost. Phase 7 can swap the transport; the label
semantics and the frontend are the parts worth keeping.

Binds to 127.0.0.1 only. The data behind it is biometric and must never leave the machine.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

LABELS = ("person", "not_of_interest", "non_face", "unsure")

# Faces judged individually are served in pages of this size.
LEFTOVER_PAGE = 40


def _person_sort_key(person_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", person_id)
    return (int(match.group(1)) if match else 1 << 30, person_id)


class GoldStore:
    """All database access for the UI. One connection, one thread of writes."""

    def __init__(self, db_path: Path) -> None:
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

    def progress(self) -> dict[str, Any]:
        total = self.conn.execute("SELECT COUNT(*) AS n FROM gold_candidates").fetchone()["n"]
        done = self.conn.execute(
            "SELECT COUNT(*) AS n FROM gold_labels g JOIN gold_candidates c "
            "ON c.face_id = g.face_id"
        ).fetchone()["n"]
        by_label = {
            str(r["label"]): int(r["n"])
            for r in self.conn.execute(
                "SELECT g.label, COUNT(*) AS n FROM gold_labels g "
                "JOIN gold_candidates c ON c.face_id = g.face_id GROUP BY g.label"
            )
        }
        people = self.conn.execute(
            "SELECT COUNT(DISTINCT person_id) AS n FROM gold_labels WHERE person_id IS NOT NULL"
        ).fetchone()["n"]
        return {
            "total": int(total),
            "done": int(done),
            "remaining": int(total) - int(done),
            "by_label": by_label,
            "n_people": int(people),
        }

    def persons(self) -> list[dict[str, Any]]:
        """Everyone named so far, each with a face to recognise them by.

        The thumbnail is the point. After an hour nobody remembers who ``cousin_a`` was, and
        a name you cannot place is a name you will accidentally duplicate -- which splits one
        person in two and makes the scorer mark correct grouping as an error.
        """
        rows = self.conn.execute(
            """
            SELECT g.person_id AS person_id, COUNT(*) AS n,
                   (SELECT g2.face_id FROM gold_labels g2
                      JOIN faces f2 ON f2.id = g2.face_id
                     WHERE g2.person_id = g.person_id
                     ORDER BY f2.interocular_px DESC LIMIT 1) AS sample
            FROM gold_labels g
            WHERE g.person_id IS NOT NULL
            GROUP BY g.person_id
            """
        ).fetchall()
        people = [(str(r["person_id"]), int(r["n"]), r["sample"]) for r in rows]
        # Numeric sort, so person_10 does not fall between person_1 and person_2 -- a lexical
        # order makes an existing id easy to miss and duplicate.
        people.sort(key=lambda entry: _person_sort_key(entry[0]))
        return [
            {"person_id": person_id, "count": count, "sample": sample}
            for person_id, count, sample in people
        ]

    def next_person_id(self) -> str:
        highest = 0
        for person in self.persons():
            match = re.search(r"(\d+)$", person["person_id"])
            if match:
                highest = max(highest, int(match.group(1)))
        return f"person_{highest + 1}"

    def next_group(self) -> dict[str, Any]:
        """The next unlabelled bootstrap cluster, else a page of leftovers and noise.

        Clusters are served largest first: confirming a big clean cluster is the single
        highest-value keystroke available, and it shrinks the queue fastest.
        """
        rows = self.conn.execute(
            """
            SELECT c.bootstrap_cluster AS cluster_id, COUNT(*) AS n
            FROM gold_candidates c
            LEFT JOIN gold_labels g ON g.face_id = c.face_id
            WHERE g.face_id IS NULL AND c.bootstrap_cluster != -1
            GROUP BY c.bootstrap_cluster
            HAVING n > 1
            ORDER BY n DESC, cluster_id ASC
            LIMIT 1
            """
        ).fetchone()

        if rows is not None:
            cluster_id = int(rows["cluster_id"])
            faces = self._faces_where(
                "c.bootstrap_cluster = ? AND g.face_id IS NULL", (cluster_id,)
            )
            return {
                "kind": "cluster",
                "cluster_id": cluster_id,
                "title": f"Cluster {cluster_id}",
                "hint": "Click any face that does NOT belong, then press Enter to accept the rest.",
                "faces": faces,
            }

        faces = self._faces_where("g.face_id IS NULL", (), limit=LEFTOVER_PAGE)
        if not faces:
            return {"kind": "done", "faces": []}

        return {
            "kind": "leftovers",
            "cluster_id": None,
            "title": "Singles, noise and pulled-out faces",
            "hint": "No pre-grouping here. Select faces that are the same person, then Enter.",
            "faces": faces,
        }

    def _faces_where(
        self, where: str, params: tuple[Any, ...], *, limit: int | None = None
    ) -> list[dict[str, Any]]:
        sql = f"""
            SELECT
                c.face_id, c.stratum, c.reserved_for, c.bootstrap_cluster,
                f.det_score, f.interocular_px, f.yaw_deg,
                p.rel_path, p.taken_at, p.kind
            FROM gold_candidates c
            JOIN faces f ON f.id = c.face_id
            JOIN photos p ON p.id = f.photo_id
            LEFT JOIN gold_labels g ON g.face_id = c.face_id
            WHERE {where}
            ORDER BY f.interocular_px DESC
        """
        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        return [
            {
                "face_id": int(r["face_id"]),
                "strata": json.loads(r["stratum"]),
                "reserved_for": r["reserved_for"],
                "cluster_id": int(r["bootstrap_cluster"]),
                "det_score": round(float(r["det_score"]), 3),
                "interocular_px": round(float(r["interocular_px"] or 0.0), 1),
                "yaw_deg": round(float(r["yaw_deg"] or 0.0), 1),
                "taken_at": r["taken_at"],
                "kind": r["kind"],
                "rel_path": r["rel_path"],
            }
            for r in self.conn.execute(sql, params)
        ]

    def crop_path(self, face_id: int, *, context: bool) -> Path | None:
        row = self.conn.execute(
            "SELECT crop_path, context_path FROM faces WHERE id = ?", (face_id,)
        ).fetchone()
        if row is None:
            return None
        chosen = row["context_path"] if context else row["crop_path"]
        if not chosen and context:
            chosen = row["crop_path"]
        return Path(chosen) if chosen else None

    def save_labels(self, entries: list[dict[str, Any]]) -> int:
        now = datetime.now(UTC).isoformat()
        rows = []
        for entry in entries:
            label = str(entry.get("label", ""))
            if label not in LABELS:
                raise ValueError(f"unknown label {label!r}")
            person_id = entry.get("person_id") if label == "person" else None
            if label == "person" and not person_id:
                raise ValueError("label 'person' requires a person_id")
            rows.append((int(entry["face_id"]), label, person_id, entry.get("occlusion"), now))

        self.conn.executemany(
            "INSERT INTO gold_labels (face_id, label, person_id, occlusion, labelled_at) "
            "VALUES (?,?,?,?,?) ON CONFLICT(face_id) DO UPDATE SET "
            "label=excluded.label, person_id=excluded.person_id, "
            "occlusion=excluded.occlusion, labelled_at=excluded.labelled_at",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def undo(self, face_ids: list[int]) -> int:
        cursor = self.conn.executemany(
            "DELETE FROM gold_labels WHERE face_id = ?", [(int(i),) for i in face_ids]
        )
        self.conn.commit()
        return cursor.rowcount

    def label_counts_by_cluster(self) -> dict[int, int]:
        counts: dict[int, int] = defaultdict(int)
        for row in self.conn.execute(
            "SELECT c.bootstrap_cluster AS cid, COUNT(*) AS n FROM gold_candidates c "
            "JOIN gold_labels g ON g.face_id = c.face_id GROUP BY c.bootstrap_cluster"
        ):
            counts[int(row["cid"])] = int(row["n"])
        return dict(counts)


def make_handler(gold: GoldStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "faceindex-label"

        def log_message(self, fmt: str, *args: Any) -> None:
            # Never log file paths at INFO level (PLAN.md section 3, privacy).
            return

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, status: int = 200) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            route = parsed.path

            try:
                if route == "/":
                    self._send(200, INDEX_HTML.encode(), "text/html; charset=utf-8")
                elif route == "/api/next":
                    self._json(gold.next_group())
                elif route == "/api/progress":
                    self._json(gold.progress())
                elif route == "/api/persons":
                    self._json({"persons": gold.persons(), "next_person_id": gold.next_person_id()})
                elif route.startswith("/crop/"):
                    self._serve_crop(route, parse_qs(parsed.query))
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as exc:  # a UI bug must not kill a labelling session
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

        def _serve_crop(self, route: str, query: dict[str, list[str]]) -> None:
            try:
                face_id = int(route.rsplit("/", 1)[1])
            except ValueError:
                self._json({"error": "bad face id"}, 400)
                return

            want_context = query.get("kind", ["context"])[0] != "aligned"
            path = gold.crop_path(face_id, context=want_context)
            if path is None or not path.exists():
                self._json({"error": "crop missing"}, 404)
                return
            self._send(200, path.read_bytes(), "image/jpeg")

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json({"error": "bad json"}, 400)
                return

            try:
                if parsed.path == "/api/label":
                    written = gold.save_labels(payload.get("entries", []))
                    self._json({"written": written, "progress": gold.progress()})
                elif parsed.path == "/api/undo":
                    removed = gold.undo(payload.get("face_ids", []))
                    self._json({"removed": removed, "progress": gold.progress()})
                else:
                    self._json({"error": "not found"}, 404)
            except ValueError as exc:
                self._json({"error": str(exc)}, 400)
            except Exception as exc:
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    return Handler


def serve(db_path: Path, *, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    gold = GoldStore(db_path)
    server = ThreadingHTTPServer((host, port), make_handler(gold))
    return server


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Gold set labelling</title>
<style>
  :root {
    --bg: #14161a; --panel: #1c2026; --line: #2c323c; --fg: #e6e9ef;
    --muted: #9aa4b2; --accent: #6ea8fe; --danger: #ff6b6b; --ok: #51cf66;
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif; }
  header { position:sticky; top:0; z-index:10; background:var(--panel);
           border-bottom:1px solid var(--line); padding:12px 18px; }
  .row { display:flex; align-items:center; gap:18px; flex-wrap:wrap; }
  h1 { font-size:16px; margin:0; font-weight:600; }
  .muted { color:var(--muted); }
  .bar { height:5px; background:var(--line); border-radius:3px; overflow:hidden;
         flex:1; min-width:180px; }
  .bar > div { height:100%; background:var(--accent); width:0%; transition:width .2s; }
  main { padding:18px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(132px,1fr)); gap:10px; }
  figure { margin:0; background:var(--panel); border:2px solid var(--line);
           border-radius:8px; overflow:hidden; cursor:pointer; position:relative; }
  figure.out { border-color:var(--danger); opacity:.45; }
  figure.sel { border-color:var(--accent); }
  figure img { width:100%; aspect-ratio:1; object-fit:cover; display:block; }
  figcaption { padding:5px 7px; font-size:11px; color:var(--muted);
               display:flex; justify-content:space-between; gap:6px; }
  .tag { position:absolute; top:5px; left:5px; background:#000a; padding:1px 6px;
         border-radius:10px; font-size:10px; }
  kbd { background:var(--line); border-radius:4px; padding:1px 6px;
        font:12px ui-monospace,monospace; }
  .keys { display:flex; gap:14px; flex-wrap:wrap; font-size:12px; color:var(--muted); }
  #toast { position:fixed; bottom:18px; left:50%; transform:translateX(-50%);
           background:var(--panel); border:1px solid var(--line); border-radius:8px;
           padding:9px 16px; opacity:0; transition:opacity .25s; pointer-events:none; }
  #toast.show { opacity:1; }
  .done { text-align:center; padding:60px 20px; }

  /* ---- the four verdicts, always on screen ---- */
  .legend { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }
  .verdict {
    position:relative; flex:1 1 168px; min-width:168px;
    border:1px solid var(--line); border-left:3px solid var(--edge);
    border-radius:6px; padding:8px 11px; background:#00000018; cursor:help;
  }
  .verdict:hover, .verdict:focus-visible { border-color:var(--edge); outline:none; }
  .verdict .k { display:flex; align-items:baseline; gap:7px; }
  .verdict .nm { font-weight:600; font-size:13px; }
  .verdict .ct {
    margin-left:auto; font:12px ui-monospace,monospace; color:var(--muted);
    font-variant-numeric:tabular-nums;
  }
  .verdict .sub { font-size:11.5px; color:var(--muted); margin-top:2px; display:block; }

  .verdict .why {
    position:absolute; z-index:20; left:0; top:calc(100% + 7px); width:320px;
    background:var(--panel); border:1px solid var(--edge); border-radius:7px;
    padding:11px 13px; font-size:12.5px; line-height:1.5; color:var(--fg);
    box-shadow:0 10px 30px #0008; opacity:0; visibility:hidden; transition:opacity .14s;
  }
  .verdict:hover .why, .verdict:focus-visible .why { opacity:1; visibility:visible; }
  .verdict .why b { color:var(--edge); }

  #v-person  { --edge:var(--accent); }
  #v-str     { --edge:#e8a33d; }
  #v-non     { --edge:var(--danger); }
  #v-unsure  { --edge:var(--muted); }

  /* ---- what the next keystroke will hit ---- */
  #aim {
    display:block; margin-top:10px; padding:9px 13px; border-radius:6px;
    background:#00000024; border:1px solid var(--line);
    font-size:13px; color:var(--fg);
  }
  #aim b { color:var(--accent); }
  #aim .sel { color:#e8a33d; }

  figure.sel::after {
    content:"✓"; position:absolute; top:5px; right:6px;
    width:19px; height:19px; border-radius:50%;
    background:var(--accent); color:#04121c;
    font-size:12px; font-weight:700; display:grid; place-items:center;
  }
  figure.out::after {
    content:"✕"; position:absolute; top:5px; right:6px;
    width:19px; height:19px; border-radius:50%;
    background:var(--danger); color:#1a0505;
    font-size:11px; font-weight:700; display:grid; place-items:center;
  }

  /* ---- person picker ---- */
  #scrim {
    position:fixed; inset:0; background:#000a; display:none;
    align-items:flex-start; justify-content:center; padding-top:9vh; z-index:50;
  }
  #scrim.on { display:flex; }
  #picker {
    width:min(520px, 92vw); background:var(--panel);
    border:1px solid var(--line); border-radius:11px; overflow:hidden;
    box-shadow:0 24px 70px #000b;
  }
  #picker .top { padding:13px 16px 11px; border-bottom:1px solid var(--line); }
  #picker .top .q { font-size:13px; color:var(--muted); }
  #nameInput {
    width:100%; margin-top:8px; padding:9px 11px; font-size:15px;
    background:var(--bg); color:var(--fg);
    border:1px solid var(--accent); border-radius:6px; outline:none;
  }
  #hits { max-height:46vh; overflow-y:auto; }
  .hit {
    display:flex; align-items:center; gap:11px; padding:8px 14px; cursor:pointer;
    border-left:3px solid transparent;
  }
  .hit.on { background:#ffffff12; border-left-color:var(--accent); }
  .hit img {
    width:40px; height:40px; border-radius:5px; object-fit:cover;
    background:var(--line); flex:none;
  }
  .hit .nm { font-size:14px; }
  .hit .meta { margin-left:auto; font:12px ui-monospace,monospace; color:var(--muted); }
  .hit.new .badge {
    width:40px; height:40px; border-radius:5px; flex:none; display:grid; place-items:center;
    background:var(--accent); color:#04121c; font-size:20px; font-weight:700;
  }
  #picker .foot {
    padding:9px 15px; border-top:1px solid var(--line);
    font-size:11.5px; color:var(--muted); display:flex; gap:14px; flex-wrap:wrap;
  }
  #warnRow {
    display:none; padding:10px 15px; background:#4a3a12;
    border-top:1px solid #6b5418; font-size:12.5px; color:#f0d79a;
  }
  #warnRow.on { display:block; }
</style>
</head>
<body>
<header>
  <div class="row">
    <h1 id="title">Loading…</h1>
    <span class="muted" id="count"></span>
    <div class="bar"><div id="fill"></div></div>
    <span class="muted" id="progress"></span>
  </div>
  <div class="row" style="margin-top:8px">
    <span class="muted" id="hint"></span>
  </div>
  <div class="legend">
    <div class="verdict" id="v-person" tabindex="0">
      <span class="k"><kbd>Enter</kbd><span class="nm">Person</span><span class="ct" id="c-person">0</span></span>
      <span class="sub">NOT clicked · <kbd>⇧Enter</kbd> for clicked</span>
      <span class="why">
        Ground truth for <b>&ldquo;these faces are the same human&rdquo;</b>. Your score is how
        well the system reproduces these groupings.<br><br>
        <b>Enter</b> names the faces you did not click — the usual case, where the pile is one
        person and you clicked the strays.<br>
        <b>Shift+Enter</b> names only the faces you clicked. Use it when one pile turns out to
        hold two people: click one of them, Shift+Enter, then Enter for the rest.<br><br>
        Reuse a name you have already used when you recognise someone again — that is how one
        person gets joined across several piles, and the only way cross-era identities enter
        the gold set.
      </span>
    </div>

    <div class="verdict" id="v-str" tabindex="0">
      <span class="k"><kbd>n</kbd><span class="nm">Stranger</span><span class="ct" id="c-str">0</span></span>
      <span class="sub">real face, no album wanted</span>
      <span class="why">
        A genuine face you would never want an album of — background people, passers-by.<br><br>
        <b>Without these the system is never tested on its right to call a face noise</b>, and
        every wedding invents a dozen phantom people. The plan wants roughly 300 of them.
      </span>
    </div>

    <div class="verdict" id="v-non" tabindex="0">
      <span class="k"><kbd>x</kbd><span class="nm">Not a face</span><span class="ct" id="c-non">0</span></span>
      <span class="sub">only when you are sure</span>
      <span class="why">
        No face is actually present: a poster, a statue, a face inside a <b>framed photograph
        on the wall</b>, or a detector mistake.<br><br>
        This is a <b>positive claim that the detector was wrong</b>, and these counts are how
        detector precision gets measured. A blurry smudge that is probably a real face is not
        this — press <b>u</b> instead, or the detector looks worse than it is and Phase 3
        gating gets tuned against a false picture.
      </span>
    </div>

    <div class="verdict" id="v-unsure" tabindex="0">
      <span class="k"><kbd>u</kbd><span class="nm">Unsure</span><span class="ct" id="c-unsure">0</span></span>
      <span class="sub">not readable from the face</span>
      <span class="why">
        <b>Excluded from scoring entirely.</b> Never guess: a wrong answer key marks correct
        behaviour as failure, and nothing downstream can detect it.<br><br>
        <b>Judge from the face alone.</b> Mentally crop away everything else — if you are
        identifying someone by earrings, hair, clothing or who else is in the shot, press this.
        The system only ever sees the face, so a face whose identity is not in its own pixels
        is unwinnable, and unwinnable cases drag down the profile slice for a reason no amount
        of work can fix.
      </span>
    </div>
  </div>

  <span id="aim"></span>

  <div class="row keys" style="margin-top:9px">
    <span><kbd>click</kbd> mark a face as different from the rest</span>
    <span><kbd>a</kbd> all</span>
    <span><kbd>d</kbd> none</span>
    <span><kbd>s</kbd> skip pile</span>
    <span><kbd>t</kbd> tight / wide crop</span>
    <span><kbd>Ctrl+Z</kbd> undo</span>
    <span class="muted">hover a card above to see what it does to the gold set</span>
  </div>
</header>
<main><div class="grid" id="grid"></div></main>

<div id="scrim">
  <div id="picker">
    <div class="top">
      <span class="q" id="pickQ"></span>
      <input id="nameInput" autocomplete="off" spellcheck="false" placeholder="Type a name…">
    </div>
    <div id="hits"></div>
    <div id="warnRow"></div>
    <div class="foot">
      <span><kbd>↑</kbd><kbd>↓</kbd> move</span>
      <span><kbd>Enter</kbd> choose</span>
      <span><kbd>Esc</kbd> cancel</span>
      <span>reuse a name to join one person across piles</span>
    </div>
  </div>
</div>

<div id="toast"></div>

<script>
let group = null, marked = new Set(), context = true, lastBatch = [];

const $ = (id) => document.getElementById(id);

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 1600);
}

async function api(path, options) {
  const response = await fetch(path, options);
  return response.json();
}

async function refreshProgress() {
  const p = await api("/api/progress");
  $("fill").style.width = p.total ? (100 * p.done / p.total) + "%" : "0%";
  $("progress").textContent =
    `${p.done}/${p.total} labelled · ${p.n_people} people · ${p.remaining} left`;
  updateCounts(p.by_label || {});
}

async function load() {
  group = await api("/api/next");
  marked = new Set();

  if (group.kind === "done") {
    $("title").textContent = "All candidates labelled";
    $("hint").textContent = "";
    $("count").textContent = "";
    $("grid").innerHTML =
      '<div class="done">Nothing left to label.<br>' +
      'Run <code>python scripts/export_gold_set.py</code> to write labels.csv.</div>';
    await refreshProgress();
    return;
  }

  $("title").textContent = group.title;
  $("hint").textContent = group.hint;
  $("count").textContent = group.faces.length + " faces";
  render();
  await refreshProgress();
}

function render() {
  const grid = $("grid");
  grid.innerHTML = "";
  for (const face of group.faces) {
    const fig = document.createElement("figure");
    fig.dataset.id = face.face_id;
    if (marked.has(face.face_id)) fig.classList.add(isCluster() ? "out" : "sel");

    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = `/crop/${face.face_id}?kind=${context ? "context" : "aligned"}`;
    fig.appendChild(img);

    if (face.reserved_for) {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = face.reserved_for === "detector_fp" ? "low conf" : "noise";
      fig.appendChild(tag);
    }

    const cap = document.createElement("figcaption");
    const left = document.createElement("span");
    left.textContent = `${face.interocular_px}px`;
    const right = document.createElement("span");
    right.textContent = (face.taken_at || "undated").slice(0, 7);
    cap.append(left, right);
    fig.appendChild(cap);

    fig.onclick = () => {
      marked.has(face.face_id) ? marked.delete(face.face_id) : marked.add(face.face_id);
      render();
    };
    grid.appendChild(fig);
  }
  updateAim();
}

// Spell out what the next keystroke will actually affect. Without this the two meanings of
// a click are invisible, and a bulk mislabel looks identical to the action you intended.
function updateAim() {
  const person = personTargets().length;
  const clicked = marked.size;
  $("aim").innerHTML = clicked
    ? `<span class="sel">${clicked} face(s) clicked.</span> ` +
      `<b>n</b>/<b>x</b>/<b>u</b> or <b>Shift+Enter</b> acts on those ${clicked}. ` +
      `<b>Enter</b> makes the other ${person} one person.`
    : `Nothing clicked. <b>Enter</b> makes all ${person} one person. ` +
      `<b>n</b>, <b>x</b> or <b>u</b> labels all ${person}. ` +
      `Click the odd ones out first to treat them separately.`;
}

function updateCounts(byLabel) {
  const map = {
    "c-person": byLabel.person || 0,
    "c-str": byLabel.not_of_interest || 0,
    "c-non": byLabel.non_face || 0,
    "c-unsure": byLabel.unsure || 0,
  };
  for (const [id, n] of Object.entries(map)) $(id).textContent = n;
}

const isCluster = () => group && group.kind === "cluster";

// Clicking always means "this face is different from the rest of the pile". What you do
// next decides which half you are talking about:
//
//   Enter  -> the faces you did NOT click become one person  (the majority case)
//   n/x/u  -> the faces you DID click get that label         (the exceptions)
//
// Both readings of a click are natural, and picking the wrong one silently labels fifty
// faces you never looked at. Earlier this applied n/x/u to the unclicked majority, which
// is how a single blurry face turned into fifty "unsure" verdicts.
function personTargets() {
  const ids = group.faces.map((f) => f.face_id);
  return isCluster() ? ids.filter((id) => !marked.has(id)) : ids.filter((id) => marked.has(id));
}

function verdictTargets() {
  const selected = group.faces.map((f) => f.face_id).filter((id) => marked.has(id));
  // Nothing clicked means the verdict is about the whole pile.
  return selected.length ? selected : group.faces.map((f) => f.face_id);
}

const BULK_CONFIRM = 8;

async function submit(label, ids, personId) {
  if (!ids.length) { toast("Nothing selected"); return; }

  if (ids.length >= BULK_CONFIRM && !personId) {
    const what = { not_of_interest: "stranger", non_face: "not a face", unsure: "unsure" }[label];
    if (!confirm(`Mark ${ids.length} faces as "${what}"?\n\nClick individual faces first if you meant only some of them.`)) {
      return;
    }
  }

  const entries = ids.map((id) => ({ face_id: id, label, person_id: personId || null }));
  const result = await api("/api/label", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ entries }),
  });

  if (result.error) { toast("Error: " + result.error); return; }
  lastBatch = ids;
  toast(`${result.written} → ${personId || label}   (Ctrl+Z to undo)`);

  // Drop the judged faces from the pile in place rather than reloading. One pile often
  // holds a few strangers, a poster and one person, and that needs three verdicts -- so
  // reloading between each would throw away your place and re-render everything you had
  // already worked through.
  const done = new Set(ids);
  group.faces = group.faces.filter((f) => !done.has(f.face_id));
  done.forEach((id) => marked.delete(id));

  if (!group.faces.length) {
    await load();
  } else {
    $("count").textContent = group.faces.length + " faces left in this pile";
    render();
    await refreshProgress();
  }
}

// ---------------------------------------------------------------------------------
// Person picker
//
// A plain text box is the single most dangerous control in this tool. Each person shows
// up in roughly forty faces spread across many piles, so over a long session one of them
// inevitably gets typed two ways -- "mom" and "Mom" -- which splits one human into two in
// the answer key. The scorer then marks the system *wrong* for grouping her correctly, and
// nothing downstream can tell that is what happened.
//
// Hence: filter as you type, a face beside every name because nobody remembers who
// "cousin_a" was after an hour, exact matching that ignores case, and a warning when a new
// name is suspiciously close to an existing one.
// ---------------------------------------------------------------------------------

let pickerOpen = false, pickRows = [], pickIndex = 0, pickIds = [], pickPersons = [], pickNext = "";

const norm = (s) => s.trim().toLowerCase().replace(/\\s+/g, " ");

function editDistance(a, b) {
  const d = Array.from({ length: a.length + 1 }, (_, i) => [i, ...Array(b.length).fill(0)]);
  for (let j = 0; j <= b.length; j++) d[0][j] = j;
  for (let i = 1; i <= a.length; i++)
    for (let j = 1; j <= b.length; j++)
      d[i][j] = Math.min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + (a[i-1] === b[j-1] ? 0 : 1));
  return d[a.length][b.length];
}

async function assignPerson(which) {
  // "rest" names the unclicked majority, "clicked" names just what you picked. A pile that
  // turns out to hold two people needs both: click one, Shift+Enter, then Enter for the rest.
  const ids = which === "clicked"
    ? group.faces.map((f) => f.face_id).filter((id) => marked.has(id))
    : personTargets();

  if (!ids.length) {
    toast(which === "clicked" ? "Click some faces first" : "Nothing left unclicked to assign");
    return;
  }

  const data = await api("/api/persons");
  pickPersons = data.persons;
  pickNext = data.next_person_id;
  pickIds = ids;
  pickIndex = 0;
  pickerOpen = true;

  $("pickQ").textContent = `Name these ${ids.length} face(s)`;
  $("nameInput").value = "";
  $("scrim").classList.add("on");
  renderPicker();
  $("nameInput").focus();
}

function closePicker() {
  pickerOpen = false;
  $("scrim").classList.remove("on");
  $("warnRow").classList.remove("on");
}

function renderPicker() {
  const typed = $("nameInput").value;
  const key = norm(typed);
  const matches = key
    ? pickPersons.filter((p) => norm(p.person_id).includes(key))
    : pickPersons;
  const exact = pickPersons.find((p) => norm(p.person_id) === key);

  pickRows = [];
  if (!exact) {
    const name = typed.trim() || pickNext;
    pickRows.push({ kind: "new", person_id: name });
  }
  matches.forEach((p) => pickRows.push({ kind: "old", ...p }));

  if (pickIndex >= pickRows.length) pickIndex = Math.max(0, pickRows.length - 1);

  const hits = $("hits");
  hits.innerHTML = "";
  pickRows.forEach((row, index) => {
    const el = document.createElement("div");
    el.className = "hit" + (index === pickIndex ? " on" : "") + (row.kind === "new" ? " new" : "");

    if (row.kind === "new") {
      const badge = document.createElement("div");
      badge.className = "badge";
      badge.textContent = "+";
      el.appendChild(badge);
    } else {
      const img = document.createElement("img");
      img.loading = "lazy";
      img.src = `/crop/${row.sample}?kind=context`;
      el.appendChild(img);
    }

    const nm = document.createElement("span");
    nm.className = "nm";
    nm.textContent = row.kind === "new" ? `Create “${row.person_id}”` : row.person_id;
    el.appendChild(nm);

    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = row.kind === "new" ? "new person" : `${row.count} faces`;
    el.appendChild(meta);

    el.onclick = () => { pickIndex = index; confirmPick(); };
    hits.appendChild(el);
  });

  // Warn only on a near miss. An outright new name should stay a two-keystroke path.
  const warn = $("warnRow");
  const chosen = pickRows[pickIndex];
  if (chosen && chosen.kind === "new" && key.length >= 3) {
    const close = pickPersons
      .map((p) => ({ p, d: editDistance(key, norm(p.person_id)) }))
      .filter((c) => c.d > 0 && c.d <= 2)
      .sort((a, b) => a.d - b.d)[0];
    if (close) {
      warn.innerHTML =
        `“${typed.trim()}” is very close to existing <b>${close.p.person_id}</b> ` +
        `(${close.p.count} faces). If that is the same person, pick them below instead — ` +
        `two names for one person splits them in the answer key, and the scorer then marks ` +
        `correct grouping as an error.`;
      warn.classList.add("on");
      return;
    }
  }
  warn.classList.remove("on");
}

async function confirmPick() {
  const chosen = pickRows[pickIndex];
  if (!chosen) return;
  const name = chosen.kind === "new" ? (chosen.person_id || pickNext) : chosen.person_id;
  const ids = pickIds;
  closePicker();
  await submit("person", ids, name);
}

$("nameInput").addEventListener("input", () => { pickIndex = 0; renderPicker(); });

$("nameInput").addEventListener("keydown", async (event) => {
  if (event.key === "ArrowDown") {
    event.preventDefault();
    pickIndex = Math.min(pickIndex + 1, pickRows.length - 1);
    renderPicker();
  } else if (event.key === "ArrowUp") {
    event.preventDefault();
    pickIndex = Math.max(pickIndex - 1, 0);
    renderPicker();
  } else if (event.key === "Enter") {
    event.preventDefault();
    await confirmPick();
  } else if (event.key === "Escape") {
    event.preventDefault();
    closePicker();
  }
});

document.addEventListener("keydown", async (event) => {
  // The picker owns the keyboard while it is open, or typing a name would also fire the
  // verdict keys -- "non" would label the pile "not a face" before you finished the word.
  if (pickerOpen) return;
  if (!group || group.kind === "done") return;
  const key = event.key.toLowerCase();

  if (event.key === "Enter") {
    event.preventDefault();
    await assignPerson(event.shiftKey ? "clicked" : "rest");
  }
  else if (key === "a") { group.faces.forEach((f) => marked.add(f.face_id)); render(); }
  else if (key === "d") { marked.clear(); render(); }
  else if (key === "n") await submit("not_of_interest", verdictTargets());
  else if (key === "x") await submit("non_face", verdictTargets());
  else if (key === "u") await submit("unsure", verdictTargets());
  else if (key === "s") { toast("Skipped"); await load(); }
  else if (key === "t") { context = !context; render(); }
  else if (key === "z" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    if (!lastBatch.length) { toast("Nothing to undo"); return; }
    await api("/api/undo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ face_ids: lastBatch }),
    });
    lastBatch = [];
    toast("Undone");
    await load();
  }
});

load();
</script>
</body>
</html>
"""
