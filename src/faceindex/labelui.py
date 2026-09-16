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
        rows = self.conn.execute(
            "SELECT person_id, COUNT(*) AS n FROM gold_labels "
            "WHERE person_id IS NOT NULL GROUP BY person_id"
        ).fetchall()
        people = [(str(r["person_id"]), int(r["n"])) for r in rows]
        # Numeric sort, so person_10 does not fall between person_1 and person_2 in the
        # merge prompt -- a lexical order makes an existing id easy to miss and duplicate.
        people.sort(key=lambda entry: _person_sort_key(entry[0]))
        return [{"person_id": person_id, "count": count} for person_id, count in people]

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
  <div class="row keys" style="margin-top:8px">
    <span><kbd>click</kbd> pull out / select</span>
    <span><kbd>Enter</kbd> accept as person</span>
    <span><kbd>a</kbd> all</span>
    <span><kbd>d</kbd> none</span>
    <span><kbd>n</kbd> stranger</span>
    <span><kbd>x</kbd> not a face</span>
    <span><kbd>u</kbd> unsure</span>
    <span><kbd>s</kbd> skip</span>
    <span><kbd>t</kbd> tight/context crop</span>
  </div>
</header>
<main><div class="grid" id="grid"></div></main>
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
}

const isCluster = () => group && group.kind === "cluster";

// In a cluster, clicks mark faces that do NOT belong; everywhere else they select
// faces that DO. Same gesture, inverted meaning, because the common case differs.
function targetFaces() {
  const ids = group.faces.map((f) => f.face_id);
  return isCluster() ? ids.filter((id) => !marked.has(id)) : ids.filter((id) => marked.has(id));
}

async function submit(label, personId) {
  const ids = targetFaces();
  if (!ids.length) { toast("Nothing selected"); return; }

  const entries = ids.map((id) => ({ face_id: id, label, person_id: personId || null }));
  const result = await api("/api/label", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ entries }),
  });

  if (result.error) { toast("Error: " + result.error); return; }
  lastBatch = ids;
  toast(`${result.written} → ${personId || label}`);
  await load();
}

async function assignPerson() {
  const { persons, next_person_id } = await api("/api/persons");
  const known = persons.map((p) => `${p.person_id} (${p.count})`).join(", ") || "none yet";
  const answer = prompt(
    `Person id for ${targetFaces().length} face(s).\\n\\nKnown: ${known}\\n\\n` +
    `Enter to create ${next_person_id}, or type an existing id to merge.`,
    next_person_id
  );
  if (answer === null) return;
  await submit("person", answer.trim() || next_person_id);
}

document.addEventListener("keydown", async (event) => {
  if (!group || group.kind === "done") return;
  const key = event.key.toLowerCase();

  if (event.key === "Enter") { event.preventDefault(); await assignPerson(); }
  else if (key === "a") { group.faces.forEach((f) => marked.add(f.face_id)); render(); }
  else if (key === "d") { marked.clear(); render(); }
  else if (key === "n") await submit("not_of_interest");
  else if (key === "x") await submit("non_face");
  else if (key === "u") await submit("unsure");
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
