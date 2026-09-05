# Development Log

Append-only record of what was done, why, what was decided, and what went wrong.
Newest entries at the top. Never rewrite history here — correct it with a new entry.

**Entry format**

```
## YYYY-MM-DD — <short title>
**Phase:** <n>  **Machine:** <mac|linux|kaggle|android>  **Status:** <done|in-progress|blocked|abandoned>

### Did
### Decided
### Measured
### Problems / surprises
### Next
```

---

## 2026-09-06 — Phase 1 step 1: corpus scanner

**Phase:** 1  **Machine:** mac (written and tested here; runs on linux)  **Status:** done

### Did
- `src/faceindex/store.py` — SQLite schema (`photos`, `meta`), WAL, resume helpers.
- `src/faceindex/ingest.py` — walk, classify, quick-hash, SHA-256 dedup, EXIF/GPS extraction.
- `src/faceindex/report.py` — corpus composition, era histogram, camera table, warnings.
- `scripts/scan_corpus.py` — CLI with progress bar, `--report-only`, `--no-dedup`.
- `tests/test_ingest.py` — 10 tests against a synthetic corpus generated with real JPEG/PNG bytes
  and real EXIF, so the scanner is fully validated on the Mac before touching the real library.
- README gained a copy-pasteable Linux section.

### Decided
- **Videos rejected on extension before any I/O.** Confirmed present in the corpus. Opening them
  would read gigabytes for nothing.
- **Nothing decodes pixels in this stage.** `Image.open` parses the header only, so cost is
  traversal plus ~128 KB per candidate file rather than a full decode of 134 GB.
- **New `forwarded` kind** for messaging-app images (`-WA####` filenames, `WhatsApp/` folders).
  Directly supports the forwarded-image slice PLAN.md Phase 1 calls for, and keeps thousands of
  stranger faces out of the main sample.
- **Documents/receipts/memes are deliberately *not* classified.** Detecting them reliably needs a
  model. They fall through as `photo` and are harmless: they simply yield no faces. Better than a
  heuristic that silently discards real photos.
- **RAW files recorded as `raw` and skipped**, with a report warning, so the loss is visible rather
  than silent.
- Canonical copy of a duplicate group is the **oldest mtime**, ties broken on path, so the choice is
  deterministic across runs and machines.

### Measured
End-to-end smoke test on a synthetic 16-file corpus (path containing a space, 2 camera eras,
2 duplicates, 1 video, 1 screenshot, 1 forwarded): 13 photo / 1 forwarded / 1 screenshot / 1 video,
2 duplicates resolved, **12 unique face candidates**. Era histogram and camera table both correct.

### Problems / surprises
- **A test caught a real fixture bug that would have invalidated the counts.** Two fixture images
  generated with the same colour and no EXIF produced byte-identical JPEGs, so the deduplicator
  correctly collapsed them — which is right behaviour but wrong intent. Fixture images now use
  distinct colours. Worth remembering: *identical pixels give identical bytes*, and dedup runs
  across different `kind` values.
- mypy rejected `PhotoRecord(**base, ...)`; the dict widened to `dict[str, object]`. Rewritten with
  explicit arguments plus a small `rejected()` helper. Type safety kept, no `Any` escape hatch.
- Partial-download and hidden-directory handling both needed explicit tests; `.hidden` folders and
  `@eaDir`/`__MACOSX` are now skipped.

### Verified
`ruff check`, `ruff format --check`, `mypy` (6 files), `pytest` (21 tests) all green.
Lock file audited for macOS-only packages: none. `librt` is the mypyc runtime, cross-platform.

### Next
- Run the scanner on the real 134 GB library on Linux and record the real composition numbers here.
  Those numbers replace the guessed "18k photos / 30k faces" and drive the gold-set strata.
- Then `scripts/build_face_pool.py`: detect + align + attributes over the face candidates.

---

## 2026-09-06 — Phase 0 complete

**Phase:** 0  **Machine:** mac (M2 Air, arm64)  **Status:** done

### Did
- Created conda env `fca` on **Python 3.11.15** (base was 3.14, no wheels for the ML stack).
- Installed and froze the dependency set to `requirements.lock.txt` (50 pinned packages).
- Scaffolded: `.gitignore`, `pyproject.toml`, `README.md`, `LICENSE`, `DEVLOG.md`,
  `configs/baseline.yaml`, `src/faceindex/{__init__,paths,cli}.py`,
  `scripts/download_models.py`, `tests/test_phase0_setup.py`.
- Downloaded and verified 4 ONNX models; recorded digests in `models/models.lock.json` (committed).
- `ruff check`, `ruff format --check`, `mypy`, `pytest` (11 tests) all green.

### Decided
- Python 3.11; `sklearn.cluster.HDBSCAN` over standalone `hdbscan`; plain `Pillow` over `pillow-simd`;
  `opencv-python-headless`. Recorded as decisions 11–14 in PLAN.md §6.
- Model checksums are **recorded on first download** rather than hardcoded, because upstream has re-cut
  archives before. The committed lock file is what makes the two machines provably identical.

### Measured

| Model | File | Size | Purpose |
|---|---|---|---|
| SCRFD-500M-KPS | `buffalo_sc/det_500m.onnx` | 2.5 MB | detector, Phases 1–2 |
| MobileFaceNet | `buffalo_sc/w600k_mbf.onnx` | 13.6 MB | embedder, Phases 1–2 |
| SCRFD-10G-KPS | `buffalo_l/det_10g.onnx` | 16.9 MB | detector, Phase 5 |
| ResNet50 | `buffalo_l/w600k_r50.onnx` | 174.4 MB | embedder, Phase 5 |

Verified by loading each under `CPUExecutionProvider`:
- Detectors expose **9 outputs** = 3 strides × (score, bbox, 5 keypoints). Input `[1,3,?,?]`, dynamic H/W.
- Embedders take `[N,3,112,112]` and return `[1,512]` float32.

Resolved versions: `onnxruntime 1.22.1`, `numpy 2.2.6`, `opencv 4.12.0`, `Pillow 11.3.0`,
`pillow-heif 1.0.0`, `scikit-learn 1.7.2`, `scipy 1.16.3`, `pandas 2.3.3`.

### Problems / surprises
- **`gnu.org` is unreachable from this network** (connection timeout). Fetched the AGPL-3.0 text from
  `raw.githubusercontent.com` instead — 34,020 bytes, correct. GitHub release URLs work fine, which is what
  the model downloader needs.
- **`.gitignore` bug caught before committing.** A blanket `models/` rule would also have excluded
  `models/models.lock.json`, silently destroying the whole cross-machine verification scheme. Changed to
  `models/*` plus `!models/models.lock.json`. Verified with `git check-ignore`.
- `ruff` flagged a partial-download handling weakness in `download_models.py`. Rewritten to stream into a
  sibling `.part` file and `Path.replace()` it into position — atomic, so a crash or network drop can never
  leave a truncated model that would then be checksummed as if it were valid.
- `test1.py` (pre-project scratch file, `print("Welcome to test")`) is tracked in git. **Left untouched**;
  excluded from ruff via `extend-exclude` pending a decision to delete it.

### Verified
- `python scripts/download_models.py --verify-only` re-verifies all 4 models offline.
- `git add -An --dry-run` confirms no `.onnx` and no `data/` file can be staged.

### Next
- Phase 1. Corpus is ~134 GB on the Linux box.
- Write `scripts/scan_corpus.py`: walk, quick-hash, dedup, classify, drop screenshots/documents/**videos**,
  extract EXIF. Must be resumable and stream-only (peak RSS < 1.5 GB).

---

## 2026-09-06 — Phase 0 started: environment and repo scaffold

**Phase:** 0  **Machine:** mac (M2 Air, arm64)  **Status:** in-progress

### Did
- Created conda env `fca` on **Python 3.11** (base was 3.14, too new for the ONNX Runtime / scikit-learn stack).
- Scaffolded repo: `.gitignore`, `pyproject.toml`, `README.md`, `LICENSE` (AGPL-3.0), `DEVLOG.md`.
- Created `src/faceindex/` package skeleton and `scripts/download_models.py`.

### Decided
- **Python 3.11**, not 3.12/3.13/3.14 — widest wheel availability for `onnxruntime` and friends.
- **`scikit-learn`'s built-in `HDBSCAN`** instead of the standalone `hdbscan` package. Removes a fragile
  C-extension build dependency that historically breaks on ARM and on constrained Linux boxes.
- **Plain `Pillow`, not `pillow-simd`.** `pillow-simd` does not build cleanly on arm64, and using different
  JPEG decoders on the two machines would silently change pixels, and therefore embeddings (PLAN.md §3).
- **`opencv-python-headless`**, not full `opencv-python` — no GUI needed, much smaller, no Qt dependency.
- Narrow version ranges in `pyproject.toml`; exact resolved versions frozen to `requirements.lock.txt`
  so the Mac and the Linux box run identical code.

### Measured
- Nothing yet. Phase 0 produces no numbers.

### Problems / surprises
- Corpus is **~134 GB**, considerably larger than the ~70 GB assumed while planning. See the resource note
  below; this changes runtime estimates but **no quality settings were reduced**.

### Next
- Install dependencies into `fca`, freeze `requirements.lock.txt`.
- Populate `scripts/download_models.py` with verified URLs + SHA-256 checksums.
- Confirm models download and load on both machines.

---

## Resource notes and quality decisions

Any place where resources pushed against quality is recorded here and mirrored into
**PLAN.md §9 Quality Compromise Register**.

### 2026-09-06 — 134 GB corpus (up from the ~70 GB planning assumption)

**No quality settings were reduced.** The consequence is *time*, not accuracy:
the one-time detect+align pass on the Linux i3 will take proportionally longer.

Flagged for attention:
- If ~134 GB is still roughly 18k photos, the average file is ~7.4 MB, implying high-resolution images
  (and possibly **videos**, which are out of scope for now and must be filtered out during scan).
- Larger source images make `Image.draft()` DCT downscaling *more* valuable, not less — it stays enabled.
- The 1280px decode cap (register entry **C4**) is a genuine, already-logged compromise. It stays in place
  for the Phase 1 bootstrap only, and Phase 5 reverts the server path to full resolution. Do not let the
  larger corpus become an excuse to keep it.
- Derived artifact sizes scale with face count, not GB, so the ~1 GB Mac transfer estimate should hold.
