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

## 2026-09-06 — Correction: forwarded images are first-class; era gap explained

**Phase:** 1  **Machine:** mac  **Status:** done

### Corrected a planning error
The plan said forwarded/messaging-app images should be kept out of the gold set except a token
slice. **Wrong for this library.** WhatsApp is the *primary* channel through which family and event
photos arrive here — 6,080 of 18,366 candidates (33%). They are now sampled into the gold set at
their real share.

The code was already correct: `FACE_CANDIDATE_KINDS = {photo, forwarded}`, so nothing was ever
excluded from face detection. Only the plan was wrong.

### New risk identified from the measured data

| Kind | Files | Size | Average |
|---|---|---|---|
| `photo` | 13,345 | 29.8 GB | 2.2 MB |
| `forwarded` | 6,080 | 2.8 GB | **0.46 MB** |

Messaging apps recompress and downscale, so forwarded images are ~5x smaller, meaning **smaller
faces in pixels**. Phase 3 quality gating on *absolute* face size would therefore reject them at a
higher rate than camera originals and quietly delete the user's party photos from their people
albums. Added to the risk register: report gating rate and `f1_by_slice` separately for
`forwarded`, and switch to *relative* face size (fraction of image height) if the bias is real.

### Investigated: "2024 shows 353 photos but a college folder has ~2,751"
Not a report bug. The era table sums to 11,648 dated + 6,718 undated = 18,366, exactly the
candidate count, so it is internally consistent. The 2024 photos were in the **undated** bucket
because WhatsApp strips EXIF — and that report was produced *before* the date-recovery code existed.

Verified by simulation: a folder of 30 undated `IMG-2024MMDD-WA####.jpg` files plus 5
arbitrarily-named ones now resolves to **30 dated from filename, 5 dated from folder**, all 2024.
Previously all 35 would have been undated.

### Did
- `--inspect SUBSTRING`: explains what happened to every file whose path matches — by kind, by
  year and date source, with duplicate counts and sample rows. Turns "where did my photos go?"
  into a fact rather than an inference.
- Largest-folders table added to `--diagnose`, flagging folders that are majority-undated in red.

### Next
- Run backfill and `--inspect` on the real corpus to confirm 2024/2025 populate.
- Identify the 282 unsupported files (3.4 GB).
- Then `scripts/build_face_pool.py`.

---

## 2026-09-06 — First real corpus scan + date recovery

**Phase:** 1  **Machine:** linux (scan) / mac (code)  **Status:** done

### Measured — the real corpus

Path: `/media/siddharth/Elements/B/Photos Timeline` (external USB drive).
22,027 files, 62.3 GB, scanned in **12m13s at ~30 files/sec**.

| Kind | Files | Size | Share |
|---|---|---|---|
| photo | 13,345 | 29.8 GB | 60.6% |
| forwarded | 6,080 | 2.8 GB | 27.6% |
| video | 1,321 | 25.8 GB | 6.0% |
| screenshot | 989 | 0.5 GB | 4.5% |
| unsupported | 282 | 3.4 GB | 1.3% |
| unreadable | 9 | — | — |
| tiny | 1 | — | — |

- Duplicates: **1,215** (0.9 GB redundant), 5.5% — normal for periodic backups.
- **Unique face candidates: 18,366.** Close to the 18k the plan assumed, so prior estimates hold.
- Era span **2010–2025** with two clear peaks: 2016 (2,752) and 2022 (2,138).
- **15+ camera models**: Xiaomi Redmi 3S/4/HM1S, realme narzo 20 / GT Neo2, Canon EOS 1500D,
  Nokia 6233, Sony DSC-WX80, Samsung GT-P3100, Apple iPhone 13, Intex, YU, alps.

### Observations that change planning
- **Videos were 25.8 GB — 41% of all bytes — and were never read.** Extension-based rejection
  paid for itself outright on the first run.
- **Forwarded images are 27.6% of the library**, higher than assumed. Isolating them as their own
  kind keeps thousands of stranger faces out of the gold set.
- **Camera diversity is unusually wide**, from a 2006 Nokia feature phone to a DSLR. Image quality
  variance will be large, which raises the importance of Phase 3 quality gating and makes the
  gold set's quality strata easy to fill.
- Two population peaks nine years apart is close to ideal for the **cross-era identity**
  requirement, which is the hardest thing the gold set has to measure.
- Corpus is on an **external USB drive**. Fine for a header-only scan; the Phase 1 detect pass will
  read ~32 GB of pixel data from it, so I/O may matter there.

### Problem found: 37% of candidates had no date
6,718 undated vs 6,080 forwarded is not a coincidence — messaging apps strip EXIF. Without dates,
era stratification and the Phase 4 time prior would run on 63% of the library.

### Did
- Added `taken_at_source` column (schema v2, with an additive migration on open).
- `date_from_filename()` — recovers dates from `IMG-20230115-WA0001`, `IMG_20230115_143022`,
  `PXL_20220704_101530123`, `photo_2021-12-25_18-45-01`, `Screenshot_2024-03-09-07-01-59`.
  Validates plausibility, so `20161131` (November has 30 days) and `20180229` are both rejected.
- `year_from_folder()` — coarse fallback for folders like `2016 Goa/`.
- `backfill_dates()` — database-only, reads no files, idempotent, runs in seconds.
- `--diagnose` flag: breaks down unsupported extensions, unreadable reasons, and image formats.
- Report gained a **date provenance** table so the trustworthiness of every date is visible.
- 21 new tests (42 total).

### Decided
- **mtime fallback is opt-in (`--use-mtime`), not default.** Copying a library resets modification
  times; a wrong date is worse than no date because it feeds a false signal into the Phase 4 time
  prior. Provenance is recorded either way so the decision stays measurable.
- Priority is strictly **exif > filename > folder > mtime**, and EXIF is never overwritten.

### Problems / surprises
- A test caught the Pixel filename format: `PXL_20220704_101530123.jpg` appends milliseconds, so
  the trailing `(?!\d)` in the time pattern rejected an otherwise valid match.

### Next
- Rerun the scan on Linux to populate dates, then `--diagnose` to identify the 282 unsupported
  files (3.4 GB, ~12 MB average — too large to be junk; likely a video format not yet listed).
- Then `scripts/build_face_pool.py`.

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
