# CONTEXT — read this first

Briefing for a fresh AI coding session or a new machine. Everything here is fact, measured
on the real corpus, not assumption.

---

## 1. What this project is

An open-source, offline, privacy-preserving photo organiser that groups a personal photo
library by person. Server/desktop indexer first, Android client later.

**It is unsupervised face *clustering*, not face *identification*.** No enrolment step, no
database of known people, no 1:N lookup. New people need zero training — they simply form a
new cluster. This is a deliberate privacy design choice, not an implementation shortcut.

Non-commercial: the InsightFace weights forbid commercial use, so the project is AGPL-3.0
and weights are downloaded, never committed.

---

## 2. Documents, in reading order

| File | Purpose |
|---|---|
| **CONTEXT.md** | This file. Orientation. |
| **PLAN.md** | The plan of record. 14 phases, decision log, risk register, quality compromise register. |
| **DEVLOG.md** | Append-only work log. Newest first. What was done, decided, measured, and what went wrong. |
| **README.md** | Install and run instructions, including the Linux quickstart. |

If a question is "why is it built this way?", the answer is in **PLAN.md section 6 (Decision
Log)**. If it is "what happened when we tried it?", the answer is in **DEVLOG.md**.

---

## 3. Where things run

| Machine | Role |
|---|---|
| **Mac M2 Air** | All code is written here. Fast experiment loop. |
| **Linux i3, 8 GB** | The photo library is attached here (external USB drive). Runs the passes over originals. |
| **Kaggle** | Phase 10 only (optional model distillation). Never for corpus processing. |
| **Android** | Photo source now; compute target from Phase 11. |

Code syncs by GitHub. Derived data syncs out-of-band (USB/rsync) and is **never committed**.

**Environment on both machines:** conda env `fca`, Python 3.11, exact versions in
`requirements.lock.txt`. They must match — differing library versions change decoded pixels,
which changes embeddings, which makes experiment results incomparable.

---

## 4. Current state (2026-09-06)

**Phase 0 complete. Phase 1 in progress.**

| Step | Status |
|---|---|
| Repo, env, model download + checksum lock | done |
| Corpus scan, dedup, classification, date recovery | done |
| Face pool: detect + align + attributes | **done** |
| Embeddings | next |
| Bootstrap clustering | not started |
| Stratified sampler | not started |
| Labelling UI | not started |
| Evaluation harness (metrics) | not started |

### Measured corpus

`/media/siddharth/Elements/B/Photos Timeline`, 22,027 files, 62.3 GB.

| Kind | Files | Size |
|---|---|---|
| photo | 13,345 | 29.8 GB |
| forwarded (messaging apps) | 6,080 | 2.8 GB |
| video (skipped, never read) | 1,321 | 25.8 GB |
| screenshot | 989 | 0.5 GB |
| audio | 236 | 2.1 GB |
| unsupported / unreadable / archive / tiny | 56 | 1.4 GB |

- **18,366 unique face candidates** after removing 1,215 duplicates
- Dates: 94.8% resolved (61.3% EXIF capture tags, 32.0% filenames, 1.5% EXIF modified)
- Era span **2007–2025**, peaks at 2016 (2,906) and 2022 (2,561), plus 2024 (2,773)
- 15+ camera models, from a 2006 Nokia 6233 to a Canon EOS 1500D

### Measured face pool

**63,878 faces from 18,366 photos = 3.48 faces/photo.** 102 minutes on the i3, 3 errors.

| Size (inter-ocular px) | Share |
|---|---|
| tiny <20 | 33.8% |
| small 20–40 | 25.4% |
| medium 40–80 | 22.1% |
| large >80 | 18.7% |

| Pose | Share |
|---|---|
| frontal <15 deg | 48.1% |
| semi 15–45 | 31.0% |
| profile >45 | 20.9% |

Faces per photo: `photo` 3.41, `forwarded` 3.64.

**Open question:** 34% of faces are under 20 px inter-ocular. Are they genuine small faces in
crowd shots, or detector false positives? Run `scripts/contact_sheet.py --bucket tiny` and
look before setting any gating threshold.

---

## 5. Commands

```bash
conda activate fca

python scripts/download_models.py                  # weights + test asset, checksum-verified
python scripts/scan_corpus.py --root "<library>"   # cheap; no pixels decoded
python scripts/scan_corpus.py --report-only        # re-print without rescanning
python scripts/scan_corpus.py --diagnose           # reject buckets, largest folders
python scripts/scan_corpus.py --inspect "College"  # what happened to these files?
python scripts/build_face_pool.py                  # the expensive pass; resumable
python scripts/build_face_pool.py --stats          # progress and distributions
python scripts/contact_sheet.py --bucket tiny      # look at the crops

ruff check . && ruff format --check . && mypy && pytest -q
```

All long passes are **resumable**: interrupt and rerun the identical command.

---

## 6. Rules this project follows

Violating these silently corrupts results, so they are not stylistic preferences.

1. **Measure before optimising.** No change lands without a number from the eval harness.
2. **Quality compromises are explicit and measured**, logged in PLAN.md section 9 with a
   measured cost. Never "probably fine". The desktop path takes no shortcuts; the phone is a
   measured port.
3. **Never commit `data/` or `models/`.** Face crops and embeddings are biometric data;
   weights are non-redistributable. `models/models.lock.json` is the one deliberate exception
   and *must* be committed.
4. **CPU execution provider only** for anything producing stored embeddings. CoreML and CUDA
   do not agree bit-for-bit with it, and results must be comparable across machines.
5. **Gold-set labels are evaluation ground truth only.** They never enter the pipeline. There
   is no training in the core track; PLAN.md phases 9 and 10 are default-skip.
6. **Alignment correctness is critical and fails silently.** Preprocessing is
   `(pixel - 127.5) / 128.0`, RGB, NCHW — not `x/255`. Verify by geometry, not by absence of
   errors.
7. **One bad file must never abort a multi-hour run.** Record the error, continue.
8. **Verify edits by their effect, not by a tool reporting success.** A bulk edit silently
   deleted a CLI argument once; it was caught by diffing before commit.

---

## 7. Things already decided — do not relitigate without evidence

| Decision | Reason |
|---|---|
| Clustering, not identification | Privacy design choice |
| No model training in the core track | Cannot beat ArcFace with ~30 identities; gains live in the pipeline |
| SCRFD-10G for the face pool, not 500M | Detection is the only stage needing originals, so a missed face is permanently absent from the gold set. Embedding can be redone from crops for free. |
| Detector input 640, not 1024 | Measured: 1024 found the same faces with lower scores at 2.2x cost |
| Forwarded/WhatsApp images are first-class | 33% of candidates; the primary channel for family photos in this library |
| `sklearn.cluster.HDBSCAN`, not standalone `hdbscan` | Avoids a fragile C-extension build |
| Plain Pillow, never `pillow-simd` | Does not build on arm64; differing decoders change pixels |
| EXIF tag 306 ranks *below* filename dates | It is a modification time; bulk edits rewrite it |

---

## 8. Known risks being tracked

- **Quality gating may be biased against forwarded images.** They are ~5x smaller
  (0.46 MB vs 2.2 MB average), so their faces are smaller in pixels. Gating on absolute size
  could silently delete the user's party photos. Track gating rate per kind; switch to
  relative face size if the bias is real.
- **Beauty-filtered photos** — one folder holds 2,163 images (~12% of candidates) from a
  retouching app. Those filters alter face geometry, which is what the embedding encodes.
- Cross-age drift, identical twins, infants: accepted as hard, mitigated by correction UX.

---

## 9. Prompt to start a session elsewhere

> This is an offline face-clustering project for a personal photo library. Read `CONTEXT.md`,
> then `PLAN.md` (plan of record, decision log, risk register) and `DEVLOG.md` (work log,
> newest first) before proposing anything. Follow the rules in CONTEXT.md section 6 —
> especially: never commit `data/` or `models/`, flag any quality compromise explicitly, and
> measure before optimising. Tell me the current phase and the next concrete step.
