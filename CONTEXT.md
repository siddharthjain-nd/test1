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
| **docs/theory.html** | How the method works and what every measurement meant, with diagrams. One self-contained file, no network. Open it in a browser. |

If a question is "why is it built this way?", the answer is in **PLAN.md section 6 (Decision
Log)**. If it is "what happened when we tried it?", the answer is in **DEVLOG.md**. If it is
"how does any of this actually work?", open **docs/theory.html**.

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

## 4. Current state (2026-09-29)

**Phases 0, 1 and 2 are complete and measured. Phase 7 (review UI) is built to v3 and
working. One question is open and it gates everything else.**

| Step | Status |
|---|---|
| Repo, env, model download + checksum lock | done |
| Corpus scan, dedup, classification, date recovery | done |
| Face pool: detect + align + attributes | done — 63,878 faces from 18,366 files |
| Embeddings | done for **three** models: MobileFaceNet, ResNet50, ResNet100 |
| Gold set `data/gold/labels.csv` | done — 2,122 labels, 1,499 identity faces, 124 people |
| Held-out split `data/gold/split.csv` | frozen — 100 tune / 24 reserved identities |
| Evaluation harness, `run_experiment.py`, results table | done |
| Algorithm + model comparison | done — see PLAN.md decisions 39 and 40 |
| Threshold chosen at full scale | done — PLAN.md decision 41 |
| **Held-out score** | **done, spent once: BCubed F1 0.9394** |
| Review UI (Phase 7) | v0–v3 built, 112 tests, verified end to end |
| Review workload — how many piles must be named | **open, unmeasured** |

### The headline number

**ResNet50 + connected components at cosine 0.51.** On the 24 reserved identities, never used
for tuning: BCubed **precision 0.976, recall 0.906, F1 0.9394**, 27 piles for 24 people. On the
tuning half, 0.9606. The gap is within the spread of a 24-person score and traces to one bucket
of 51 faces.

The holdout is **spent**. Any further tuning goes back to the tuning half; a fresh measurement
would need newly reserved identities.

### The one open question

Pile sizes are extremely lopsided — 106 piles hold 26,358 faces. If a few hundred names cover
most of a library, the review design holds. If it takes thousands, it does not, and the
clustering must join more per pile or poor faces must be kept out of the queue.

`python scripts/coverage_curve.py` answers it and has not been run. **It outranks any
remaining threshold work.**

### Errors that remain, consistent across both halves of the gold set

Bad-quality faces 0.515, tiny faces 0.89, marginal quality 0.91, profile 0.89, middle era 0.90.
Every one is about the face being hard to read. Slices that *reverse* between the halves
(photos-per-person, beauty-filtered) are noise on 221 faces and must not be read as findings.

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

# Gold set, in order. Each prints what to run next.
python scripts/verify_embedding_parity.py          # run once per machine, before labelling
python scripts/embed_faces.py --model w600k_r50.onnx   # resumable; --stats to check coverage
python scripts/bootstrap_cluster.py                # throwaway pre-grouping for labelling
python scripts/sample_gold_set.py                  # stratified sample + composition report
python scripts/label_gold_set.py                   # the only manual step; opens a browser
python scripts/export_gold_set.py                  # freeze to data/gold/labels.csv
python scripts/make_holdout.py                     # freeze data/gold/split.csv

# Experiments and the measurements that chose the settings.
python scripts/run_experiment.py --results         # every run ever recorded
python scripts/run_experiment.py --model w600k_r50.onnx --algorithm components \
       --similarity 0.51 --split tune --label x    # one scored run
python scripts/diagnose_threshold.py --model w600k_r50.onnx   # where welding starts, no labels
python scripts/inspect_pile.py --model w600k_r50.onnx --similarity 0.51 --contact-sheet
python scripts/preflight.py --model w600k_r50.onnx --algorithm components --similarity 0.51
python scripts/estimate_error_bar.py --model w600k_r50.onnx --similarity 0.51
python scripts/per_person_report.py --model w600k_r50.onnx --similarity 0.51 --split holdout

# Review UI (Phase 7).
python scripts/build_review_index.py --model w600k_r50.onnx --similarity 0.51
python scripts/review_faces.py                     # localhost:8766; --merge-floor to tune
python scripts/tune_merge_threshold.py             # what similarity a suggestion should need
python scripts/coverage_curve.py                   # THE OPEN QUESTION — not yet run

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
   `(pixel - 127.5) / 127.5`, RGB, NCHW — not `x/255`. Verify by geometry, not by absence of
   errors. *(Corrected 2026-09-11: this rule previously said `/128.0`. InsightFace's reference
   `ArcFaceONNX` uses 127.5, and the two were measured to differ by 3.1e-06 cosine — 300x below
   the parity tolerance, so the constant is not what will bite you. Channel order and layout are.
   Run `scripts/verify_embedding_parity.py` before freezing a gold set.)*
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
| **ResNet50 + connected components @ cosine 0.51** | Measured against three models and four algorithms; PLAN.md decisions 39–41 |
| **The clustering threshold is found per collection, never shipped as a constant** | One bad edge welds two people, and the risk scales with pair count. `diagnose_threshold.py` finds it without labels |
| **Human decisions are keyed on face ids, never pile ids** | A pile id is meaningless after re-clustering; a face id comes from detection and is permanent. This is what lets a better model land without costing naming work |
| **No tuning knob is ever exposed to a user** | A photo app cannot ask someone to set a cosine threshold. Flags like `--merge-floor` are development tools |
| Agglomerative clustering is unusable here | Measured 6.1 GB at 63,878 faces on an 8 GB machine |

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

Paste this verbatim:

> This is an offline face-clustering project for a personal photo library, running on my Linux
> laptop. Read `CONTEXT.md` first, then `PLAN.md` (plan of record, decision log §6, risk
> register) and `DEVLOG.md` (work log, newest entry first). `docs/theory.html` explains the
> method and the measurements with diagrams — open it if you want the reasoning behind the
> numbers.
>
> Phases 0–2 are complete and measured; the held-out score is BCubed F1 0.9394 with ResNet50 +
> connected components at cosine 0.51, and the holdout is spent. The review UI (Phase 7) is
> built to v3. One question is open and gates the rest: how many piles a person must name to
> cover their library — `scripts/coverage_curve.py` answers it and has not been run.
>
> Follow CONTEXT.md §6 (rules) and §10 (how I work). In particular: **do not run `git commit`
> or `git push`** — leave changes in the working tree and tell me what changed. Do not
> relitigate anything in §7 without a measurement.
>
> Tell me the current state and the next concrete step.

---

## 10. How I work — agreements that are not negotiable

These came from corrections during earlier sessions. They are written here so any machine and
any session behaves the same way.

1. **Never `git commit`, never `git push`.** Leave changes in the working tree and say plainly
   which files changed. Siddharth decides when work travels between his two machines — a push
   silently changes what he is running mid-experiment.
2. **Every script prints its own verdict in plain English** — passed, failed, or in between —
   next to what was expected. He should never have to paste a table back and be told what it
   means. Exit codes match the verdict so a chained command stops on failure.
3. **Every command handed over states three things**: why it is being run, what its output will
   look like and roughly how long, and what result means "good, carry on" versus "stop and
   report back". Prefer verification he can perform himself over reassurance.
4. **Anything slower than a few seconds shows progress** — a bar with a count and an ETA where
   work divides into units, or at minimum a live elapsed time and the stage currently running.
5. **Compute time is free.** The machine is idle most of the day. Never recommend a cheaper
   model, coarser setting or smaller sample *because it is faster*. "Four hours, run it
   overnight" is a fine answer. Memory is a real constraint; the machine has 8 GB.
6. **Judge designs by whether they work for any gallery**, not by what scores best on this
   corpus. Prefer criteria that are relative over absolute constants. Say plainly when a number
   was tuned on this library and should not ship as a default.
7. **Say when a choice is arbitrary.** If two options differ by less than the noise, say so
   rather than writing a justification. Where running both is cheap, do that instead of arguing.
8. **Exhaust the data already in hand before writing another diagnostic.** Compare existing
   runs against each other first — a slice that reverses between two splits is noise, one that
   holds is a finding. Only then decide whether new code is warranted, and say whether it is
   needed for the decision or merely nice to have.
9. **Plain language.** No invented vocabulary, no dense metric-speak. Short sentences. If a
   term is needed, define it once.
10. **Test every code path before handing it over.** Blind string edits have silently failed
    here more than once; a CLI flag that was never run is not shipped. See §6 rule 8.
