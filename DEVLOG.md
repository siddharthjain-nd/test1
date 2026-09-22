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

## 2026-09-21 — Gold set frozen; evaluation harness built

**Phase:** 1  **Machine:** linux (labels) / mac (code)  **Status:** Phase 1 complete

### The gold set exists

**2,122 judged faces · 1,499 named across 124 identities.** Larger and harder than the plan
asked for: 124 people against a 25-35 target means far more chances to confuse similar faces,
so the test discriminates better rather than worse.

| Label | Faces | Share |
|---|---|---|
| person | 1,499 | 70.6% |
| unsure | 291 | 13.7% |
| not_of_interest | 174 | 8.2% |
| non_face | 158 | 7.4% |

Time spans: 7 people across 10-14 years, 11 across 5-9. Widest are `shanu` 2007-2021 (14y),
`maa` 2011-2024 (13y), `me` 2013-2025 (12y).

### Measured: where human readability collapses

| Face size | Named | Unreadable |
|---|---|---|
| large | 385 / 420 | 3.3% |
| medium | 610 / 735 | 3.3% |
| small | 361 / 550 | 16.4% |
| tiny | 143 / 417 | 39.1% |

Flat at 3.3% down to 40px, then a cliff. **This is Phase 3's gating threshold, measured
rather than guessed** — a clusterer has no business forming identities from faces a human
cannot read. It came free from pressing `u` honestly on unreadable crops.

### Corrected: the cross-era check was measuring the wrong quantity

The check required a face in the `oldest` bucket *and* one in `recent`. Those buckets exist
to spread the **sample** across time; reusing them as a measure of one person's time span was
a category error. It rejected a 2010-to-2022 person at twelve years while accepting a
2016-to-2024 one at eight.

Replaced with the actual gap between a person's first and last photo. The user spotted this
-- they asked why middle-era photos were excluded when 6-7 years also shows real ageing.
Their instinct was right, though the fix was not "also count middle": buckets were simply the
wrong unit. The requirement now splits in two, since these are different tests:
**>=10 people spanning 5+ years**, and **>=5 of those spanning 10+**.

Under the corrected measure the set passes on evidence that was there all along.

### Declined: sourcing recent photos of old friends

Proposed as a way to manufacture cross-era pairs. Refused on three grounds: it would measure
a corpus that is not the library; photos obtained deliberately are near-always sharp frontal
portraits, making the hardest slice artificially easy; and it would require rescanning and
re-embedding underneath finished labels. Also out of keeping with the project's stated intent
-- photos a friend knowingly sends are the library, photos taken from social media are not.

### Built: the evaluation harness

`src/faceindex/eval/` with `metrics.py` and `goldset.py`, plus `scripts/run_experiment.py`.
Pairwise and BCubed P/R/F1, NMI, ARI, cluster counts, noise share, per-slice breakdowns, and
one row appended per run to `data/results/results.csv`.

Four label kinds are treated differently, and the reasoning matters more than the code:

- `person_N` -- the only faces scored for identity.
- `not_of_interest` -- **excluded**. The system may group strangers or call them noise;
  neither is wrong. Scoring them as one shared class would mark it wrong for correctly
  noticing that two photographs of the same passer-by are the same person.
- `non_face` -- excluded from identity scoring, counted separately as **contamination**.
  A poster filed into someone's album is invisible to precision and recall, because the
  poster carries no identity label, yet a user would notice it instantly.
- `unsure` -- excluded entirely, as promised at labelling time.

Noise in the *prediction* is expanded into singletons before any metric runs. Left as a
shared `-1`, every pair of ungrouped faces would count as a successful grouping, and a system
that clustered nothing would score perfect recall. There is a test for exactly that.

Slices use BCubed rather than pairwise because BCubed is defined per face, so averaging it
over a subset is meaningful; a pair can straddle two slices, which makes "pairwise F1 on
profile faces" ambiguous.

### Problems / surprises

A test asserting `bcubed_precision < pairwise_precision` failed, and the code was right: on
that construction BCubed precision is 0.9623 against pairwise 0.9608. The property being
demonstrated -- that pairwise is dominated by whoever appears most often -- shows up in
*recall*, since pairwise weights a person by their pairs, which grows with the square of
their photo count. Rewritten with 50 faces of one person and 4 of another: pairwise recall
0.995, BCubed recall 0.944.

Verified end to end on a synthetic corpus of 12 planted identities plus 20 junk faces: all 12
recovered, contamination 0%, slices and the results table correct.

**132 tests. ruff, ruff format, mypy all green.**

### Next
- Run the baseline on the real corpus. **Expect a mediocre number** -- it is a baseline, not
  a result, and its only job is to be what later changes must beat.
- Then Phase 2 proper, and Phase 3 gating against the 40px cliff measured above.

---

## 2026-09-20 — Clustering is good; the sampler was not

**Phase:** 1  **Machine:** linux  **Status:** fixed, resample required

### Measured — the bootstrap clustering is far better than assumed

`diagnose_clusters.py --sampled-only`, on the 285 sampled clusters:

| Tightness | Clusters | Share | Median face |
|---|---|---|---|
| tight >=0.55 | 239 | 83.9% | 61 px |
| fair 0.40-0.55 | 40 | 14.0% | 42 px |
| loose 0.30-0.40 | 4 | 1.4% | 16 px |
| junk <0.30 | 2 | 0.7% | 7 px |

Two unrelated faces baseline at 0.086. **98% of clusters are "probably one person" or
better; six are doubtful.** Junk clusters have a median face of 7 px against 61 px in the
tight ones, so the few bad ones are exactly the unreadable-face effect predicted — but they
are a rounding error, not the story.

A working hypothesis that mixed piles came from chaining and degenerate embeddings was
therefore **wrong for this corpus**. Worth recording: the explanation was plausible, fluent,
and not true. The measurement is what settled it.

### The actual bug: the sample was 63% noise

`label_gold_set.py --queue` after ~531 faces showed **no multi-face piles left at all** —
138 singles plus 1,131 noise faces, 70% of the sample, every one judged individually.

Cause: noise is exempt from the per-person caps, which is correct in itself — noise is not a
person. But the caps cut clustered faces from 37,745 to ~15,644 while leaving all 26,133
noise faces untouched, so **noise went from 41% of the pool to 63% of what the greedy fill
was choosing from**, and a proportional sampler drew accordingly. `noise_review_share` was
set to 0.10 and produced 0.63.

The slow labelling was the visible symptom. The real damage was the composition: unclustered
faces are overwhelmingly strangers and unreadable crops, so the set was heading for roughly
600 person labels against a ~1,400 target and possibly under 25 identities — **a failure that
would only have surfaced at export, after every one of the 1,800 faces had been judged.**

### Fixed
- Noise now enters only through the explicit reserve; the greedy fill skips it entirely.
- `select(..., pinned=...)` carries already-labelled faces into a new sample, so resampling
  costs none of the work done so far.
- Three regression tests, one of which immediately earned its keep: the first version of the
  pin fix still truncated with `[:total]`, which sorts by face id and silently dropped
  pinned faces that sorted late — discarding finished labelling to satisfy a target count.
  `_trim` now keeps every pinned face and lets the target give way instead.

### Next
- Re-run `sample_gold_set.py`; the 531 existing labels are preserved.
- Expect the remaining ~1,270 faces to arrive as multi-face piles again, one keystroke each.

---

## 2026-09-19 — The tiny-face question is answered: they are real

**Phase:** 1  **Machine:** linux  **Status:** done

### Answered, by looking

The open question since the face pool was built — *are the 33.8% of faces under 20px genuine
small faces, or detector false positives?* — is settled. The contact sheet was inspected.

**They are overwhelmingly genuine faces.** Poor quality, but faces: background people in group
shots, and **faces inside framed photographs** hanging in the scene.

Expectation was wrong in a useful direction. A quarter to a third were assumed to be detector
noise — fabric patterns, hands, bark — which is typical behaviour below 20px. Almost none were.
**SCRFD-10G is more precise than assumed**, which retroactively strengthens decision 21.

### What it changes

**The question shifts from "is it a face?" to "is it usable?"** A 12px background stranger is a
real face and still cannot be identified by a human or a machine, because warping 12px up to
112px is mostly invented detail. They are still gated out of cluster formation in Phase 3 — but
for a better-founded reason than "junk detections".

**The `non_face` target of ~50 is probably unreachable from detector errors.** The sampler
reserves the 50 lowest-confidence faces expecting false positives; if the detector barely makes
any, those slots return real faces. That is a finding about detector quality, not a gap.

**Tiny target raised 10% → 15%** (decision 29), with small/medium/large scaled to 24/38/23.
Changing one target backed by evidence is better practice than loosening `--tolerance`, which
would have relaxed all seven dimensions to fix the one we had actually looked at.

### New risk found: framed photographs carry the wrong date

A framed photo on a wall, photographed in 2020, may hold a face from 1985 — but the pipeline
records it as 2020, because that is the containing photo's capture time. Two consequences:

- **Phase 4's time prior** ("raise the merge bar as the date gap grows") would be fed false dates.
- **The era stratification is silently corrupted.** A genuinely cross-era face is filed as
  "recent". There is a certain irony in this given the cross-era assertion that just failed.

Labelled `non_face` (decision 30), since PLAN.md's definition already names "poster" and a framed
photograph is the same category. Logged caveat: if the face is genuinely a family member, the
clusterer grouping it with that person is reasonable behaviour that this label scores as an
error. Accepted while the population is small.

### Also verified the same day: the clusters are coherent

`--cluster-overview` was inspected. **Faces within a row are correctly grouped, and dates span
a real time range within rows.** This is the reassurance the phase most needed: accept-a-whole-
cluster labelling is viable, which is the assumption the "few hundred keystrokes" estimate rests
on entirely. Until now nobody had confirmed a single one of the 4,023 piles held one person.

### Measured: cross-era supply vs sight

| Era | Faces | Share |
|---|---|---|
| oldest | 21,742 | 34.0% |
| middle | 29,591 | 46.3% |
| recent | 11,627 | 18.2% |
| undated | 918 | 1.4% |

**Sight: 7 of 4,023 clusters (0.2%) span oldest+recent.**

Supply, by best cross-era neighbour among 8,000 recent faces: 65.3% of oldest faces reach 0.30,
29.6% reach 0.35, 13.6% reach 0.40, 5.4% reach 0.50, 2.3% reach 0.60. Median 0.319, max 0.721.

**The headline figure is inflated and the script now says so.** Taking the best of 8,000
candidates raises the score even when nothing matches, which is why the median sits at 0.319
rather than near zero — the "1,087 faces above 0.40" is largely coincidence. The *shape* is
still informative: the collapse from 65% at 0.30 to 2.3% at 0.60 means the tail is doing real
work, since a 0.60 cosine between two different people is hard to produce even taking the best
of 8,000 tries. Hundreds of genuine cross-era faces, not eleven hundred.

A chance baseline was added to the diagnostic: the same search repeated against progressively
larger candidate pools. A coincidental statistic keeps climbing with pool size; a real match was
already there at small N. Anything flat is signal.

### Decided: the cross-era check moves (decision 31)

Demoted to a **warning** in the sampler, kept as a hard check in `export_gold_set.py`. The
numbers make the case: the bootstrap sees 0.2% of what exists, because cross-age drift is
precisely what splits a person across clusters — the failure this project was built to measure.
Asking the sampler to enforce it is asking the clusterer to have already solved it.

This is not a check being deleted because it was inconvenient. Cross-era identities are still
required; they are simply discovered during labelling, when one `person_id` is assigned across
several clusters, and verified afterwards against real labels. The export failure message now
says how to fix a shortfall: label more faces of already-identified people from the era they are
missing from, rather than resampling.

**A limitation worth stating plainly.** Neighbour search finds the *easy* cross-era pairs —
people who changed little. A child photographed at 4 and at 18 scores low and is invisible to
it, while being the single most valuable face the gold set could hold. No pre-labelling signal
can find those. Only a human can, which is the whole argument for where this check belongs.

### Next
- Re-run the sampler against the revised targets. It should now pass.

---

## 2026-09-16 — Embeddings built; bootstrap clustering cost measured

**Phase:** 1  **Machine:** linux (embed) / mac (benchmark)  **Status:** embed done

### Measured — the full pool is embedded

**63,878 faces in 11.6 minutes at 91 face/s.** MobileFaceNet, CPU execution provider,
`Linux-x86_64`, onnxruntime 1.22.1. Zero pending, zero unreadable crops.

Faster than the 8–15 min PLAN.md estimate despite carrying **twice** the assumed face count,
which confirms the reasoning behind decision 21: quality belongs in detection, speed in
embedding. The expensive irreversible pass took 102 minutes; the cheap repeatable one takes 12.

Provenance is a single row — one model, one platform, one runtime version — so nothing is
silently mixing incomparable vectors. Resume was exercised for real: 63,378 processed in this
run plus the 500-face trial equals the full 63,878.

### Measured — HDBSCAN scaling, before running it on the real pool

Benchmarked on synthetic unit-norm data of face-pool shape, on the M2:

| N | dim | seconds | peak RSS | 
|---|---|---|---|
| 4,000 | 512 | 9.9 | 228 MB |
| 8,000 | 512 | 45.3 | 272 MB |
| 16,000 | 512 | 190.7 | 320 MB |

Empirical scaling **t ~ N^2.14**, predicting **~62 min at 63,878 on the M2**, so roughly 2–4 h
on the i3. With `--pca 128`: **8.4 min predicted, a 6x speedup**, scaling N^2.00.

**The 8 GB OOM risk is closed.** Peak RSS never exceeded 320 MB, because sklearn's HDBSCAN
never materialises the dense 64k x 64k matrix that `metric="cosine"` would have forced
(decision 26). Cost is time, not memory.

### Not measured, and therefore not adopted: PCA

An attempt to measure PCA's quality cost returned ARI 0.018 between full-dimension and PCA-128
labelings — apparent total disagreement. **That number is discarded, not reported as a finding.**
The synthetic data was 69% noise with no real cluster structure, so both runs were clustering
random points, and two clusterings of noise agree at chance. The benchmark measured *time*
validly and *quality* not at all.

So `--pca` stays off. PLAN.md section 9 is explicit that an `UNMEASURED` compromise blocks the
phase introducing it, and this one is not merely cosmetic: **the sampler uses the bootstrap
cluster as its stand-in for identity** when enforcing the per-person-per-day and per-person caps
and when detecting cross-era identities. Degrading it degrades gold-set composition, which is
exactly the silent, undetectable failure the caps exist to prevent. Calling this step
"throwaway" earlier in the log was too glib — it is throwaway as an *output*, not as an *input*.

With 63,878 real embeddings now on disk, the comparison can be done properly: run full
dimension, then PCA, and compare ARI on real vectors. That would earn a register entry.

### Next
- `bootstrap_cluster.py` at full dimension, overnight, ~2–4 h.
- Then `sample_gold_set.py`, which is expected to fail its composition report on the first run.

---

## 2026-09-16 — Correction: the parity FAIL was the harness, not the pipeline

**Phase:** 1  **Machine:** linux (run) / mac (diagnosis)  **Status:** done

### What happened
`verify_embedding_parity.py` reported **FAIL, worst cosine distance 7.74e-02** against a
1e-3 tolerance — 77x over. Taken at face value that condemns every embedding in the project.

It was wrong. **The pipeline is correct; the verification script was not.**

### Diagnosis
The run itself contained the first clue: `/127.5` and `/128.0` gave *identical* distance to
the reference (6.06e-02 vs 6.06e-02). A normalisation error would have moved them apart, so
the constant was exonerated immediately and only channel order or layout remained.

InsightFace's `ArcFaceONNX.get_feat()` calls `cv2.dnn.blobFromImages(..., swapRB=True)`. It is
built around `cv2.imread`, so it expects **BGR** and performs the swap to RGB itself. The script
handed it RGB, which it dutifully swapped to BGR — so the reference embedded channel-swapped
faces while ours embedded correct ones.

Confirmed by prediction rather than by argument. If the cause is a red/blue swap, then
`distance(ours_RGB, ours_BGR)` computed entirely within our own code must reproduce the
reported figures:

| Crop | ours_RGB vs ours_BGR | parity reported vs insightface |
|---|---|---|
| 0 | 6.03e-02 | 6.06e-02 |
| 1 | 7.16e-02 | 7.20e-02 |
| 2 | 7.78e-02 | 7.74e-02 |
| 3 | 4.35e-02 | 4.34e-02 |
| 4 | 6.66e-02 | 6.61e-02 |

Three significant figures on all five crops. The mismatch is exactly one channel swap.

### Also checked: the detector, because it processed all 63,878 faces
`detect.py` passes `image_rgb` straight to the model with no swap; InsightFace's SCRFD reaches
the same place via `blobFromImage(swapRB=True)` on BGR. Both deliver RGB to the network, so
**the existing face pool is unaffected** and does not need rebuilding.

### Fixed
- `reference.get_feat()` is now fed BGR, with a comment explaining why, because this will
  otherwise be re-broken by the next person who assumes RGB in, RGB out.
- The parity table gained a **control column** that deliberately feeds RGB into `get_feat`.
  A correct setup now shows a small first column *and* a large last one, so the script
  diagnoses a mismatch instead of merely announcing one.
- `f.interocular_px()` -> `f.interocular_px`. It is a property; calling it raised TypeError.

### Verified — parity is now bit-exact

Re-run on Linux after the fix:

| Crop | ours /127.5 | alt /128.0 | control (RGB into get_feat) |
|---|---|---|---|
| 0 | **0.00e+00** | 1.61e-06 | 6.06e-02 |
| 1 | -1.19e-07 | 2.26e-06 | 7.20e-02 |
| 2 | **0.00e+00** | 2.26e-06 | 7.74e-02 |
| 3 | **0.00e+00** | 2.26e-06 | 4.34e-02 |
| 4 | **0.00e+00** | 2.92e-06 | 6.61e-02 |

**PASS. Worst cosine distance 0.00e+00** — not "within tolerance" but bit-identical to the
reference, with the single -1.19e-07 being float rounding. The control column stays at 6-7e-02,
so the harness demonstrably distinguishes right from wrong rather than passing everything.

Four things are proven at once: **RGB channel order, NCHW layout, `(x - 127.5) / 127.5`
normalisation, and no hidden resize or interpolation difference.** The register's highest-rated
silent risk — "preprocessing mismatch, critical, silent" — is closed on the Linux box. It must be
run once per machine; arm64 and x86_64 will agree to ~1e-4, not exactly.

### Added: golden-value regression test

Parity proves preprocessing is correct *today*; it says nothing about tomorrow. `tests/
test_golden_embeddings.py` re-embeds the five fixed crops and fails if they drift beyond 1e-4,
which is the PLAN.md section 3 deliverable. Generate the baseline with
`verify_embedding_parity.py --write-golden`, which deliberately refuses to run until reference
parity has passed — pinning unverified values would lock in whatever bug was present.

**Caught while writing it: the baseline was going to be written into `tests/`, and therefore
committed.** Those vectors are embeddings of a real person's face. CONTEXT.md is explicit that
embeddings are biometric templates, not "just floats", and this repository is intended to be
public, so that would have been an irreversible leak of exactly the kind `.gitignore` exists to
prevent. Moved to gitignored `data/`, which also suits it being platform-specific and per-machine.

### The lesson worth keeping
Every safeguard in CONTEXT.md section 6 is aimed at the silent *false negative* — a check that
passes when it should not. This was the opposite: a **false alarm** from an unverified verifier.
Both cost the same amount of trust. The verification harness needs a control case proving it
can tell right from wrong, or "FAIL" is just an unvalidated claim. Rule 8 — verify by effect,
not by a tool reporting a result — applies to tools reporting *failure* too.

### Open question: the two machines are not running the same code
The Linux run did not crash on `f.interocular_px()`, which is a `@property` in the committed
`detect.py` and raises TypeError when called. It therefore cannot be the committed file. Nothing
in this session has been committed, so the Linux copy arrived out-of-band and has diverged.
CONTEXT.md section 3 requires both machines to run identical code, or results stop being
comparable. **Confirm with `git log --oneline -1` and `git status` on both before the embedding
pass runs.**

---

## 2026-09-11 — Gold-set pipeline: embed, cluster, sample, label, export

**Phase:** 1  **Machine:** mac (written and tested here; runs on linux)  **Status:** done, ready to run

### Did
Built the entire remaining automated half of Phase 1. Five new stages, six new scripts:

| Stage | Module | Script |
|---|---|---|
| Embed 63,878 crops (MobileFaceNet) | `embed.py` | `embed_faces.py` |
| Bootstrap clustering | `cluster.py` | `bootstrap_cluster.py` |
| Stratified sampling + composition report | `sampling.py` | `sample_gold_set.py` |
| Keyboard grid labelling UI | `labelui.py` | `label_gold_set.py` |
| Freeze to `labels.csv` | — | `export_gold_set.py` |
| Preprocessing arbitration | — | `verify_embedding_parity.py` |

Schema v4 adds `face_embeddings`, `bootstrap_clusters`, `gold_candidates`, `gold_labels`.
All additive, so existing databases migrate on open without a rescan.

46 new tests (**106 total**). `ruff`, `ruff format`, `mypy`, `pytest` all green.

### Measured — the normalisation conflict is real but numerically irrelevant

`configs/baseline.yaml` said `scale: 127.5`; **CONTEXT.md rule 6 says `/128.0`**. InsightFace's
reference `ArcFaceONNX` uses 127.5. Rather than pick a side from documentation, both were run
against the real sample photograph:

| Crop | Cosine similarity | Distance |
|---|---|---|
| 0–5 | 0.999997–0.999998 | 1.7e-06 – 3.1e-06 |

**Worst cosine distance 3.1e-06 — 300x below the 1e-3 parity tolerance.** The two constants are
interchangeable in practice. The code follows 127.5 anyway, because matching the reference
implementation exactly is what makes the upstream parity test meaningful. CONTEXT.md rule 6
corrected to say so.

This does **not** discharge the preprocessing risk. Channel order and layout are still unverified
against upstream; only `verify_embedding_parity.py` with the `insightface` package installed can
close that, and it exits non-zero rather than skipping when the package is absent. A check that
silently passes when it did not run is worse than no check — the face-pool tests already made
that mistake once with `/tmp`.

### Measured — end-to-end smoke test

Real detect → align → embed on the sample photograph, then a 240-face synthetic corpus through
cluster → sample → label:

- Embeddings L2-normalised to 1.000000, self-similarity 1.000000, **distinct faces mean +0.065 /
  max +0.200** — genuinely discriminative, not noise.
- Re-running the identical batch returns byte-identical vectors. Determinism holds.
- Clustering recovered **all 12 planted identities**, 13.8% noise (the planted strangers).
- Sampler landed size 14/25/38/23 against a 10/25/40/25 target, kind 66/34 against 67/33,
  filtered 88/12 exactly. Per-cluster cap held at 17 of a permitted 70.
- Label queue drains cluster → leftovers → done, and **the noise bucket is genuinely reached**.

The first smoke run failed its own noise assertion. The cause was the test, not the code: the
synthetic corpus clustered perfectly, so no noise bucket existed to serve. Planting unclusterable
faces fixed it. Worth recording because a test that cannot fail proves nothing.

### Decided

- **Labelling UI on the standard library's `http.server`, not FastAPI.** FastAPI and uvicorn are
  absent from `requirements.lock.txt`, and that lock file must stay identical on both machines or
  embeddings drift. Adding two dependencies plus their transitive tree for a single-user localhost
  tool is a real cost against no benefit. Phase 7 can swap the transport; the frontend and the
  label semantics are the parts worth keeping, and they carry over unchanged. Binds to 127.0.0.1.
- **Euclidean distance on L2-normalised vectors, not `metric="cosine"`.** Squared Euclidean is
  `2 - 2*cos` on unit vectors — a monotonic function of cosine, so clusters are identical — but it
  lets sklearn use a tree instead of materialising a dense 64k x 64k distance matrix. That matrix
  is **32 GB**; the target machine has 8 GB. PCA is available via `--pca` but **off by default**:
  measure before optimising.
- **Caps are applied by trimming the pool before quota filling**, not as a constraint inside the
  greedy loop. That makes the per-person-per-day and per-person caps a guarantee rather than a
  best effort, and keeps selection deterministic.
- **The sampler exits non-zero when its composition report misses.** A gold set whose strata were
  assumed rather than verified produces confident numbers about the wrong population, and the
  error is undetectable once labelling starts.

### Fixed: config drift

`configs/baseline.yaml` still described the pipeline that was *planned*, not the one that **ran**:
`det_500m.onnx` and a 1280px decode cap, against the `det_10g.onnx` at 2048px the face pool was
actually built with (decisions 21/22; C4 superseded, C5 withdrawn). Since "all tunables live in
YAML, never hardcoded" is a project rule and Phase 2 onward reads this file, the config was
actively misdescribing the 63,878 faces already on disk. Corrected, with the decision numbers
referenced inline.

### Problems / surprises

- **`_person_sort_key` sorted lexically at first**, so `person_10` fell between `person_1` and
  `person_2` in the merge prompt. During labelling that makes an existing id easy to miss, which
  produces a duplicate identity — and a split identity in the gold set is exactly the kind of
  error that is invisible afterwards. Now sorted numerically, with a regression test.
- The greedy sampler cannot hit seven marginals exactly and is not meant to. It minimises the
  largest outstanding deficit each pick; `--tolerance` sets how much residual skew is acceptable.
  Expect the **size** dimension to be the one that misses on the real corpus — the pool is 33.8%
  tiny against a 10% target, because a 12px face cannot be labelled by a human at all.

### Next
- Look at the tiny bucket (`contact_sheet.py --bucket tiny`) — still the open gate on the strata.
- Run the five commands on Linux, in order, and record the real numbers here.
- Then `eval/metrics.py` and `run_experiment.py`, which are all that remain of Phase 1.

---

## 2026-09-06 — Face pool built on the real corpus

**Phase:** 1  **Machine:** linux  **Status:** done

### Measured

**63,878 faces from 18,366 photos = 3.48 faces/photo.**
102.3 minutes at 3.0 photo/s, SCRFD-10G @640, 2 workers x 2 threads, decode cap 2048px.

Faster than the 2–3 h estimate. Resumability confirmed: 18,166 pending + 200 from the trial
run = 18,366 total.

| Size (inter-ocular px) | Faces | Share |
|---|---|---|
| tiny <20 | 21,594 | 33.8% |
| small 20–40 | 16,241 | 25.4% |
| medium 40–80 | 14,088 | 22.1% |
| large >80 | 11,955 | 18.7% |

| Pose | Faces | Share |
|---|---|---|
| frontal <15 deg | 30,742 | 48.1% |
| semi 15–45 | 19,771 | 31.0% |
| profile >45 | 13,365 | 20.9% |

| Kind | Photos | Faces | Faces/photo |
|---|---|---|---|
| photo | 12,733 | 43,394 | 3.41 |
| forwarded | 5,633 | 20,484 | 3.64 |

Errors: 3 of 18,366, all truncated JPEGs. Left unrecovered — three files is noise, and
enabling `LOAD_TRUNCATED_IMAGES` would silently accept corrupt pixel data everywhere.

### What the numbers say
- **3.48 faces/photo is well above the 1.5–2.5 assumed while planning.** This is a
  group-photo-heavy library, consistent with college and family events.
- **Forwarded images yield *more* faces per photo than camera originals (3.64 vs 3.41)**
  despite being ~5x smaller. Group photos are what people share. Combined with their lower
  resolution, this sharpens the gating-bias risk already in the register.
- **34% of faces are under 20 px inter-ocular.** At that size, warping to 112x112 is mostly
  interpolation and the embedding will be close to noise. Whether these are genuine crowd
  faces or detector false positives is unresolved and must be settled by looking, not by
  statistics — it sets the Phase 3 gating threshold and changes the gold-set strata.
- Real pose distribution (20.9% profile) is higher than the 15% assumed for the gold set.
  The strata targets in PLAN.md should follow the corpus, not the other way round.

### Did
- `scripts/contact_sheet.py` — renders a labelled grid of crops for any size/pose/blur/kind
  bucket, so the tiny-face question can be answered by eye.
- `--stats` gained a **face size by photo kind** table, directly testing whether messaging
  recompression biases face size and therefore gating.
- `CONTEXT.md` — a briefing for a fresh session or another machine.

### Next
- Look at the tiny bucket, and at forwarded vs photo separately.
- Embeddings (MobileFaceNet), bootstrap clustering, stratified sampler, labelling UI.

---

## 2026-09-06 — Face pool builder (detect + align)

**Phase:** 1  **Machine:** mac  **Status:** done, ready to run on the corpus

### Scan verdict: ready
The tag-306 fix is confirmed working. 116 photos moved off untrustworthy modification
timestamps onto filename dates, and the spurious **2002** entry vanished from the era
histogram — exactly the bug being hunted. `audio` split out cleanly: unsupported 282 → 45.

Final corpus: **18,366 candidates, 94.8% dated, era 2007–2025** with peaks at 2016 and 2022,
15+ camera models. Good enough to proceed.

Decided without asking, per the user's request to handle minor calls directly: **skipping the
54 MB zip.** It is roughly 50 photos in a farewell folder; for validation data that is noise.

### The decision that matters: SCRFD-10G, not 500M

Detection is the **only** stage that needs the original photos. Embedding works from the saved
112x112 crops forever after. So:

- A face the detector misses now is permanently absent from the gold set.
- An embedding choice made now can be revised on the Mac for free.

Therefore quality goes into detection, speed into embedding.

| Detector | WIDER-hard mAP | Measured on M2 |
|---|---|---|
| SCRFD-500M @640 | 68.5 | 38 ms/photo |
| **SCRFD-10G @640** | **83.1** | **124 ms/photo** |
| SCRFD-10G @1024 | — | 260 ms/photo |

Register entries **C4 and C5 are withdrawn**, not deferred. Decode cap raised 1280 → 2048px,
which costs almost nothing because `draft()` scales by powers of two.

**1024px input rejected on evidence:** on a real 6-face photo it found the *same six faces*
with *lower* confidence (0.78–0.89 vs 0.87–0.92) at 2.2x the cost.

Estimated corpus runtime: **~30–45 min on the M2, ~2–3 h on the i3.** Resumable, unattended.

### Verified against a real photograph, not synthetic data
- Both detectors find the same 6 faces with sensible boxes, and yaw values that match the
  image (one face at -88 degrees is genuinely in profile).
- Umeyama similarity transform inverts a known rotate/scale/translate to 1e-3, and is proven
  shear-free (`M @ M.T` is a multiple of the identity — affine would not be).
- Aligned crop inspected visually: correctly centred at 112x112.
- Residual on a real face is ~9.6 px max across 5 landmarks. Expected: the ArcFace template is
  an *average* face, so individual proportions differ even when perfectly frontal.
- Decode is only ~4 ms at 2048px thanks to `draft()`; the model dominates, as predicted.

### Did
- `detect.py` — SCRFD ONNX wrapper. Nine outputs decoded as 3 strides x (score, bbox-distance,
  keypoint-distance), 2 anchors per cell, letterboxed top-left as the model was trained.
- `align.py` — Umeyama similarity warp to the ArcFace template, context crops for human
  labelling, and auto-derived attributes (inter-ocular size, yaw/roll, blur, exposure).
- `facepool.py` + `scripts/build_face_pool.py` — parallel, resumable, streaming.
- `faces` and `photo_pool_status` tables.
- 17 new tests (60 total), including end-to-end runs on a real multi-face photo.

### Problems / surprises
- **A `multi_replace` edit silently deleted the `--packs` argument** from
  `download_models.py`. Caught by diffing before committing. Worth repeating: verify edits by
  their effect, not by the tool reporting success.
- Face-pool tests originally depended on a file in `/tmp`, so they would have **silently
  skipped** on Linux — the worst kind of test failure. The sample photograph is now a managed,
  checksummed asset in gitignored `data/test_assets/`, fetched by `download_models.py`.
  Detection genuinely cannot be tested on synthetic images, and committing photos of real
  people to a public repository is not acceptable.
- ONNX Runtime emits shape warnings at non-640 input sizes; the models carry static output
  shapes baked for 640x640. Values are still correct, but it is another reason to stay at 640.

### Next
- Run the pool on the corpus, record faces/photo and the size and pose distributions.
- Then embeddings, bootstrap clustering, and the labelling UI.

---

## 2026-09-06 — Date recovery validated; EXIF tag 306 demoted

**Phase:** 1  **Machine:** linux (data) / mac (code)  **Status:** done

### Measured — date recovery worked

| | Before | After |
|---|---|---|
| Undated candidates | 6,718 (37%) | **960 (5.2%)** |
| 2024 | 353 | **2,773** |
| 2025 | 375 | **1,366** |

Provenance: exif 11,648 (63.4%) · filename 5,758 (31.4%) · none 960 (5.2%).
The user's recollection of "~2,751 images in 2024" matched the corrected figure of 2,773; the
original report was wrong, not their memory. `--inspect "IISc"` confirmed: 4,401 files, 2,556 of
which recovered a 2024 date from filenames alone.

### Bug found: EXIF tag 306 was outranking filenames
Sample rows from `IISc/COORG/` showed seven photos with EXIF times inside a 25-second window
(14:58:49 … 14:59:14) but *different aspect ratios* (1280x960, 960x1280, 1222x720, 1089x960).
A burst cannot produce that; those are bulk-operation timestamps.

Cause: the reader fell back to EXIF tag 306 (`DateTime`), which is a **modification** time, and
treated it as a capture time. Fixed:

- `taken_at` now only ever holds `DateTimeOriginal` / `DateTimeDigitized`.
- Tag 306 is stored separately in `exif_modified_at` and ranked **below** filename dates.
- New priority: `exif_original > exif_digitized > filename > exif_modified > folder > mtime`.
- `filename_exif_disagreements()` reports how often the two sources diverge by more than a week,
  surfaced as a warning above 10%.

Schema v3, `SCAN_VERSION` bumped to 2, so a rescan is required to populate the new column.

### Other findings from the real corpus
- **911 MPO files** (5% of candidates) — dual-camera JPEGs from the Xiaomi phones. Pillow reads
  frame 0, so detection works.
- **80 HEIF** — `pillow-heif` confirmed working on real files.
- **`Moblie clicks/You cam perfect`: 2,163 files (~12% of candidates)** produced by a beauty
  retouching app. These filters alter face *geometry*, which is what the embedding encodes, so the
  same person filtered and unfiltered may not cluster together. Added to the gold-set strata at its
  real share and to the risk register.
- **Unsupported files explained**: 236 mp3 (2 GB of music), 13 `.dat` (VCD MPEG), 10 `.vcd`,
  4 `.ppt`, 3 `.psd` (138 MB), 1 `.zip` (54 MB). No photo format is being wrongly rejected.
- **`UIT/farewell/drive-download-...zip` (54 MB)** almost certainly contains photos. New `archive`
  kind plus a warning telling the user to extract and rescan.
- The 9 unreadable files are genuinely zero bytes (Android partial writes). Nothing recoverable.

### Did
- New kinds `audio` and `archive`, so 2 GB of music is no longer reported as "unsupported photos".
- Video extension list extended: `.mts .m2ts .3gpp .mpe .ogv .asf .rm .rmvb`.
- `.dat` and `.vcd` deliberately **left** as unsupported — `.dat` is genuinely ambiguous and
  claiming it is video would be overreach.
- Largest-folders table in `--diagnose`, flagging majority-undated folders in red.

### Next
- Rescan on Linux (SCAN_VERSION bump forces it, ~12 min) to populate `exif_modified_at`.
- Then `scripts/build_face_pool.py`.

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
