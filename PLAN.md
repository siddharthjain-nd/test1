# Offline Face Grouping — Development Plan

> Living document. Update the **Decision Log** and **Phase Status** tables as work progresses.
> Every phase has an **Exit Criteria** section — do not start the next phase until it is met.

---

## 0. Project Charter

### Goal
An open-source, privacy-preserving photo library organiser that groups photos by person, with:
- a **server/desktop indexer** that does the heavy, high-quality work locally on your own machine
- an **Android app** that browses results offline and handles newly-taken photos on-device

### Non-goals (explicitly out of scope)
- Cloud/SaaS hosting of user photos
- Commercial licensing (see §Licensing — model weights forbid it)
- Real-time face identification, surveillance, access control, or any authentication use
- Matching Google Photos feature-for-feature

### Guiding principles
1. **Measure before optimising.** No change lands without a number from the eval harness.
2. **Data never leaves the machine.** Any cloud usage is limited to anonymised 112×112 crops or public data.
3. **The user is part of the algorithm.** Correction UX beats a 2% model gain.
4. **Boring tech.** Postgres, SQLite, ONNX. No distributed systems for a personal app.
5. **Reproducible.** Every experiment is a config file + a row in a results table.

### Hard constraints
- InsightFace model weights are **non-commercial research use only**
- Training data lineage (MS1M, VGGFace2, Glint360K, WebFace600K) is research-restricted
- Therefore: project is **AGPL-3.0**, weights are **downloaded, never vendored**

### Glossary — what problem we are actually solving

| Task | Question | Enrolled identity DB? | This project |
|---|---|---|---|
| **Verification** (1:1) | "Are these two faces the same person?" | No | Used internally as a distance |
| **Identification** (1:N) | "Who is this, from a database of known people?" | **Yes** | ❌ Explicitly not built |
| **Clustering** (unsupervised) | "Group these unknown faces into people" | **No** | ✅ **This is the product** |

Consequences of being a *clustering* system:
- **No training is required for the system to work on anyone.** The pretrained backbone embeds any face; a new person simply forms a new cluster.
- There is no enrolment step and no identity database. Person records are created by the clustering, and named by the user afterwards purely for display.
- Gold-set labels are **evaluation ground truth only**. They never enter the pipeline.

---

## 1. Phase Status

| Phase | Name | Track | Status |
|---|---|---|---|
| 0 | Foundations & decisions | Core | ☐ Not started |
| 1 | Gold set + evaluation harness | Core | ☐ Not started |
| 2 | Baseline pipeline | Core | ☐ Not started |
| 3 | Quality gating | Core | ☐ Not started |
| 4 | Constrained clustering | Core | ☐ Not started |
| 5 | Model & inference upgrade | Core | ☐ Not started |
| 6 | Scale, storage, incremental indexing | Core | ☐ Not started |
| 7 | API + human review UI | Core | ☐ Not started |
| 8 | Personalisation head (frozen backbone) | **Adaptation** | ☐ Optional — try after Phase 7 |
| 9 | Backbone fine-tuning | **Training** | ☐ Optional — **default: SKIP** |
| 10 | Knowledge distillation → mobile model | **Training** | ☐ Optional — **default: SKIP** |
| 11 | Android app — on-device inference | Mobile | ☐ Not started |
| 12 | Server ↔ phone sync | Mobile | ☐ Not started |
| 13 | Packaging & public release | Release | ☐ Not started |

**Track meanings**
- **Core** — required, sequential, each phase depends on the previous
- **Adaptation** — trains only a small head on frozen features; cheap, high ROI, no GPU needed
- **Training** — modifies or creates neural network weights; independent research tracks, skippable, needs GPU
- **Mobile** — Android client; can start in parallel after Phase 5

> Phases 8, 9, 10 are deliberately separated. They are **three different things** that are often confused:
> - **8 = adaptation**: learn a transform on top of frozen embeddings using your own labels. Minutes, CPU.
> - **9 = fine-tuning**: unfreeze and update part of an existing backbone. Hours, GPU, risks catastrophic forgetting.
> - **10 = distillation**: train a *new small* model to imitate a *big* model. Hours-to-days, GPU, a real ML project.
>
> **None of them are required, and 9 and 10 are default-skip.** Pretrained ArcFace models were trained on hundreds
> of thousands of identities across many GPUs; a personal library of ~30 people cannot improve on that. The
> project's quality gains come from Phases 3, 4, 5 and 7 — alignment, gating, constraints, thresholds, and
> correction UX — not from touching model weights. Do not open a training phase until the Core track has
> plateaued and the results table shows exactly where the remaining errors are.

---

## 2. Target Repository Layout

```
.
├── PLAN.md                     # this file
├── README.md
├── LICENSE                     # AGPL-3.0
├── pyproject.toml
├── docker-compose.yml
├── configs/
│   ├── baseline.yaml
│   └── experiments/            # one yaml per experiment run
├── models/                     # gitignored; populated by scripts/download_models.py
├── scripts/
│   ├── download_models.py
│   ├── build_gold_set.py
│   └── run_experiment.py
├── src/faceindex/
│   ├── ingest/                 # filesystem scan, EXIF, hashing, decoding
│   ├── detect/                 # detector wrappers
│   ├── align/                  # similarity transform to 112x112
│   ├── quality/                # blur, pose, size, confidence scoring
│   ├── embed/                  # embedding model wrappers, TTA
│   ├── cluster/                # graph build, constrained clustering
│   ├── store/                  # SQLite (dev) / Postgres+pgvector (prod)
│   ├── eval/                   # metrics, gold set loading, report generation
│   └── api/                    # FastAPI
├── web/                        # review UI
├── android/                    # Android client
└── data/                       # gitignored
    ├── gold/                   # labelled crops + labels.csv
    └── results/                # experiment result rows
```

---

## 3. Cross-Cutting Concerns (apply to every phase)

### Privacy & security
- [ ] No network calls in the indexing path. Add a test that asserts this.
- [ ] Model downloads are an explicit, separate, user-run step.
- [ ] Store aligned crops and embeddings, not copies of originals.
- [ ] Provide a single `purge-face-data` command that removes all derived biometric data.
- [ ] Never log embeddings or file paths at INFO level.
- [ ] Treat embeddings as sensitive: they are biometric templates, not "just floats".
- [ ] DB not exposed outside localhost by default in `docker-compose.yml`.

### Legal
- [ ] README states: non-commercial model weights, AGPL project, personal-use intent.
- [ ] README explicitly forbids surveillance/identification-of-strangers use cases.
- [ ] Note GDPR Art. 9 / Illinois BIPA / Texas CUBI implications for anyone deploying for others.

### Engineering hygiene
- [ ] `ruff` + `mypy` + `pytest` in CI from Phase 1 onward.
- [ ] Every pipeline stage is pure-ish: `input -> output`, no hidden global state.
- [ ] Determinism: seed everything; same input + same config = same clusters.
- [ ] All tunables live in YAML configs, never hardcoded.
- [ ] Golden-value tests: assert embeddings for 5 fixed crops stay stable across refactors.

### Performance discipline
- [ ] Profile before optimising. Expect **JPEG decode, not inference, to be the bottleneck**.
- [ ] Decode workers (CPU) must be decoupled from inference (GPU) via a queue.
- [ ] Every stage records wall-time into the results row.

---

## 4. Phases

---

### Phase 0 — Foundations & Decisions
**Track:** Core

**Goal:** Nothing runs yet; make the decisions that are expensive to reverse.

**Deliverables**
- [ ] Repo initialised, AGPL-3.0 LICENSE, README stub
- [ ] `pyproject.toml` with pinned deps (onnxruntime, numpy, opencv, pillow-simd, scikit-learn, hdbscan, pydantic, typer)
- [ ] `scripts/download_models.py` fetching SCRFD + MobileFaceNet + R50/R100 ONNX
- [ ] Decision Log table filled in (§6)
- [ ] Choose the photo corpus you will develop against (a real, messy, personal folder)

**Key considerations**
- Pin ONNX opset and runtime version — silent numerical drift across versions is real.
- Decide the canonical face record schema **now**; changing it later means reindexing.
  Minimum: `photo_id, bbox, landmarks[5], det_score, quality_score, embedding, model_version, taken_at, gps`.
- Include `model_version` and `pipeline_version` on every face row. You will re-run with new models and need to compare.
- Choose SQLite for Phases 1–5 (single file, trivially resettable), migrate to Postgres in Phase 6.

**Exit criteria**
- `python scripts/download_models.py` works, models land in `models/`, checksums verified.

**Risks**
- Model download links rot (Google Drive links in the InsightFace zoo are notorious). Mirror what you download and record checksums.

---

### Phase 1 — Gold Set + Evaluation Harness
**Track:** Core · **This is the most important phase in the project.**

**Goal:** Be able to score any pipeline configuration with one command.

**What the gold set is — and is not**
- It **is** a regression test suite: ground-truth groupings used to compute how well the clustering did.
- It is **not** training data. No label ever reaches the model or the pipeline.
- It is **not** an identity database. Person IDs are arbitrary (`person_1`, `person_2`); shuffling them changes nothing.
- Public benchmarks (LFW, IJB-C) cannot substitute: they measure *verification on celebrity photos*, not *clustering on your library*. A model can score 99.8% on LFW and still merge two of your relatives. Crucially, the **clustering threshold is library-specific** and can only be chosen against your own data.
- Hard cases are included so that regressions are *detectable*, not so the model can *learn* them. A gold set of easy frontal portraits scores ~0.98 on everything and teaches you nothing.

**Deliverables**
- [ ] `scripts/build_gold_set.py` — extract face crops from a chosen subset of your library into `data/gold/`
- [ ] A minimal labelling tool (CLI or tiny local web page) to assign person IDs to crops
- [ ] `data/gold/labels.csv` with **1,000–2,000 labelled faces across ~30 identities**
- [ ] `src/faceindex/eval/metrics.py` implementing:
  - Pairwise Precision / Recall / F1
  - BCubed Precision / Recall / F1
  - NMI / Adjusted Rand Index
  - Cluster count (predicted vs true), % faces labelled noise
- [ ] `scripts/run_experiment.py --config configs/X.yaml` → appends one row to `data/results/results.csv`
- [ ] A results-table renderer so you can diff experiments at a glance

**Key considerations**
- **Deliberately over-sample hard cases** in the gold set, or your metrics will be optimistically useless:
  - the same child at ages 2 / 6 / 12
  - siblings and parent/child pairs who look alike
  - sunglasses, masks, hats
  - profile / >45° yaw faces
  - low light, motion blur, heavy backlight
  - group photos with 5+ faces
  - a few *non-face* false positives from the detector
- Include a **"not a person of interest"** label for strangers/background — your clusterer must be allowed to call them noise.
- **Bootstrap the labelling, don't start from a blank slate.** Run detect+align, then a quick throwaway clustering pass, and label by *correcting* pre-grouped clusters (confirm / split / merge). Far faster than sorting loose crops. Caveat: you must also review the noise bucket and the rejected faces, or the gold set silently inherits the baseline's blind spots.
- The Phase 1 labelling tool and the Phase 7 review UI are **the same tool**. Build it once, here, and grow it.
- Freeze the gold set once built. If you keep editing it, your numbers stop being comparable.
- Keep a **held-out** slice (~20%) you only look at when you think you're done, to catch overfitting to the gold set.
- Store gold crops as 112×112 aligned JPEGs — small, portable, and safe to upload for cloud experiments.

**Exit criteria**
- Running the harness twice on the same config produces identical numbers.
- You can articulate a baseline F1 number, even if it comes from random clustering.

**Risks**
- Labelling fatigue → shortcuts → a bad gold set → every downstream conclusion is wrong. Do it carefully, once.

---

### Phase 2 — Baseline Pipeline
**Track:** Core

**Goal:** End-to-end detect → align → embed → cluster on a folder. Deliberately naive.

**Deliverables**
- [ ] `ingest`: recursive scan, content hash, EXIF (timestamp, GPS, orientation), HEIC support
- [ ] `detect`: SCRFD-500M-KPS via ONNX Runtime, returns bbox + 5 landmarks + score
- [ ] `align`: similarity transform of the 5 landmarks onto ArcFace canonical points → 112×112
- [ ] `embed`: MobileFaceNet ONNX → 512-d, L2-normalised
- [ ] `cluster`: plain HDBSCAN on cosine distance
- [ ] `store`: SQLite schema
- [ ] Baseline number recorded in results table

**Key considerations**
- **Alignment is the #1 source of silent bugs.** Verify by dumping aligned crops to disk and eyeballing them — eyes must sit on the same pixels every time.
- **Preprocessing must match training exactly.** Check the specific model's expectations:
  - channel order (RGB vs BGR)
  - normalisation (`(x - 127.5) / 128.0` for ArcFace, not `x / 255`)
  - NCHW vs NHWC
  - Validate by comparing your embeddings against the reference `insightface` Python package on the same crop — cosine distance should be < 1e-3.
- Always L2-normalise embeddings before any distance computation.
- Honour EXIF orientation before detection, or you'll miss most faces in portrait photos.
- Decode at a bounded size (~1280 px long edge) for now; full-res comes in Phase 5.
- Handle: corrupt files, 0-byte files, animated formats, screenshots, non-image files with image extensions.

**Exit criteria**
- Full run over your photo folder completes without crashing.
- Gold-set F1 recorded. **This number is your baseline for the rest of the project.**

**Risks**
- Temptation to tune before measuring. Resist. This phase's output is *a number*, not a good result.

---

### Phase 3 — Quality Gating
**Track:** Core · Expect the single largest quality jump here.

**Goal:** Stop feeding garbage faces into the clusterer.

**Deliverables**
- [ ] `quality`: per-face score combining
  - face size in pixels (inter-ocular distance is better than bbox height)
  - blur (variance of Laplacian, normalised by face size)
  - pose (yaw/pitch/roll estimated from the 5 landmarks)
  - detector confidence
  - over/under-exposure fraction
- [ ] Configurable thresholds; faces below threshold are stored but excluded from clustering
- [ ] Sweep experiment across threshold values, recorded in results table

**Key considerations**
- Gate, don't delete. Keep low-quality faces in the DB so they can be *assigned* to a person later even if they can't *form* a cluster.
- Two-tier design: `cluster_eligible` (strict) vs `assign_eligible` (loose).
- Normalise blur by face size — a small sharp face and a large blurry face have similar raw Laplacian variance.
- Watch the recall trade-off: aggressive gating raises precision and tanks recall. The results table will show you the knee.
- Consider a learned quality model later (e.g. SER-FIQ / CR-FIQA style) — but only if the heuristics plateau.

**Exit criteria**
- Documented threshold set with a measured F1 improvement over Phase 2.

---

### Phase 4 — Constrained Clustering
**Track:** Core · Highest ratio of impact to effort in the whole project.

**Goal:** Use free metadata signals that the embedding model cannot see.

**Deliverables**
- [ ] **Cannot-link**: two faces from the same source photo may never join the same cluster
- [ ] **Must-link (soft)**: burst shots / same-second captures bias toward merging
- [ ] Time-aware similarity: raise the merge bar as the gap between capture dates grows
- [ ] GPS/session grouping as a weak prior
- [ ] kNN graph construction + constrained agglomerative or constrained HDBSCAN
- [ ] Two-threshold scheme: strict for cluster *formation*, loose for *assignment* to confirmed clusters
- [ ] Multi-prototype clusters (keep k centroids per person, not one mean)

**Key considerations**
- Cannot-link is a hard constraint and must be enforced at *cluster* level, not pair level — after merging, the union must stay conflict-free.
- Edge case: the same person appearing twice in one photo (mirrors, collages, panoramas). Rare; accept the error or detect duplicates by high self-similarity.
- Multi-prototype matters because a person at 5 and at 35 are far apart in embedding space; a single centroid drifts to a meaningless average.
- Time-decay must be tunable and *evaluated* — it can hurt if your library has clusters of a person that only appear in one era.
- Keep the clustering step idempotent and re-runnable over the whole library; you will run it hundreds of times.

**Exit criteria**
- Measured F1 improvement from constraints alone (ablation: with/without cannot-link, with/without time prior).

---

### Phase 5 — Model & Inference Upgrade
**Track:** Core

**Goal:** Buy accuracy with compute, now that the surrounding logic is sound.

**Deliverables**
- [ ] Swap detector: SCRFD-500M → **SCRFD-10G-KPS**, run at full resolution or multi-scale
- [ ] Swap embedder: MobileFaceNet → **ResNet50 (buffalo_l)** and **ResNet100 (Glint360K)**
- [ ] **Flip-TTA**: average the L2-normalised embeddings of the crop and its mirror, re-normalise
- [ ] Optional: ensemble two backbones (concatenate or average normalised embeddings)
- [ ] GPU execution path (CUDA EP) with CPU fallback
- [ ] Decode/inference pipelining with a bounded queue
- [ ] Ablation table: every combination scored

**Key considerations**
- Detecting at full resolution dramatically improves recall of small faces in group photos — often a bigger win than the embedder upgrade.
- Multi-scale detection is expensive; measure whether it beats simply detecting at 1×.
- Re-tune clustering thresholds after **every** embedder change. Thresholds are model-specific; reusing them is a classic mistake.
- Track model size / latency / F1 as a three-way trade-off table — you need this for the mobile phase.
- Ensembling gives small gains for 2× cost. Verify it's worth it before keeping it.
- **Do not** add face restoration (GFPGAN / CodeFormer) before embedding — these hallucinate plausible faces and can invent identity. If you test it, test it as a measured experiment and expect it to hurt.

**Exit criteria**
- A documented "best known configuration" with its F1, and a latency/quality trade-off table.

---

### Phase 6 — Scale, Storage, Incremental Indexing
**Track:** Core

**Goal:** Run over the full library repeatedly without pain.

**Deliverables**
- [ ] Postgres 17 + **pgvector** with an HNSW index on embeddings
- [ ] Alembic migrations
- [ ] Job queue as a plain `jobs` table + worker loop (no Celery/Redis)
- [ ] Resumable indexing: crash mid-run and continue from the last checkpoint
- [ ] Incremental ingest: filesystem watcher / periodic scan by mtime + content hash
- [ ] Incremental assignment: new faces → nearest confirmed cluster, else to an `unassigned` pool
- [ ] Scheduled full re-clustering of the unassigned pool
- [ ] Store embeddings as `float32` for the server; keep an `int8` quantised column for mobile export

**Key considerations**
- Content hash, not path, is the identity of a photo — handles moves and duplicates.
- Store `model_version` per embedding so a model upgrade means "re-embed rows where version < X", not "wipe everything".
- Never let re-clustering destroy user-confirmed assignments. Confirmed labels are anchors and survive re-clustering.
- Design the re-clustering to be *stable*: person IDs must not shuffle between runs, or the UI becomes unusable.
- int8 quantisation of L2-normalised embeddings costs almost no accuracy and cuts storage 4×. Measure it to confirm.
- Brute-force cosine over ~100k vectors is milliseconds — only add HNSW when you actually measure a problem.

**Exit criteria**
- Full library indexed. Re-run after adding 100 new photos touches only those photos.

---

### Phase 7 — API + Human Review UI
**Track:** Core · This is where the product actually becomes good.

**Goal:** Make correcting the algorithm fast and pleasant.

**Deliverables**
- [ ] FastAPI: list people, list faces in a person, name/rename, merge, split, hide, "not this person"
- [ ] Web UI (SvelteKit or HTMX) with a grid of face crops
- [ ] **Merge suggestions**: surface the top-N most similar cluster pairs for one-click confirm/reject
- [ ] **Split assist**: within a cluster, show the sub-clusters so a bad merge can be undone in one action
- [ ] Bulk keyboard-driven labelling (this is a data-entry tool — optimise for speed)
- [ ] Every user action is persisted as a labelled constraint and fed back into clustering
- [ ] "Delete all face data" action

**Key considerations**
- User confirmations become **must-link** and rejections become **cannot-link** constraints. This closes the loop with Phase 4 and compounds over time.
- Order merge suggestions by expected information gain, not just similarity — asking about a pair you're already confident about wastes a click.
- Show *why* the system merged two faces (nearest neighbours, similarity score) — builds trust and helps you debug.
- Never auto-merge two clusters the user has explicitly separated.
- The corrections you collect here become the training labels for Phase 8. Design the schema with that in mind.

**Exit criteria**
- You can go from a fresh index to correctly-named people for your own family without touching the CLI.

---

### Phase 8 — Personalisation Head (Frozen Backbone)
**Track:** **Adaptation** — no backbone weights are modified. CPU-only, minutes to train. **Optional.**

> **Do not start this until Phases 3–7 have plateaued.** The system works without it. This is not required for
> new/unknown people to be grouped — clustering is unsupervised and handles strangers natively. This phase only
> re-shapes the *distance metric* to better separate the specific people already in your library.

**Goal:** Adapt the embedding space to *your specific* set of people using the labels from Phase 7.

**Deliverables**
- [ ] Export `(embedding, person_id)` pairs from confirmed labels
- [ ] Train a small projection head: `512 → 512` linear (or 2-layer MLP) with ArcFace / triplet / supervised-contrastive loss
- [ ] Alternatively/additionally: learn a **pair scorer** MLP over `(e1, e2, |e1-e2|, e1*e2)` trained on confirmed merges/splits
- [ ] Evaluate on the held-out gold slice
- [ ] Config flag to enable/disable the head, so it's always ablatable

**Key considerations**
- This is the **highest-ROI "make it better for me" step** and it is not real training — it's a linear probe on frozen features.
- Needs surprisingly few labels (a few hundred faces across 20–30 people) to help meaningfully.
- Risk: overfitting to your ~30 people so that *new* people cluster worse. Always evaluate on identities held out from head training.
- Keep the original embedding stored; the head is applied at query/cluster time so it can be retrained without re-embedding.
- Retrain automatically as the user adds more corrections.

**Exit criteria**
- Measured improvement on held-out identities, or a documented decision that it doesn't help on your data.

---

### Phase 9 — Backbone Fine-Tuning
**Track:** **Training** · Requires GPU · Independent of Phases 10–13

> **Default decision: SKIP.** The pretrained backbone was trained on hundreds of thousands of identities; a
> library of ~30 people is roughly four orders of magnitude less data. The expected outcome is a model that is
> slightly better on your family and meaningfully worse on everyone else. Only attempt this if Phase 8 showed a
> clear gain *and* the results table proves the remaining errors are embedding-quality errors rather than
> clustering, gating, or threshold errors. Documenting "we tried it and it hurt" is a valid, useful outcome.

**Goal:** Update part of an existing backbone's weights on your own data.

**Deliverables**
- [ ] Training script (PyTorch) with frozen-stem / unfrozen-last-blocks configuration
- [ ] Strong augmentation pipeline (random crop/scale, colour jitter, blur, JPEG artefacts, low-light simulation)
- [ ] ArcFace/CosFace margin loss head
- [ ] Regularisation against catastrophic forgetting (low LR, LwF/EWC, or mixing in a public dataset)
- [ ] Evaluation on the held-out gold slice **and** on a public benchmark to check for regression

**Key considerations**
- **Do Phase 8 first.** If the linear head already gets you most of the gain, fine-tuning may not be worth it.
- Very high risk of **catastrophic forgetting**: the model gets great at your 30 people and worse at everyone else. Always evaluate general performance, not just personal performance.
- Your dataset is tiny (thousands of images, tens of identities) versus the original (millions / hundreds of thousands). Use a very low learning rate and few epochs.
- Free compute reality: feasible on a Kaggle T4 for a small backbone (MobileFaceNet); painful for ResNet100.
- Data leakage: identities in the eval set must not appear in the fine-tuning set.
- Keep the original weights. Always be able to A/B against them.

**Exit criteria**
- Fine-tuned model beats the frozen + Phase 8 head on held-out data **without** regressing on the general benchmark. If not, abandon this phase — that is a valid result.

---

### Phase 10 — Knowledge Distillation → Mobile Model
**Track:** **Training** · Requires GPU · The most research-flavoured phase

> **Default decision: SKIP.** This is a **compression** project, not an accuracy project — a student can never
> exceed its teacher. Its only purpose is shrinking server-grade quality into a phone-sized model. Only worth
> doing if Phase 11 ships and the measured mobile-vs-server quality gap is actually bothering you in daily use.

**Goal:** Train a small on-device model to imitate the big server model, narrowing the mobile/server quality gap.

**Deliverables**
- [ ] Teacher: ResNet100 embeddings computed over a large set of unlabelled face crops (your library + public faces)
- [ ] Student: MobileFaceNet / EdgeFace-class architecture
- [ ] Distillation loss (cosine/L2 to teacher embeddings, optionally + margin loss on labelled subset)
- [ ] Export student to ONNX, int8 quantise, verify numerics
- [ ] Compare student vs stock MobileFaceNet vs teacher on the gold set

**Key considerations**
- Distillation on **unlabelled** data works well here — you don't need identity labels, just teacher outputs. This is why it's tractable.
- Needs a lot of face crops. Your own library may be enough; augment with public face data if not.
- This is where free GPU (Kaggle ~30 T4-hours/week) genuinely earns its keep. Checkpoint every epoch to `/kaggle/working` — sessions get killed.
- Upload only 112×112 aligned crops, never original photos. Shuffle and rename them.
- Quantisation-aware training beats post-training quantisation for the final mobile model, but adds complexity — try PTQ first.
- Success bar: student meaningfully closer to teacher than stock MobileFaceNet is, at the same latency.

**Exit criteria**
- A quantised ONNX student model with a documented accuracy/latency point better than the stock mobile baseline.

---

### Phase 11 — Android App (On-Device Inference)
**Track:** Mobile · Can begin in parallel once Phase 5 fixes the model choice

**Goal:** Offline browsing + on-device processing of new photos.

**Deliverables**
- [ ] Kotlin app, `READ_MEDIA_IMAGES`, MediaStore scan + `ContentObserver`
- [ ] ONNX Runtime Mobile with XNNPACK; NNAPI/NPU as an opportunistic, benchmarked option
- [ ] The same detect → quality → align → embed pipeline, ported
- [ ] Room/SQLite store with int8 embedding BLOBs + face thumbnails
- [ ] `WorkManager` job: `requiresCharging` + `requiresDeviceIdle`, chunked and resumable
- [ ] Person browsing UI, on-device assignment of new faces to known people
- [ ] Visible indexing progress + a "delete all face data" control

**Key considerations**
- **Parity test first**: assert that Android embeddings match the Python pipeline within ~1e-3 cosine distance on fixed test crops. Do this before building any UI. Mismatches are almost always channel order or normalisation.
- **Decode dominates cost**, not inference. Use `ImageDecoder` with a target size; don't decode 12 MP.
- Android 14+ restricts `dataSync` foreground services (~6h/day cap) and requires a declared FGS type. Design for chunked, resumable work.
- Doze mode will suspend you. Never assume a continuous run.
- Thermal throttling on sustained batches — check `PowerManager.getThermalHeadroom()` and back off.
- NNAPI is inconsistent across OEMs and silently falls back to slow CPU. Default to XNNPACK; treat NPU as a per-device measured optimisation.
- Handle HEIC/HEIF, RAW/DNG, motion photos, screenshots.
- `READ_MEDIA_IMAGES` needs a Play Store declaration justifying broad media access.
- On-device clustering must be incremental and cheap — full re-clustering is a server job.

**Exit criteria**
- Full-library index completes on a real device across charging sessions, with results matching the server pipeline's person assignments on the same photos.

---

### Phase 12 — Server ↔ Phone Sync
**Track:** Mobile

**Goal:** Server quality, offline mobile browsing.

**Deliverables**
- [ ] Local-network pairing (QR code), no cloud broker
- [ ] Sync protocol: photo hashes → person assignments, names, thumbnails
- [ ] Conflict resolution when both sides made changes
- [ ] Phone works fully offline when the server is unreachable
- [ ] Transport encryption + device authentication

**Key considerations**
- Sync assignments and labels, not raw embeddings, where possible — less biometric data on the wire.
- Content hash is the join key between server and phone.
- User corrections on either side must propagate and must never be silently overwritten.
- Server is authoritative for clustering; phone is authoritative for photos it has that the server hasn't seen.
- Keep it dumb: last-write-wins per field with a timestamp is fine for a personal app.

**Exit criteria**
- Add a photo on the phone, index it on the server, see the person assignment appear on the phone — offline afterwards.

---

### Phase 13 — Packaging & Public Release
**Track:** Release

**Deliverables**
- [ ] `docker-compose.yml` one-command server setup
- [ ] README: install, model download + licence warning, privacy statement, screenshots
- [ ] Benchmark results published (your gold-set numbers, honestly reported including failures)
- [ ] `CONTRIBUTING.md`, issue templates
- [ ] Android build via F-Droid and/or GitHub Releases APK
- [ ] Reproducible eval: anyone can run the harness on their own library

**Key considerations**
- Do **not** ship model weights in the repo or the APK if their licence forbids redistribution. Download at first run.
- State clearly: non-commercial weights, AGPL project, no surveillance use.
- Publish the negative results too — "fine-tuning didn't help" is useful to others.

---

## 5. Metrics Tracked in Every Experiment Row

| Field | Notes |
|---|---|
| `experiment_id`, `timestamp`, `config_hash` | reproducibility |
| `detector`, `embedder`, `pipeline_version` | what was run |
| `n_photos`, `n_faces_detected`, `n_faces_gated` | recall sanity |
| `pairwise_P/R/F1` | primary metric |
| `bcubed_P/R/F1` | less biased by cluster size |
| `nmi`, `ari` | secondary |
| `n_clusters_pred` vs `n_clusters_true` | fragmentation check |
| `pct_noise` | over-gating check |
| `t_decode`, `t_detect`, `t_embed`, `t_cluster` | perf |
| `peak_ram_mb`, `model_size_mb` | mobile feasibility |

---

## 6. Decision Log

| # | Decision | Choice | Rationale | Date | Revisit if |
|---|---|---|---|---|---|
| 1 | Licence | AGPL-3.0 | Model weights are non-commercial; prevents commercial repackaging | | Ever wanting to commercialise |
| 2 | Dev DB | SQLite → Postgres+pgvector at Phase 6 | Fast iteration first, scale later | | |
| 3 | Runtime | ONNX Runtime | One model format for server and phone | | |
| 4 | Job system | Plain DB table + worker | Avoid Celery/Redis for a personal app | | |
| 5 | Problem framing | Unsupervised **clustering**, not identification | No enrolment DB, no 1:N lookup; new people are handled with zero training | 2026-09-04 | Never — this is a privacy design choice |
| 6 | Model training | Use pretrained backbones as-is; Phases 9 & 10 default-skip | Cannot beat ArcFace with ~30 identities; gains live in the pipeline, not the weights | 2026-09-04 | Core track plateaus and errors are proven to be embedding-quality errors |
| 7 | Gold-set labels | Evaluation ground truth only, never training input | Enables measured comparison of configs; keeps the system unsupervised | 2026-09-04 | |
| 8 | | | | | |

---

## 7. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Bad gold set → all conclusions wrong | Critical | Label carefully once; hold out a slice; over-sample hard cases |
| Preprocessing mismatch (RGB/BGR, normalisation) | Critical, silent | Parity-test against reference `insightface` output on day one |
| Model download links rot | High | Mirror + checksum everything in Phase 0 |
| Overfitting to your own 30 people (Phases 8–9) | High | Always evaluate on held-out identities and a general benchmark |
| Demographic bias in pretrained models | High | Measure per-person recall; models are Western-data-skewed |
| Android background execution kills indexing | High | Chunked, resumable, checkpointed work |
| Thresholds not re-tuned after model change | Medium | Make threshold sweep part of every model-change experiment |
| Scope creep into a full photo manager | Medium | Non-goals are listed in §0; re-read them |
| Kids / siblings / twins mis-cluster | Medium | Accept; rely on Phase 7 correction UX |
| Re-clustering destroys user labels | High | Confirmed labels are immutable anchors |

---

## 8. Known Hard Problems (accepted, not solved)

- Identical twins — will merge. No fix.
- Infants and toddlers — cluster poorly and merge with each other.
- The same person across 10+ years — often splits into multiple clusters. Mitigated by multi-prototype clusters + user merges.
- Heavy occlusion (masks, sunglasses, hats) — falls to noise.
- Demographic performance gaps inherited from training data.
- We will not match Google Photos. The target is "good enough that a privacy-conscious user prefers it."
