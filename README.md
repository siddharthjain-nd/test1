# faceindex

Offline, privacy-preserving face grouping for personal photo libraries.

Groups the photos in your library by person, entirely on your own machine. No cloud, no accounts,
no uploads. See [PLAN.md](PLAN.md) for the full development plan and [DEVLOG.md](DEVLOG.md) for the
running work log.

> **Status: pre-alpha.** Phase 0 of 13. Nothing works yet.

---

## What this is

An unsupervised **face clustering** system: it takes unknown faces and groups them into people.

It is deliberately **not** a face *identification* system. There is no enrolment step, no database of
known individuals, and no 1:N lookup. New people are handled with zero training — they simply form a
new group.

## What this is not

- Not a cloud service. Photos never leave your machine.
- Not for surveillance, access control, authentication, or identifying strangers.
- Not commercially usable — see Licensing below.

---

## Licensing — read before use

This project is **AGPL-3.0-or-later**.

The face recognition **model weights it downloads are licensed for non-commercial research use only**
(InsightFace model zoo). Their training datasets (MS1M, VGGFace2, Glint360K, WebFace600K) carry further
research-only restrictions.

Consequences:
- **Weights are never committed to this repository or redistributed.** They are downloaded on first run.
- **Do not use this project commercially.**
- Face templates are **biometric data**. Depending on where you live, processing them may fall under
  GDPR Art. 9, Illinois BIPA, or Texas CUBI. Running it on your own photos, on your own machine, is the
  intended use. Deploying it for other people is your legal responsibility.

---

## Setup

Requires [conda](https://docs.conda.io/) and Python 3.11.

```bash
conda create -n fca python=3.11 -y
conda activate fca

# Install the exact pinned versions, then the project itself without re-resolving.
pip install -r requirements.lock.txt
pip install -e . --no-deps

python scripts/download_models.py
```

`requirements.lock.txt` pins exact versions. Use it on **every** machine — differing library versions
change decoded pixels and therefore change embeddings, which makes results incomparable.

### Running on the Linux indexing box

The photo library lives on Linux, so that machine runs the one-time passes over the originals.
Copy and paste, adjusting only the corpus path:

```bash
# 1. Get the code
git clone <your-repo-url> faceindex && cd faceindex     # first time
git pull                                                # subsequently

# 2. Environment
conda create -n fca python=3.11 -y
conda activate fca
pip install -r requirements.lock.txt
pip install -e . --no-deps

# 3. Confirm this machine matches the Mac. Every version must be identical.
faceindex env

# 4. Download the models. Digests are checked against the committed
#    models/models.lock.json, which proves both machines run identical weights.
python scripts/download_models.py

# 5. Prove the code works here before pointing it at real photos.
pytest -q

# 6. Scan the library. Quote the path: it contains a space.
python scripts/scan_corpus.py --root "/path/to/photo Timeline"
```

Step 6 is resumable — if it is interrupted, rerun the identical command and it continues.
Re-print the report at any time without rescanning:

```bash
python scripts/scan_corpus.py --report-only
```

**Expected outcome of step 4:** every model prints `verified`, not `recorded`. `recorded` means the
lock file had no entry, which should not happen on a second machine — it indicates the lock was not
committed. `CHECKSUM MISMATCH` means the two machines have different weights; stop and investigate.

---

## Pipeline stages

| Stage | Command | Cost |
|---|---|---|
| Scan corpus | `scripts/scan_corpus.py` | Cheap. No pixels decoded; videos rejected on extension |
| Build face pool | *(not yet written)* | Expensive. The one long pass |

---

## Repository layout

| Path | Contents |
|---|---|
| `src/faceindex/` | Library code |
| `scripts/` | One-shot pipeline entry points |
| `configs/` | Experiment configuration (YAML) |
| `tests/` | Test suite |
| `data/` | **Gitignored.** Crops, embeddings, labels — biometric data |
| `models/` | **Gitignored.** Downloaded weights — non-redistributable |

---

## Documentation

- [PLAN.md](PLAN.md) — phased development plan, decision log, risk register, quality compromise register
- [DEVLOG.md](DEVLOG.md) — chronological work log
