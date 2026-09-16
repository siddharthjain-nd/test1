#!/usr/bin/env python3
"""Prove the embedding preprocessing is right, and pin it with golden values.

Preprocessing mismatch is the highest-rated silent risk in the register: wrong channel
order, wrong normalisation or wrong layout all produce confident, plausible, meaningless
embeddings and never raise an error. Nothing else in the pipeline can detect it.

Two independent checks:

1. **Normalisation arbitration.** CONTEXT.md section 6 says ``(x - 127.5) / 128.0``;
   configs/baseline.yaml and InsightFace's reference ``ArcFaceONNX`` both say 127.5.
   This measures the actual divergence instead of picking a side from documentation.

2. **Reference parity.** Compares against the `insightface` package on identical crops.
   Requires ``pip install insightface`` -- a dev-only dependency that must never be
   imported by the pipeline. If it is absent this script says so loudly and exits
   non-zero rather than skipping quietly: a check that silently passes when it did not
   run is worse than no check.

Run once per machine before freezing any gold set.

Usage
    python scripts/verify_embedding_parity.py
    python scripts/verify_embedding_parity.py --write-golden   # pin current values
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from rich.console import Console
from rich.table import Table

from faceindex import align, embed, facepool, paths
from faceindex.detect import ScrfdDetector

console = Console()

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "tests" / "golden_embeddings.json"
N_GOLDEN = 5


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_reference_crops() -> np.ndarray:
    """Aligned crops from the managed sample photograph.

    Detection cannot be validated on synthetic images, and committing photographs of real
    people to a public repository is not acceptable -- hence a checksummed, gitignored asset.
    """
    photo = paths.sample_faces_photo()
    if not photo.exists():
        console.print(f"[red]Sample photo missing: {photo}[/red]")
        console.print("Run: python scripts/download_models.py")
        raise SystemExit(1)

    detector_path = paths.models_dir() / "buffalo_l" / "det_10g.onnx"
    detector = ScrfdDetector(detector_path, input_size=(640, 640))
    image, _ = facepool.decode(photo, 2048)
    faces = detector.detect(image)

    if len(faces) < N_GOLDEN:
        console.print(f"[red]Only {len(faces)} faces in the sample photo.[/red]")
        raise SystemExit(1)

    faces = sorted(faces, key=lambda f: -f.interocular_px)[:N_GOLDEN]
    return np.stack([align.align_face(image, face.landmarks) for face in faces])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model", default="buffalo_sc/w600k_mbf.onnx")
    parser.add_argument("--write-golden", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    model_path = paths.models_dir() / args.model
    if not model_path.exists():
        console.print(f"[red]Model not found: {model_path}[/red]")
        return 1

    crops = load_reference_crops()
    console.print(f"Using {len(crops)} aligned crops from the sample photograph.\n")

    ours = embed.ArcFaceEmbedder(embed.EmbedConfig(model_path=model_path, std=127.5))
    alt = embed.ArcFaceEmbedder(embed.EmbedConfig(model_path=model_path, std=128.0))
    vectors_127 = ours.embed(crops)
    vectors_128 = alt.embed(crops)

    # ---- Check 1: how much does the documentation disagreement actually matter? ----
    table = Table(
        title="Normalisation: /127.5 (reference) vs /128.0 (CONTEXT.md)", header_style="bold"
    )
    table.add_column("Crop", justify="right")
    table.add_column("Cosine similarity", justify="right")
    table.add_column("Cosine distance", justify="right")
    worst = 1.0
    for index, (a, b) in enumerate(zip(vectors_127, vectors_128, strict=True)):
        similarity = cosine(a, b)
        worst = min(worst, similarity)
        table.add_row(str(index), f"{similarity:.8f}", f"{1 - similarity:.2e}")
    console.print(table)

    divergence = 1 - worst
    if divergence < args.tolerance:
        console.print(
            f"[green]Divergence {divergence:.2e} is below the {args.tolerance:.0e} parity "
            f"tolerance.[/green] The two constants are interchangeable in practice, but the "
            f"code follows the reference (127.5) so it stays provably aligned with upstream.\n"
        )
    else:
        console.print(
            f"[red]Divergence {divergence:.2e} exceeds the {args.tolerance:.0e} tolerance.[/red] "
            f"The constant is not cosmetic. Use 127.5 -- it is what the weights were trained "
            f"with -- and correct CONTEXT.md section 6.\n"
        )

    # ---- Check 2: parity against the reference implementation ----
    try:
        from insightface.model_zoo import get_model  # type: ignore[import-not-found]
    except ImportError:
        console.print(
            "[red]The `insightface` package is not installed, so reference parity was NOT "
            "verified.[/red]\n\n"
            "  pip install insightface        # dev-only; never imported by the pipeline\n\n"
            "Until this passes, the preprocessing is unverified against upstream. That is the "
            "single highest-rated silent risk in the register -- do not freeze a gold set first."
        )
        return 2

    reference = get_model(str(model_path))
    reference.prepare(ctx_id=-1)

    # insightface's get_feat() calls cv2.dnn.blobFromImages(..., swapRB=True). It is built
    # around cv2.imread, so it expects **BGR** and swaps to RGB itself. Our crops are RGB,
    # so they must be flipped on the way in or the reference silently embeds a
    # channel-swapped face -- which looks exactly like a preprocessing bug in our code.
    #
    # Both orders are measured rather than assumed, so this script diagnoses the mismatch
    # instead of merely reporting one. Feeding BGR is what makes the two pipelines agree.
    def reference_feat(batch: np.ndarray) -> np.ndarray:
        vectors = np.stack([reference.get_feat(crop) for crop in batch]).reshape(len(batch), -1)
        return embed.l2_normalise(vectors.astype(np.float32))

    reference_vectors = reference_feat(np.ascontiguousarray(crops[:, :, :, ::-1]))
    reference_rgb_in = reference_feat(crops)

    parity = Table(title="Parity vs insightface reference", header_style="bold")
    parity.add_column("Crop", justify="right")
    parity.add_column("ours /127.5", justify="right")
    parity.add_column("alt /128.0", justify="right")
    parity.add_column("RGB fed to get_feat", justify="right")
    worst_ours = 0.0
    worst_swapped = 0.0
    for index in range(len(crops)):
        d127 = 1 - cosine(vectors_127[index], reference_vectors[index])
        d128 = 1 - cosine(vectors_128[index], reference_vectors[index])
        dswap = 1 - cosine(vectors_127[index], reference_rgb_in[index])
        worst_ours = max(worst_ours, d127)
        worst_swapped = max(worst_swapped, dswap)
        parity.add_row(str(index), f"{d127:.2e}", f"{d128:.2e}", f"{dswap:.2e}")
    console.print(parity)

    console.print(
        f"\nThe last column is the control: it feeds RGB straight into get_feat, so the "
        f"reference swaps it to BGR. It should be ~{worst_swapped:.0e} -- large, and roughly "
        f"the size of a red/blue channel swap. If the first column is small and the last is "
        f"large, both pipelines agree and the harness is wired correctly."
    )

    if worst_ours < args.tolerance:
        console.print(
            f"\n[green]PASS.[/green] Worst cosine distance to the reference is "
            f"{worst_ours:.2e}, within {args.tolerance:.0e}."
        )
    else:
        console.print(
            f"\n[red]FAIL.[/red] Worst cosine distance {worst_ours:.2e} exceeds "
            f"{args.tolerance:.0e}. Check channel order (RGB vs BGR), layout (NCHW) and "
            f"normalisation before trusting any embedding."
        )
        return 1

    if args.write_golden:
        GOLDEN_PATH.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "mean": 127.5,
                    "std": 127.5,
                    "embed_version": embed.EMBED_VERSION,
                    "platform": embed.runtime_fingerprint()[0],
                    "onnxruntime": embed.runtime_fingerprint()[1],
                    "vectors": [v.tolist() for v in vectors_127],
                },
                indent=2,
            )
        )
        console.print(f"\nGolden values written to {GOLDEN_PATH.name}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
