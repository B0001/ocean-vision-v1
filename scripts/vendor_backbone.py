"""Build-time vendoring for the pretrained video transformer backbone
(ocean-vision-v1-dxj, spec 3's "ViT Embeddings" branch of the Spatiotemporal
Feature Extractor).

This is a **build-time-only** tool. It is the one place in this repo that is
allowed to touch the network for the ViT backbone: it downloads a pinned
Hugging Face Hub revision, checksums every vendored file, and records the
model's license into `models/<name>/MANIFEST.json`. `src/ov1/model/backbone.py`
(the runtime loader) only ever reads the directory this script writes, with
`local_files_only=True` -- it never calls the Hub.

Run once, by hand, at build/setup time:

    uv run python scripts/vendor_backbone.py

Re-running is idempotent: it skips the download if the local files already
match the pinned revision's checksums (via `huggingface_hub.snapshot_download`'s
own cache-and-copy behavior) and always rewrites MANIFEST.json.

Not part of the test suite: `tests/` never imports this module, since the
repo's tests must not touch the network (spec 5.2) and this script's entire
job is a network call.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

#: MCG-NJU/videomae-base: the VideoMAE (Tong et al. 2022, arXiv:2203.12602)
#: base-size masked-autoencoder pretraining checkpoint -- 12-layer, 768-hidden
#: ViT-Base encoder over 16-frame x 224x224 RGB clips (2x16x16 tubelets). This
#: script vendors it via `VideoMAEModel` (the bare encoder), not
#: `VideoMAEForPreTraining` (encoder+decoder+pixel-reconstruction head) --
#: spec 3's diagram wants "ViT Embeddings" out of this branch, and the MAE
#: decoder exists only to produce a pretraining loss, not embeddings for a
#: downstream head.
#:
#: Chosen over a `*-finetuned-kinetics` checkpoint (also available from the
#: same org) because a Kinetics action-classification head is irrelevant here
#: -- this branch needs the general-purpose visual encoder, not a 400-class
#: action-recognition readout, and the encoder weights are shared regardless
#: of which head was fine-tuned on top.
REPO_ID = "MCG-NJU/videomae-base"

#: Pinned to a specific commit, not "main" -- so a re-run of this script
#: three months from now vendors the exact bytes this bead's evidence was
#: gathered against, not whatever the Hub's default branch has drifted to.
#: Taken from `GET https://huggingface.co/api/models/MCG-NJU/videomae-base`
#: ("sha" field) on 2026-08-11.
REVISION = "dc740ceda42fce44faed2ea03c6d447db72f6af9"

#: License as reported by the Hub's model card metadata for this repo/revision
#: (`cardData.license` in the same API response). CC-BY-NC-4.0 is a
#: **non-commercial** license -- see the flag this script prints and the
#: handoff for why that is a decision for a human, not this script, to
#: resolve before this backbone ships in a commercial deployment.
LICENSE = "cc-by-nc-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc/4.0/"

#: Only what the runtime loader needs. Excludes `pytorch_model.bin` (the
#: safetensors file is the same weights in a safer format -- no reason to
#: vendor both) and `README.md` (model card prose, not needed to load).
ALLOW_PATTERNS = ["config.json", "model.safetensors", "preprocessor_config.json"]

MODEL_STORE_ROOT = Path(__file__).resolve().parent.parent / "models"
LOCAL_DIR_NAME = "videomae-base"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vendor() -> Path:
    local_dir = MODEL_STORE_ROOT / LOCAL_DIR_NAME
    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        local_dir=local_dir,
        allow_patterns=ALLOW_PATTERNS,
    )

    vendored_files = sorted(p for p in local_dir.iterdir() if p.is_file() and p.name != "MANIFEST.json")
    checksums = {p.name: _sha256(p) for p in vendored_files}

    manifest = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "source_url": f"https://huggingface.co/{REPO_ID}/tree/{REVISION}",
        "license": LICENSE,
        "license_url": LICENSE_URL,
        "license_note": (
            "CC-BY-NC-4.0 is non-commercial. Recorded as required by this "
            "bead's acceptance criteria; whether a non-commercial license is "
            "acceptable for this deployment is a legal/product decision this "
            "script and its caller do not make."
        ),
        "files_sha256": checksums,
    }
    (local_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    return local_dir


if __name__ == "__main__":
    path = vendor()
    print(f"vendored {REPO_ID}@{REVISION} -> {path}")
    print(f"license: {LICENSE} ({LICENSE_URL}) -- NON-COMMERCIAL, see MANIFEST.json license_note")
