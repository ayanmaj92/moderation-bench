#!/usr/bin/env python3
"""Download the ModerationBench-4K dataset from the Hugging Face Hub into this repo.

Files land under <repo>/data_files/. Needs a HF token with access to the dataset
(--token, $HF_TOKEN, or `huggingface-cli login`). Run the pipeline from the repo root
afterwards so the data_files/... paths resolve.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HF_REPO_ID = "ayanmaj/ModerationBench-4K"
REPO_TYPE = "dataset"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--token", default=None, help="HF token (else $HF_TOKEN / cached login)")
    args = ap.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        sys.exit("huggingface_hub not installed -- `pip install huggingface_hub`")

    if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
        try:
            import hf_transfer  # noqa: F401
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        except ImportError:
            pass

    # fail early and clearly if the repo is missing / private / gated for this token
    try:
        HfApi(token=token).repo_info(HF_REPO_ID, repo_type=REPO_TYPE)
    except Exception as e:  # noqa: BLE001
        sys.exit(
            f"cannot access {HF_REPO_ID}: {type(e).__name__}: {e}\n"
            f"request access at https://huggingface.co/datasets/{HF_REPO_ID} "
            "and pass a valid token (--token / $HF_TOKEN)."
        )

    print(f"downloading {HF_REPO_ID} -> {PROJECT_ROOT}/data_files")
    # only the data -- not the dataset repo's own README.md / .gitattributes,
    # which would clobber this code repo's versions.
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(PROJECT_ROOT),
        allow_patterns=["data_files/**"],
        token=token,
    )

    # sanity check: every media path in every metadata file resolves from the repo root
    missing = 0
    for meta in sorted((PROJECT_ROOT / "data_files").glob("*_metadata.jsonl")):
        rows = [json.loads(l) for l in meta.open() if l.strip()]
        refs = [m["file"] for r in rows for m in (r.get("images") or [])]
        refs += [r["video"] for r in rows if r.get("video")]
        gone = sum(1 for p in refs if not (PROJECT_ROOT / p).exists())
        missing += gone
        print(f"  {meta.name}: {len(rows)} posts, {len(refs)} media refs, {gone} missing")

    if missing:
        sys.exit(f"\n{missing} media files missing -- re-run to resume the download.")
    print("\nOK. Everything downloaded and verified.")


if __name__ == "__main__":
    main()
