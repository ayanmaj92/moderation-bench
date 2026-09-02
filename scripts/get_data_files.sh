#!/bin/bash
set -e

# Reconstructs the four instruction-driven benchmark subsets from their committed
# uri-only seeds: fetches each post's record + media from its PDS via
# utils/download_media.py, writing data_files/<subset>_metadata.jsonl.
# (Option A in the README -- the gated Hugging Face dataset -- is the complete,
# pre-fetched alternative.)
#
# Run from anywhere: this cd's to the repo root so the relative paths below
# resolve. Safe to re-run to resume: already-fetched records are left alone.

cd "$(dirname "$0")/.."

for subset in moderated random near-moderated safe; do
    echo "Fetching ${subset}..."
    python utils/download_media.py \
        "data_files/${subset}_uri.jsonl" "data_files/${subset}_metadata.jsonl"
done

echo "Media and content for data files downloaded successfully."
