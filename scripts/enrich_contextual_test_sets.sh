#!/bin/bash
set -e

# Contextual counterpart of enrich_example_pools.sh.
#
# Hydrates the committed uri-only seeds in
# data_files/example_driven/contextual_test_sets/ into runnable test sets: fetches
# text + images/video for every query post AND all ~90 of its per-label
# neighbours, via utils/enrich_contextual_test_set.py (which drives the
# same utils/download_media.py fetch/download/validate primitive over the seed's
# nested `neighbors_per_label` structure -- that nesting is why it's a Python
# script and this is just the loop around it).
#
# MUCH heavier than the pools: each seed is ~1000 rows x ~90 neighbours ~= 90k
# posts, so expect hours per category and Bluesky rate limiting -- pass a lower
# concurrency (arg 1, default 10) if you see a lot of fetch failures.
#
# Safe to re-run to resume: the Python script processes CHUNK query rows at a
# time and streams each finished chunk to <OUT>.partial (atomically renamed to
# <OUT> on completion), so memory stays bounded and a killed run keeps its
# progress. On the next run this feeds the most complete file back in as input
# -- <OUT> if it finished, else a leftover <OUT>.partial -- and records that
# already have cid+record+(images|video|skip) are not re-fetched (blobs already
# on disk are re-validated, not re-downloaded).
#
# Run from the repo root.

DATA_DIR="data_files/example_driven/contextual_test_sets"
CONCURRENCY="${1:-10}"
# Same four subset names as the instruction-driven benchmark.
SUBSETS=(moderated random near-moderated safe)

for subset in "${SUBSETS[@]}"; do
    SEED="${DATA_DIR}/${subset}_top10_per_label_flattened_uri.jsonl"
    OUT="${DATA_DIR}/${subset}_top10_per_label_flattened.jsonl"

    if [ ! -f "$SEED" ]; then
        echo "Skipping ${subset}: seed not found at ${SEED}"
        continue
    fi

    # Resume from whichever prior artefact is most complete, else the seed.
    if [ -f "$OUT" ]; then
        IN="$OUT"
    elif [ -f "${OUT}.partial" ]; then
        echo "  resuming ${subset} from a leftover ${OUT}.partial"
        mv "${OUT}.partial" "$OUT"
        IN="$OUT"
    else
        IN="$SEED"
    fi

    echo "Enriching ${subset} (concurrency=${CONCURRENCY})..."
    python utils/enrich_contextual_test_set.py "$IN" "$OUT" \
        --concurrency "$CONCURRENCY" --keep-failures
done

echo "Contextual test sets enriched successfully."
