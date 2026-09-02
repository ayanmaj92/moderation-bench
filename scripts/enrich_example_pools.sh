#!/bin/bash
set -e

# Hydrates the committed uri-only seed files in data_files/example_driven/example_pools/
# into the enriched pool files that utils/example_driven.py loads for
# runners/example_driven.py -- text/images/video fetched live from each post's PDS via
# utils/download_media.py (same tool scripts/get_data_files.sh uses for the top-level
# data_files/*_uri.jsonl files).
#
# Safe to re-run: download_media.py leaves already-enriched records alone and
# only retries ones that previously failed.

POOL_DIR="data_files/example_driven/example_pools"

for name in examples_random examples_random_safe examples_prototypical examples_prototypical_safe; do
    echo "Enriching ${name}..."
    # Resume from a previous partial run if one exists, so retries only hit the
    # records that failed last time -- otherwise start fresh from the seed file.
    if [ -f "${POOL_DIR}/${name}.jsonl" ]; then
        SRC="${POOL_DIR}/${name}.jsonl"
    else
        SRC="${POOL_DIR}/${name}_uri.jsonl"
    fi
    # --keep-failures so a record with a transient failure stays in the file
    # (rather than being dropped) and gets retried on the next re-run.
    python utils/download_media.py "$SRC" "${POOL_DIR}/${name}.jsonl" --keep-failures
done

echo "Example-driven example pools enriched successfully."
