# Example-driven example pools

This directory holds the static example pools used by `random` and
`prototypical` in-context selection.

## Seed files (shipped in git)

Each pool has a `*_uri.jsonl` seed file containing only what can't be
re-derived from the post itself — no post content, media, or other
personal data:

| File | Used for |
|---|---|
| `examples_random_uri.jsonl` | `random` selection, main per-label pool |
| `examples_random_safe_uri.jsonl` | `random` selection, safe/contrast examples |
| `examples_prototypical_uri.jsonl` | `prototypical` selection, main per-label pool |
| `examples_prototypical_safe_uri.jsonl` | `prototypical` selection, safe/contrast examples; also the default fallback for `contextual` when `dataset.safe_path` is unset |

Each line is `{"uri": <at:// post uri>, "cid": <post cid>, "label": <category>}`.

## Enriching into runnable pools

`main.py` (example-driven paradigm) needs the post text and media, not just the uri, so the
seed files must be hydrated before use. From the repo root:

```bash
bash scripts/enrich_example_pools.sh
```

This runs `utils/download_media.py` (the same tool `scripts/get_data_files.sh` uses
for the top-level `data_files/*_uri.jsonl` files) over each seed file,
fetching each post's current record from its PDS and downloading its
image/video blobs into sibling `images/`/`videos/` directories. The result
is written to the non-`_uri` filename `main.py` (example-driven paradigm) actually loads
(`examples_random.jsonl`, etc.) — those enriched files and the downloaded
media are gitignored, since they contain real post content. The script is
safe to re-run to retry any posts that failed the first time (e.g. a
transient network error); posts that no longer exist (deleted, blocked)
are marked `skip` and left out of the enriched pool for good.

Loaded via `utils.main_helpers.load_examples()`.
