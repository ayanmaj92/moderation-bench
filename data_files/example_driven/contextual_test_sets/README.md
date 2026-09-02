# Contextual test sets

These are the test sets used with `-ics contextual` (see
`scripts/slurm_example_driven_full_sweep.sh` / `slurm_example_per_group_sweep.sh`,
which sweep all four). Each query post already carries a precomputed
`neighbors_per_label` field, as `contextual` selection requires.

## Seed files (shipped in git)

One per benchmark subset, same four subset names the instruction-driven pipeline
uses (`data_files/<subset>_uri.jsonl`):

| File | Subset |
|---|---|
| `safe_top10_per_label_flattened_uri.jsonl` | `safe` |
| `moderated_top10_per_label_flattened_uri.jsonl` | `moderated` |
| `random_top10_per_label_flattened_uri.jsonl` | `random` |
| `near-moderated_top10_per_label_flattened_uri.jsonl` | `near-moderated` |

Each line is:

```json
{
  "uri": "<query post uri>",
  "cid": "<query post cid>",
  "label": "<category, if applicable>",
  "neighbors_per_label": {
    "<label>": [
      {"uri": "<neighbor uri>", "cid": "<neighbor cid>", "label": "<label>", "cosine_similarity": 0.0},
      ...
    ],
    ...
  }
}
```

No post text, images, or video are included — just enough to identify and
re-fetch each post, both the query post and all ~90 of its neighbors (10 per
unsafe label).

## Enriching into runnable test sets

All four at once (writes `<subset>_top10_per_label_flattened.jsonl` next to each seed):

```bash
bash scripts/enrich_contextual_test_sets.sh          # optional arg: concurrency (default 10)
```

Or one subset:

```bash
python utils/enrich_contextual_test_set.py \
  data_files/example_driven/contextual_test_sets/random_top10_per_label_flattened_uri.jsonl \
  data_files/example_driven/contextual_test_sets/random_top10_per_label_flattened.jsonl
```

This fetches and downloads media for every post involved — the query post
*and* every neighbor — reusing the same fetch/download/validation machinery
as `utils/download_media.py` (used for the top-level `data_files/*_uri.jsonl`
files and the example-driven pools). See that script's docstring for the field
semantics it fills in.

Lower `--concurrency` if you see a lot of fetch failures
(Bluesky rate-limits aggressively). It's safe to re-run to resume: pass the
previous run's output back in as input (with `--keep-failures` on the run
that produced it) and only records that still need fetching get retried.
