"""Hydrate a uri-only contextual test set into a runnable one.

Input is one of data_files/example_driven/contextual_test_sets/*_uri.jsonl: each line is
a query post (`uri`, `cid`, optionally `label`) plus a `neighbors_per_label`
field, itself a dict of label -> list of neighbor posts (`uri`, `cid`, `label`,
`cosine_similarity`). Neither the query post nor its neighbors carry text/media --
just enough to identify and re-fetch them.

This fetches and downloads media for every post involved -- the query post AND
every neighbor across every label -- reusing the same fetch/download/validate
machinery as utils/download_media.py (DID resolution, PDS lookup, image/video
blob download with PIL/decord validation). Each record ends up with the same
fields utils/download_media.py fills in (cid, record, text, etype, images, video,
skip, failures); the query post keeps its enriched `neighbors_per_label` nested
inside it, same shape as the input.

This is much larger than a flat uri file: each query post carries ~90 neighbors
(10 per unsafe label), so a 1000-row test set is ~90,000 individual posts. To
keep memory bounded it processes CHUNK query rows at a time behind a pool of
exactly `--concurrency` workers (NOT one task per post -- that pile of suspended
coroutines is what used to OOM), streaming each finished chunk to disk.

Usage:
    python3 utils/enrich_contextual_test_set.py <input.jsonl> [output.jsonl] \
        [--concurrency N] [--chunk N] [--keep-failures]

If output.jsonl is omitted, writes <input>.with_media.jsonl next to the input.
Output is built at <output>.partial and atomically renamed on completion.
Safe to re-run to resume: feed the previous output back in as input -- records
that already have cid+record+(images|video|skip) are not re-fetched, and blobs
already on disk are re-validated, not re-downloaded.
"""
import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.download_media import process_record  # noqa: E402


def _iter_query_rows(path):
    """Stream query rows from the input file -- never load the whole file (an
    already-enriched input can be hundreds of MB). Tolerates a truncated final
    line, e.g. a *.partial left by a killed run."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"  [warn] skipping unparseable line in {path}", file=sys.stderr)


# Fields process_record() fills in; copied from an already-fetched post to every
# other slot that refers to the same uri (a post recurs as a neighbour across
# labels and query rows -- ~2x on average, up to hundreds of times for some).
_ENRICHED_KEYS = ("cid", "record", "text", "etype", "images", "video", "skip")


def _apply_enriched(dst: dict, src: dict) -> None:
    for k in _ENRICHED_KEYS:
        if k in src:
            dst[k] = src[k]


def _chunks(iterable, size):
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


async def _process_records(records, concurrency, session, did_cache, pds_cache,
                           sem, images_dir, videos_dir, stats, progress):
    """Run process_record over `records` with a fixed pool of `concurrency`
    workers -- only `concurrency` coroutines are ever live at once."""
    queue: asyncio.Queue = asyncio.Queue()
    for r in records:
        queue.put_nowait(r)

    async def worker():
        while True:
            try:
                rec = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await process_record(rec, session, did_cache, pds_cache, sem,
                                     images_dir, videos_dir, stats)
            except Exception as e:  # never let one bad record kill the pool
                rec.setdefault("failures", []).append(f"process_record crashed: {e}")
            finally:
                progress[0] += 1
                if progress[0] % 500 == 0:
                    print(f"  {progress[0]} records processed", file=sys.stderr)
                queue.task_done()

    await asyncio.gather(*[worker() for _ in range(max(1, concurrency))])


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path, nargs="?")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="number of concurrent fetch workers (default 20)")
    parser.add_argument("--chunk", type=int, default=20,
                        help="query rows processed + flushed per batch (default 20 "
                             "-> ~1800 records held at once)")
    parser.add_argument("--keep-failures", action="store_true",
                        help="write records with failures to the output too "
                             "(default: drop them so a re-run retries them)")
    args = parser.parse_args()

    out_path = args.output_jsonl or args.input_jsonl.with_suffix(".with_media.jsonl")
    tmp_path = out_path.with_name(out_path.name + ".partial")
    data_dir = args.input_jsonl.resolve().parent
    images_dir = data_dir / "images"
    videos_dir = data_dir / "videos"

    sem = asyncio.Semaphore(args.concurrency)
    stats: Counter = Counter()
    did_cache: dict = {}   # shared across chunks -- neighbors reuse DIDs/PDS URLs
    pds_cache: dict = {}
    seen: dict = {}        # uri -> first successfully enriched record for that post
    progress = [0]

    n_rows_in = n_rows_out = 0
    n_records = n_rec_failed = 0

    async with aiohttp.ClientSession() as session:
        with open(tmp_path, "w") as out:
            for chunk in _chunks(_iter_query_rows(args.input_jsonl), args.chunk):
                n_rows_in += len(chunk)

                flat = []
                for row in chunk:
                    flat.append(row)
                    for neighbors in (row.get("neighbors_per_label") or {}).values():
                        flat.extend(neighbors)

                # Fetch each distinct post at most once: fill known ones from
                # `seen`, hand one representative per remaining uri to the pool.
                todo, todo_by_uri = [], {}
                for rec in flat:
                    u = rec.get("uri")
                    if u and u in seen:
                        _apply_enriched(rec, seen[u])
                    elif u and u in todo_by_uri:
                        continue  # sibling already queued; backfilled below
                    else:
                        if u:
                            todo_by_uri[u] = rec
                        todo.append(rec)

                await _process_records(todo, args.concurrency, session, did_cache,
                                       pds_cache, sem, images_dir, videos_dir, stats, progress)

                for u, rec in todo_by_uri.items():
                    if not rec.get("failures"):
                        seen[u] = rec
                # Backfill every sibling slot from its representative -- carrying
                # failures too, so a duplicate of a post that failed to fetch is
                # dropped by the --keep-failures filter just like the original.
                for rec in flat:
                    u = rec.get("uri")
                    if not u:
                        continue
                    src = seen.get(u) or todo_by_uri.get(u)
                    if src is not None and rec is not src:
                        _apply_enriched(rec, src)
                        if src.get("failures"):
                            rec["failures"] = list(src["failures"])

                n_records += len(flat)
                n_rec_failed += sum(1 for r in flat if r.get("failures"))

                for row in chunk:
                    if not args.keep_failures:
                        for lbl, neighbors in (row.get("neighbors_per_label") or {}).items():
                            row["neighbors_per_label"][lbl] = [n for n in neighbors if not n.get("failures")]
                    if args.keep_failures or not row.get("failures"):
                        out.write(json.dumps(row) + "\n")
                        n_rows_out += 1
                out.flush()
                print(f"  {n_rows_in} query rows done ({n_rows_out} written, "
                      f"{progress[0]} records)", file=sys.stderr)

    os.replace(tmp_path, out_path)

    n_top_failed = n_rows_in - n_rows_out if not args.keep_failures else \
        "n/a (--keep-failures)"
    print(f"wrote {out_path}")
    print(f"{n_rows_out}/{n_rows_in} query rows written, {n_records} total records fetched "
          f"({n_rows_in} top-level + {n_records - n_rows_in} neighbors)")
    print(f"failures: {n_rec_failed} records"
          f"{'' if args.keep_failures else f' ; {n_top_failed} query rows dropped from output'}")
    print(f"  image blobs: {stats['image_validation_failures']} failed validation, "
          f"{stats['image_fetch_failures']} failed to fetch")
    print(f"  video blobs: {stats['video_validation_failures']} failed validation, "
          f"{stats['video_fetch_failures']} failed to fetch")


if __name__ == "__main__":
    asyncio.run(main())
