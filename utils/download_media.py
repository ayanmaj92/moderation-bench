"""Fetch post records and download their images/video for a batch of posts given their URIs.

Input is a jsonl file where every line has a "uri" (either "at://did/collection/rkey"
or "https://bsky.app/profile/<handle-or-did>/post/<rkey>") plus whatever other keys.

For each record this fills in, if not already present:
    cid       -- the post record's current CID
    record    -- the full post record fetched from its PDS (text, embed, langs, ...)
    text      -- record["text"], hoisted to the top level for convenience
    etype     -- the embed's $type (after unwrapping recordWithMedia), "" if none
    images    -- [{"file": <path>, "alt": <str>}, ...] for downloaded image/thumb blobs
    video     -- path to the downloaded video file, or null
    skip      -- true if the record could not be resolved (deleted post, bad uri, ...)
    failures  -- list of error strings

Media is saved next to the metadata file, in sibling images/ and videos/ directories
(same layout the existing dataset already uses): images/<last 2 chars of blob cid>/<blob
cid>.jpeg, videos/<last 2 chars of post cid>/<post cid>.mp4. Records that already have
"cid" and "record" are not re-fetched from the network (only media is filled in); records
that already have "images"/"video"/"skip" are left alone entirely, so it's safe to re-run
a batch to resume after a partial failure. By default, records that end up with any
"failures" are dropped from the output jsonl entirely (they're not written at all, so
re-running the same input will retry them from scratch); pass --keep-failures to write
them anyway. The failure count is always printed regardless of this setting.

Every downloaded (and every already-cached) image/video is decoded with the same
libraries the eval pipeline uses (PIL for images, decord for video, see utils/media.py)
before being accepted -- a truncated download or corrupt blob is treated as a failed
download (retried on the next run) rather than being written out as if it were fine.

Usage:
    python3 download_media_for_batch.py <metadata.jsonl> [output.jsonl] [--concurrency N]

If output.jsonl is omitted, writes <metadata>.with_media.jsonl next to the input.
"""
import argparse
import asyncio
import contextlib
import json
import os
import sys
import tempfile
import threading
from collections import Counter
from pathlib import Path

import aiofiles
import aiohttp
import decord
from PIL import Image

# decord/ffmpeg write decode diagnostics straight to C-level stderr (fd 2), which
# floods the console with "Invalid NAL unit size" / "pix_fmt to value -1" lines
# whenever a corrupt or truncated video blob is probed. The Python exception is
# still raised and handled -- only the chatter is unwanted. The lock keeps at most
# one thread pointing fd 2 at /dev/null at a time (verify runs in a thread pool).
_STDERR_LOCK = threading.Lock()


@contextlib.contextmanager
def _suppress_c_stderr():
    with _STDERR_LOCK:
        try:
            saved = os.dup(2)
        except OSError:
            yield
            return
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, 2)
            yield
        finally:
            os.dup2(saved, 2)
            os.close(devnull)
            os.close(saved)


def parse_uri(uri: str) -> tuple[str, str, str]:
    """Returns (did_or_handle, collection, rkey). did_or_handle may still need resolving."""
    if uri.startswith("at://"):
        parts = uri.removeprefix("at://").split("/")
        if len(parts) < 3:
            raise ValueError(f"unrecognized at:// uri: {uri}")
        return parts[0], parts[1], parts[2]
    if "bsky.app/profile/" in uri:
        parts = uri.rstrip("/").split("/")
        rkey = parts[-1]
        handle_or_did = parts[parts.index("profile") + 1]
        return handle_or_did, "app.bsky.feed.post", rkey
    raise ValueError(f"unrecognized post uri: {uri}")


def classify_embed(embed: dict | None) -> tuple[bool, bool, bool, str]:
    """Returns (has_image, has_video, has_external_thumb, etype), unwrapping recordWithMedia
    like the live pipeline's resolver.py does. An app.bsky.embed.external (link-preview
    card) can carry its own thumbnail blob -- this dataset's own generator downloads
    those too (alt="external_thumb"), so match that convention rather than dropping them."""
    if not embed:
        return False, False, False, ""
    etype = embed.get("$type", "")
    src = embed.get("media", {}) if etype == "app.bsky.embed.recordWithMedia" else embed
    stype = src.get("$type", "")
    has_external_thumb = stype == "app.bsky.embed.external" and bool(
        src.get("external", {}).get("thumb", {}).get("ref", {}).get("$link")
    )
    return stype == "app.bsky.embed.images", stype.startswith("app.bsky.embed.video"), has_external_thumb, stype


async def resolve_did(session: aiohttp.ClientSession, handle_or_did: str, cache: dict) -> str | None:
    if handle_or_did.startswith("did:"):
        return handle_or_did
    if handle_or_did in cache:
        return cache[handle_or_did]
    try:
        async with session.get(
            "https://public.api.bsky.app/xrpc/com.atproto.identity.resolveHandle",
            params={"handle": handle_or_did}, timeout=10,
        ) as resp:
            if resp.status != 200:
                cache[handle_or_did] = None
                return None
            did = (await resp.json()).get("did")
    except Exception:
        did = None
    cache[handle_or_did] = did
    return did


async def get_pds_endpoint(session: aiohttp.ClientSession, did: str, cache: dict) -> str | None:
    if did in cache:
        return cache[did]
    try:
        async with session.get(f"https://plc.directory/{did}/data", timeout=10) as resp:
            if resp.status != 200:
                cache[did] = None
                return None
            data = await resp.json()
    except Exception:
        cache[did] = None
        return None
    services = data.get("services", {})
    services = services.values() if isinstance(services, dict) else services
    endpoint = None
    for svc in services:
        if svc.get("type") == "AtprotoPersonalDataServer":
            endpoint = svc.get("serviceEndpoint") or svc.get("endpoint")
            break
    cache[did] = endpoint
    return endpoint


async def fetch_post_record(session: aiohttp.ClientSession, pds: str, did: str,
                             collection: str, rkey: str) -> dict | None:
    try:
        url = f"{pds}/xrpc/com.atproto.repo.getRecord"
        params = {"repo": did, "collection": collection, "rkey": rkey}
        async with session.get(url, timeout=15, params=params) as resp:
            if resp.status != 200:
                return None
            return await resp.json()
    except Exception:
        return None


def _image_is_valid(path: Path) -> bool:
    """Decodes the file the same way downstream consumers (utils/media.py's
    resize path) do, so a truncated/corrupt JPEG that would blow up in vLLM
    preprocessing is caught here instead."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:  # verify() leaves the file unusable; reopen to decode
            img.load()
        return True
    except Exception:
        return False


def _video_is_valid(path: Path) -> bool:
    """Mirrors utils/media.py's decord.VideoReader frame-extraction path, so a
    video that would fail there (bad container, zero readable frames) is caught here."""
    try:
        decord.bridge.set_bridge("native")
        with _suppress_c_stderr():
            vr = decord.VideoReader(str(path), ctx=decord.cpu(0))
            n = len(vr)
        return n > 0
    except Exception:
        return False


async def verify_media(path: Path, is_video: bool) -> bool:
    return await asyncio.to_thread(_video_is_valid if is_video else _image_is_valid, path)


async def download_blob(session: aiohttp.ClientSession, pds: str, did: str, cid: str,
                         out_path: Path, is_video: bool) -> tuple[bool, str | None]:
    """Returns (ok, failure_reason). failure_reason is "validation" when a file was
    fetched (or already cached) but failed the PIL/decord check, "fetch" for any
    HTTP/network failure, or None on success."""
    if out_path.exists():
        if await verify_media(out_path, is_video):
            return True, None
        out_path.unlink(missing_ok=True)  # cached file is corrupt -- re-download it

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Unique temp name per download: the same blob is often fetched by several
    # workers at once (a post can appear as a neighbour hundreds of times), and a
    # shared "<name>.part" path let them interleave writes into one file (corrupt
    # bytes) or rename/unlink it out from under each other ("No such file or
    # directory"). mkstemp in the target dir keeps os.replace atomic + on-fs.
    fd, tmp_name = tempfile.mkstemp(dir=out_path.parent,
                                    prefix=out_path.name + ".", suffix=".part")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        timeout = aiohttp.ClientTimeout(total=180 if is_video else 30)
        url = f"{pds}/xrpc/com.atproto.sync.getBlob?did={did}&cid={cid}"
        async with session.get(url, timeout=timeout) as resp:
            if resp.status != 200:
                return False, "fetch"
            async with aiofiles.open(tmp, "wb") as f:
                async for chunk in resp.content.iter_chunked(16384):
                    await f.write(chunk)
        if not await verify_media(tmp, is_video):
            tmp.unlink(missing_ok=True)
            return False, "validation"
        os.replace(tmp, out_path)  # atomic; concurrent winners both had a valid file
        return True, None
    except Exception:
        tmp.unlink(missing_ok=True)
        return False, "fetch"


async def process_record(record: dict, session: aiohttp.ClientSession, did_cache: dict,
                          pds_cache: dict, sem: asyncio.Semaphore, images_dir: Path,
                          videos_dir: Path, stats: Counter) -> dict:
    if record.get("skip") or record.get("images") or record.get("video"):
        return record  # already processed -- nothing to do

    uri = record.get("uri")
    if not uri:
        record["skip"] = True
        record["failures"] = record.get("failures", []) + ["missing uri"]
        return record

    try:
        did_or_handle, collection, rkey = parse_uri(uri)
    except ValueError as e:
        record["skip"] = True
        record["failures"] = record.get("failures", []) + [str(e)]
        return record

    async with sem:
        did = await resolve_did(session, did_or_handle, did_cache)
    if not did:
        record["skip"] = True
        record["failures"] = record.get("failures", []) + [f"could not resolve did: {did_or_handle}"]
        return record

    async with sem:
        pds = await get_pds_endpoint(session, did, pds_cache)
    if not pds:
        record["skip"] = True
        record["failures"] = record.get("failures", []) + ["no pds endpoint"]
        return record

    if not record.get("cid") or not record.get("record"):
        async with sem:
            fetched = await fetch_post_record(session, pds, did, collection, rkey)
        if fetched is None:
            record["skip"] = True
            record["failures"] = record.get("failures", []) + ["post record not found"]
            record.setdefault("images", [])
            record.setdefault("video", None)
            return record
        record["cid"] = fetched.get("cid")
        record["record"] = fetched.get("value", {})

    post_record = record.get("record") or {}
    record["text"] = post_record.get("text", "")
    record.setdefault("images", [])
    record.setdefault("video", None)

    embed = post_record.get("embed") or {}
    has_image, has_video, has_external_thumb, etype = classify_embed(embed)
    record["etype"] = etype
    if not (has_image or has_video or has_external_thumb):
        return record

    media_src = embed.get("media", embed) if embed.get("$type") == "app.bsky.embed.recordWithMedia" else embed

    if has_image:
        for img in media_src.get("images", []):
            link_cid = img.get("image", {}).get("ref", {}).get("$link")
            if not link_cid:
                continue
            path = images_dir / link_cid[-2:] / f"{link_cid}.jpeg"
            async with sem:
                ok, reason = await download_blob(session, pds, did, link_cid, path, is_video=False)
            if ok:
                record["images"].append({"file": str(path), "alt": img.get("alt", "")})
            else:
                stats[f"image_{reason}_failures"] += 1
                record["failures"] = record.get("failures", []) + [f"image download failed ({reason}): {link_cid}"]

    if has_external_thumb:
        link_cid = media_src.get("external", {}).get("thumb", {}).get("ref", {}).get("$link")
        path = images_dir / link_cid[-2:] / f"{link_cid}.jpeg"
        async with sem:
            ok, reason = await download_blob(session, pds, did, link_cid, path, is_video=False)
        if ok:
            record["images"].append({"file": str(path), "alt": "external_thumb"})
        else:
            stats[f"image_{reason}_failures"] += 1
            record["failures"] = record.get("failures", []) + [f"external thumb download failed ({reason}): {link_cid}"]

    if has_video:
        video_obj = media_src.get("video", {})
        link_cid = video_obj.get("ref", {}).get("$link")
        post_cid = record.get("cid")
        if link_cid and post_cid:
            path = videos_dir / post_cid[-2:] / f"{post_cid}.mp4"
            async with sem:
                ok, reason = await download_blob(session, pds, did, link_cid, path, is_video=True)
            if ok:
                record["video"] = str(path)
            else:
                stats[f"video_{reason}_failures"] += 1
                record["failures"] = record.get("failures", []) + [f"video download failed ({reason}): {link_cid}"]

    return record


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path, nargs="?")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--keep-failures", action="store_true",
                         help="write records with failures to the output jsonl too "
                              "(default: drop them so a re-run retries them from scratch)")
    args = parser.parse_args()

    out_path = args.output_jsonl or args.metadata_jsonl.with_suffix(".with_media.jsonl")
    data_dir = args.metadata_jsonl.resolve().parent
    images_dir = data_dir / "images"
    videos_dir = data_dir / "videos"

    with open(args.metadata_jsonl) as f:
        records = [json.loads(line) for line in f]

    sem = asyncio.Semaphore(args.concurrency)
    n_done = 0
    stats: Counter = Counter()
    async with aiohttp.ClientSession() as session:
        did_cache: dict[str, str | None] = {}
        pds_cache: dict[str, str | None] = {}
        tasks = [process_record(r, session, did_cache, pds_cache, sem, images_dir, videos_dir, stats)
                 for r in records]
        results = []
        for coro in asyncio.as_completed(tasks):
            results_r = await coro
            results.append(results_r)
            n_done += 1
            if n_done % 100 == 0:
                print(f"  {n_done}/{len(records)} processed", file=sys.stderr)

    n_failed = sum(1 for r in results if r.get("failures"))
    written = results if args.keep_failures else [r for r in results if not r.get("failures")]

    with open(out_path, "w") as f:
        for r in written:
            f.write(json.dumps(r) + "\n")

    n_images = sum(1 for r in written if r.get("images"))
    n_video = sum(1 for r in written if r.get("video"))
    n_skipped = sum(1 for r in written if r.get("skip"))
    print(f"wrote {out_path}")
    print(f"{n_images} posts with images, {n_video} posts with video, "
          f"{n_skipped} skipped, {n_failed} with failures"
          f"{'' if args.keep_failures else ' (dropped from output)'}")
    print(f"  image blobs: {stats['image_validation_failures']} failed validation, "
          f"{stats['image_fetch_failures']} failed to fetch")
    print(f"  video blobs: {stats['video_validation_failures']} failed validation, "
          f"{stats['video_fetch_failures']} failed to fetch")


if __name__ == "__main__":
    asyncio.run(main())
