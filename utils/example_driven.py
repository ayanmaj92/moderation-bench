"""Example-driven in-context example assembly.

Pure data-wrangling: selecting / capping / chunking demonstration examples for
the example-driven paradigm. No torch / vLLM imports, so this is unit-testable
without a GPU.

Three selection strategies:
  * ``random`` / ``prototypical`` -- fixed pools on disk under
    ``data_files/example_driven/example_pools/`` (paths hardcoded here; edit
    ``_EXAMPLE_POOL_DIR`` or replace the files to use a different pool).
  * ``contextual`` -- per-query nearest neighbours carried on each test row
    in a ``neighbors_per_label`` map, plus a fixed safe/contrast pool.

Each strategy has a flat variant (one combined prompt) and a ``*_groups``
variant for ``example_per_group`` (chunked multi-call + majority vote).
"""

import os
from collections import defaultdict

from utils.main_helpers import load_examples

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXAMPLE_POOL_DIR = os.path.join(_REPO_ROOT, "data_files", "example_driven", "example_pools")
_DEFAULT_SAFE_FALLBACK = os.path.join(_EXAMPLE_POOL_DIR, "examples_prototypical_safe.jsonl")

_MAX_GROUPS = 10  # cap on chunks per query item

_stats = {
    "examples_seen": 0,
    "examples_video_only_dropped": 0,
    "image_paths_checked": 0,
    "image_paths_missing": 0,
}


def reset_media_stats():
    for k in _stats:
        _stats[k] = 0


def log_media_stats_summary():
    print(
        "[example_driven] media stats -- examples_seen={examples_seen} "
        "video_only_dropped={examples_video_only_dropped} "
        "image_paths_checked={image_paths_checked} "
        "image_paths_missing={image_paths_missing}".format(**_stats)
    )


def _check_images(images, source_label):
    for p in images or []:
        _stats["image_paths_checked"] += 1
        if p and not os.path.exists(p):
            _stats["image_paths_missing"] += 1
            print(f"[example_driven] WARNING: image path does not exist ({source_label}): {p}")
    return images


def _keep_example(images, video, source_label):
    """Drop examples that have video but no images."""
    _stats["examples_seen"] += 1
    if not images and video:
        _stats["examples_video_only_dropped"] += 1
        # print(f"[example_driven] Dropping video-only example ({source_label}), "
        #       f"example-driven examples don't support video: video={video}")
        return False
    return True


def _load_static_pool(select):
    if select == "random":
        return (
            load_examples(os.path.join(_EXAMPLE_POOL_DIR, "examples_random.jsonl")),
            load_examples(os.path.join(_EXAMPLE_POOL_DIR, "examples_random_safe.jsonl")),
        )
    if select == "prototypical":
        return (
            load_examples(os.path.join(_EXAMPLE_POOL_DIR, "examples_prototypical.jsonl")),
            load_examples(os.path.join(_EXAMPLE_POOL_DIR, "examples_prototypical_safe.jsonl")),
        )
    raise ValueError(f"Unsupported in_context_select for static pools: {select}")


def _append_safe_examples(texts, images, labels, safe_examples, number_example, source_label):
    kept = 0
    for item in safe_examples:
        if kept >= number_example:
            break
        if not _keep_example(item.get("images"), item.get("video"), source_label):
            continue
        texts.append(item.get("text"))
        images.append(_check_images(item.get("images"), source_label))
        labels.append(item.get("label"))
        kept += 1
    return texts, images, labels


def _chunk_examples(texts, images, labels, chunk_size):
    """Split a flat (texts, images, labels) example list into consecutive
    chunks of ``chunk_size``, capped at ``_MAX_GROUPS`` chunks."""
    n = len(labels)
    chunks = []
    for i in range(0, n, chunk_size):
        if len(chunks) >= _MAX_GROUPS:
            break
        chunks.append((texts[i:i + chunk_size], images[i:i + chunk_size], labels[i:i + chunk_size]))
    return chunks or [([], [], [])]


def build_random_or_prototypical_examples(select, number_example):
    """Static, batch-broadcast example pool (same examples for every query item):
    up to ``number_example`` per unsafe label, then up to ``number_example`` safe
    examples appended. Returns flat (texts, images, labels)."""
    examples, safe_examples = _load_static_pool(select)

    label_to_items = defaultdict(list)
    for obj in examples:
        label_to_items[obj.get("label")].append(obj)

    texts, images, labels = [], [], []
    for label, items in label_to_items.items():
        kept = 0
        for item in items:
            if kept >= number_example:
                break
            if not _keep_example(item.get("images"), item.get("video"), f"{select} pool [{label}]"):
                continue
            texts.append(item.get("text"))
            images.append(_check_images(item.get("images"), f"{select} pool [{label}]"))
            labels.append(item.get("label"))
            kept += 1
    return _append_safe_examples(texts, images, labels, safe_examples, number_example, f"{select} safe pool")


def build_contextual_examples(neighbors_per_label, safe_path, number_example):
    """Per-item nearest-neighbour examples, capped at ``number_example`` PER LABEL
    (same balanced-per-category convention as random/prototypical), flattened into
    one prompt, plus a fixed safe/contrast pool appended on top."""
    all_safe_examples = load_examples(safe_path) if safe_path else load_examples(_DEFAULT_SAFE_FALLBACK)

    texts, images, labels = [], [], []
    for lbl, neighbors in (neighbors_per_label or {}).items():
        sorted_neighbors = sorted(neighbors, key=lambda n: n.get("cosine_similarity", 0), reverse=True)
        kept = 0
        for n in sorted_neighbors:
            if kept >= number_example:
                break
            n_images = [i if isinstance(i, str) else i.get("file") for i in (n.get("images") or [])]
            if not _keep_example(n_images, n.get("video"), f"contextual neighbor [{lbl}]"):
                continue
            texts.append(n.get("text"))
            images.append(_check_images(n_images, f"contextual neighbor [{lbl}]"))
            labels.append(lbl)
            kept += 1
    return _append_safe_examples(texts, images, labels, all_safe_examples, number_example, "contextual safe pool")


def build_static_groups(select, number_example):
    """Per-group version of ``build_random_or_prototypical_examples``: the same
    static per-label pool, chunked into ``number_example``-sized groups (capped at
    ``_MAX_GROUPS``), with its own slice of the safe pool appended to each group.
    Computed once since these pools are identical for every query item."""
    examples, safe_examples = _load_static_pool(select)

    label_to_items = defaultdict(list)
    for obj in examples:
        label_to_items[obj.get("label")].append(obj)

    texts, images, labels = [], [], []
    for label, items in label_to_items.items():
        kept = 0
        for item in items:
            if kept >= number_example:
                break
            if not _keep_example(item.get("images"), item.get("video"), f"{select} pool [{label}]"):
                continue
            texts.append(item.get("text"))
            images.append(_check_images(item.get("images"), f"{select} pool [{label}]"))
            labels.append(item.get("label"))
            kept += 1

    groups = _chunk_examples(texts, images, labels, number_example)
    return [
        _append_safe_examples(list(t), list(i), list(l), safe_examples, number_example, f"{select} safe pool")
        for t, i, l in groups
    ]


def build_contextual_groups(neighbors_per_label, safe_path, number_example):
    """Chunked version of ``build_contextual_examples``: pool this item's
    nearest-neighbour examples across ALL unsafe labels, sort by cosine similarity
    globally, then split into consecutive ``number_example``-sized groups (capped
    at ``_MAX_GROUPS``) -- each destined for its own model call. Each group gets
    its own slice of the safe pool appended for contrast."""
    all_safe_examples = load_examples(safe_path) if safe_path else load_examples(_DEFAULT_SAFE_FALLBACK)

    all_neighbors = []
    for lbl, neighbors in (neighbors_per_label or {}).items():
        if lbl in ("S0", "safe"):
            continue
        for n in neighbors:
            n_images = [i if isinstance(i, str) else i.get("file") for i in (n.get("images") or [])]
            if not _keep_example(n_images, n.get("video"), f"contextual neighbor [{lbl}]"):
                continue
            all_neighbors.append({
                "label": lbl,
                "text": n.get("text"),
                "images": _check_images(n_images, f"contextual neighbor [{lbl}]"),
                "cosine_similarity": n.get("cosine_similarity", 0),
            })
    all_neighbors.sort(key=lambda n: n["cosine_similarity"], reverse=True)

    texts = [n["text"] for n in all_neighbors]
    images = [n["images"] for n in all_neighbors]
    labels = [n["label"] for n in all_neighbors]

    groups = _chunk_examples(texts, images, labels, number_example)
    return [
        _append_safe_examples(list(t), list(i), list(l), all_safe_examples, number_example, "contextual safe pool")
        for t, i, l in groups
    ]


def majority_vote_with_first_run_fallback(labels):
    """Mode across a query item's per-group predictions. A tie (including the
    all-groups-disagree case) falls back to the first group's own prediction
    rather than picking arbitrarily among the tied labels.

    Returns ``(voted_label, majority_used, fallback_used)``."""
    from collections import Counter

    if not labels:
        return None, False, False
    valid_labels = [l for l in labels if l]
    if not valid_labels:
        return None, False, False
    counts = Counter(valid_labels)
    top_label, top_count = counts.most_common(1)[0]
    tied = [l for l, c in counts.items() if c == top_count]
    if len(tied) == 1:
        return top_label, True, False
    return labels[0], False, True
