from .base_dataset import BaseDataset
from utils.helpers import read_jsonl

_BLUESKY_LABELS = {
    "S0": "no-moderation",
    "S1": "porn",
    "S2": "sexual",
    "S3": "sexual-figurative",
    "S4": "self-harm",
    "S5": "nudity",
    "S6": "intolerant",
    "S7": "graphic-media",
    "S8": "rude",
    "S9": "threat",
    "S10": "other-unsafe",
}


class BlueskyDataset(BaseDataset):
    """Flat-JSONL Bluesky post loader for both paradigms.

    ``load_all()`` fills a column-oriented ``self.data`` dict with a superset of
    fields; columns absent from a given file come back as ``None`` / ``[]``.
    A ``*_metadata.jsonl`` fills ``platform_label`` / ``annotator_label`` /
    ``embed_type``; a consolidated test set fills ``label`` instead. (The
    ``-ics contextual`` neighbours are a separate ``-nb`` file, not a column
    here.)

    ``posts_with_video=False`` skips any row that has a video.
    """

    COLUMNS = (
        "text", "images", "video", "cid", "uri", "did", "langs", "is_reply",
        "embed_type", "platform_label", "annotator_label", "label",
    )

    def __init__(self, path, posts_with_video=False, split="test"):
        assert split in ("train", "test")
        self.path = path
        self.posts_with_video = posts_with_video
        self.split = split
        self.target_labels = list(_BLUESKY_LABELS.values())
        self.category_to_label = _BLUESKY_LABELS
        self.label_to_category = {v: k for k, v in _BLUESKY_LABELS.items()}

    def load_all(self):
        self.data = {k: [] for k in self.COLUMNS}
        for item in read_jsonl(self.path):
            video = item.get("video") or None
            if not self.posts_with_video and video:
                continue

            record = item.get("record", {}) or {}
            imgs = [
                f for f in ((i.get("file") or "").strip() for i in (item.get("images") or []))
                if f
            ] or None

            self.data["text"].append(item.get("text", "").strip().replace("\n", "\\n"))
            self.data["images"].append(imgs)
            self.data["video"].append(video)
            self.data["cid"].append(item.get("cid", "").strip())
            self.data["uri"].append(item.get("uri", "").strip())
            self.data["did"].append(item.get("did", "").strip() or None)
            self.data["langs"].append(record.get("langs", []))
            self.data["is_reply"].append(1 if "reply" in record else 0)
            self.data["embed_type"].append(record.get("etype") or None)
            self.data["platform_label"].append(item.get("platform_label") or None)
            self.data["annotator_label"].append(item.get("annotator_label") or None)
            self.data["label"].append(item.get("label") or None)
