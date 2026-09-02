from .bluesky_dataset import BlueskyDataset


def load_dataset(cfg, split="test"):
    ds_cfg = cfg["dataset"]
    path = ds_cfg["train_path"] if split == "train" else ds_cfg["test_path"]
    dataset = BlueskyDataset(
        path,
        posts_with_video=ds_cfg.get("posts_with_video", False),
        split=split,
    )
    dataset.load_all()
    print(f"Dataset {split} created, samples: {len(dataset.data['text'])}")
    return dataset
