"""
Load model result jsonl files into one flat DataFrame, joined against the
human-annotated ground truth.

Handles both paradigms -- pass ``paradigm="instruction_driven"`` (the default) or
``"example_driven"``. That only selects the default ``{subset: dir}`` layout and
(in ``analysis.report``) the report's grouping columns; the rows themselves carry
the same fields either way, with the example-driven-only columns
(``incontext_select`` / ``example_per_group`` / ``use_safe_examples``) left None
for instruction-driven rows.
"""
import json
from pathlib import Path

import pandas as pd

from utils.helpers import read_jsonl

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data_files"

SUBSETS = ["moderated", "near-moderated", "random", "safe"]

# Where each subset's model-result *.jsonl files live by default. For
# instruction-driven this comes from analysis/subset_dirs.json (edit the file,
# not the constant); for example-driven it's one dir per subset under
# outputs_example-driven/. Override fully or partially via load_results(subset_dirs=...)
# or `report.py --subset-dirs`.
DEFAULT_SUBSET_DIRS_FILE = Path(__file__).resolve().parent / "subset_dirs.json"
with open(DEFAULT_SUBSET_DIRS_FILE) as _f:
    DEFAULT_SUBSET_DIRS: dict = json.load(_f)

DEFAULT_DIRS_BY_PARADIGM = {
    "instruction_driven": DEFAULT_SUBSET_DIRS,
    "example_driven": {s: f"outputs_example-driven/{s}" for s in SUBSETS},
}

# What binary_classification_report groups by, per paradigm.
GROUP_COLS_BY_PARADIGM = {
    "instruction_driven": ("subset", "model", "prompt_mode"),
    "example_driven": ("subset", "incontext_select", "example_per_group",
                       "use_safe_examples", "model", "prompt_mode"),
}


# "model" prefixes for the safety-classifier models (Llama Guard, Shieldstral)
# as opposed to the standard VLMs --> used to exclude them from Gwet
# pairwise-agreement analysis.
SAFETY_MODEL_PREFIXES = ("llama_guard", "shieldstral")


def is_safety_model(model: str) -> bool:
    return str(model or "").startswith(SAFETY_MODEL_PREFIXES)


def _ground_truth_path(subset: str) -> Path:
    return DATA_DIR / f"{subset}_uri.jsonl"


def _parse_row(row: dict) -> tuple:
    model = row.get("model") or ""
    output = row.get("output") or {}

    if model.startswith("llama_guard"):
        verdict = str(output.get("safety_verdict") or "").strip().lower()
        pred_label = {"safe": "Safe", "unsafe": "Unsafe"}.get(verdict)
        return output.get("predicted_category"), pred_label

    pred_cat = output.get("PREDICTED_CATEGORY_ID") or output.get("predicted_category")
    if pred_cat == "S0":
        pred_label = "Safe"
    elif pred_cat:
        pred_label = "Unsafe"
    else:
        pred_label = None
    return pred_cat, pred_label


def load_results(subset_dirs: dict | None = None,
                 paradigm: str = "instruction_driven") -> pd.DataFrame:
    if subset_dirs is None:
        subset_dirs = DEFAULT_DIRS_BY_PARADIGM[paradigm]

    rows = []
    for subset, dir_paths in subset_dirs.items():
        if subset not in SUBSETS:
            print(f"[WARN] Unknown subset '{subset}' (expected one of {SUBSETS}) -- skipping")
            continue

        gt_path = _ground_truth_path(subset)
        if not gt_path.exists():
            print(f"[WARN] Ground truth file not found for subset '{subset}': {gt_path} -- skipping")
            continue
        gt = {r["uri"]: r["annotator_label"] for r in read_jsonl(gt_path)}

        if isinstance(dir_paths, (str, Path)):
            dir_paths = [dir_paths]

        matched = []
        for dir_path in dir_paths:
            dir_path = Path(dir_path)
            if not dir_path.is_absolute():
                dir_path = REPO_ROOT / dir_path
            if not dir_path.is_dir():
                print(f"[WARN] No results yet for subset '{subset}' -- directory doesn't exist: {dir_path}")
                continue
            found = sorted(dir_path.glob("*.jsonl"))
            if not found:
                print(f"[WARN] No results yet for subset '{subset}' -- no .jsonl files in: {dir_path}")
                continue
            matched.extend(found)

        for file_path in matched:
            for row in read_jsonl(file_path):
                uri = row.get("input_uri")
                output = row.get("output") or {}
                pred_cat, pred_label = _parse_row(row)
                rows.append({
                    "subset": subset,
                    "model": row.get("model"),
                    "prompt_mode": row.get("prompt_mode"),
                    "prompt_type": row.get("prompt_type"),
                    # example-driven-only; None for instruction-driven rows
                    "incontext_select": row.get("incontext_select"),
                    "example_per_group": row.get("example_per_group"),
                    "use_safe_examples": row.get("use_safe_examples"),
                    "uri": uri,
                    "annotator_label": gt.get(uri),
                    "pred_category": pred_cat,
                    "pred_label": pred_label,
                    "confidence": output.get("CONFIDENCE_SCORE"),
                    "processing_time": row.get("processing_time"),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        print("[WARN] No rows loaded for any subset")
        return df

    n_missing_gt = df["annotator_label"].isna().sum()
    if n_missing_gt:
        print(f"[WARN] {n_missing_gt}/{len(df)} rows have no ground-truth annotator_label "
              f"(uri not found in the matching data_files/<subset>_uri.jsonl)")
    n_unparsed = df["pred_label"].isna().sum()
    if n_unparsed:
        print(f"[WARN] {n_unparsed}/{len(df)} rows have no parseable PREDICTED_CATEGORY_ID")

    # Dedup key includes the example-driven axes so multiple strategies in one dir
    # don't look like duplicates; for instruction-driven those cols are all None
    # so this reduces to (subset, model, prompt_mode).
    dup_key = ["subset", "model", "prompt_mode", "incontext_select",
               "example_per_group", "use_safe_examples"]
    dup_counts = df.groupby(dup_key, dropna=False)["uri"].apply(lambda s: s.duplicated().sum())
    dup_groups = dup_counts[dup_counts > 0]
    if not dup_groups.empty:
        print(f"[WARN] {dup_groups.sum()} duplicate (config, uri) rows found -- likely more than "
              f"one result file for the same config in a subset dir (e.g. a rerun left both the "
              f"old and new file), which double-counts those uris in every metric below:\n"
              f"{dup_groups.to_string()}")

    return df
