"""Metrics over a results DataFrame produced by analysis.results.load_results().

- binary_classification_report(): precision/recall/F1/accuracy/flagging_rate.
- pairwise_gwet_matrix() / plot_gwet_heatmap(): pairwise Gwet AC1 agreement between models.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def binary_classification_report(
    df: pd.DataFrame,
    group_cols: tuple[str, ...] = ("subset", "model", "prompt_mode"),
    pos_label: str = "Unsafe",
) -> pd.DataFrame:
    """Precision/recall/F1 (binary, pos_label="Unsafe" by default) + accuracy,
    one row per group_cols combination. Rows with no ground truth or no
    parseable prediction are dropped before scoring."""
    scored = df.dropna(subset=["annotator_label", "pred_label"])
    if scored.empty:
        print("[WARN] No rows have both ground truth and a parseable prediction -- "
              "returning an empty classification report")
        return pd.DataFrame(columns=list(group_cols) + ["precision", "recall", "f1", "accuracy", "flagging_rate", "n"])

    rows = []
    for keys, g in scored.groupby(list(group_cols)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        precision, recall, f1, _ = precision_recall_fscore_support(
            g["annotator_label"], g["pred_label"],
            labels=["Safe", "Unsafe"], pos_label=pos_label,
            average="binary", zero_division=0,
        )
        row = dict(zip(group_cols, keys))
        row.update({
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy_score(g["annotator_label"], g["pred_label"]), 4),
            "flagging_rate": round((g["pred_label"] == "Unsafe").mean() * 100, 2),
            "n": len(g),
        })
        rows.append(row)

    return pd.DataFrame(rows).sort_values(list(group_cols)).reset_index(drop=True)


def pairwise_gwet_matrix(
    df: pd.DataFrame,
    value_col: str = "pred_label",
    categories: tuple[str, ...] = ("Safe", "Unsafe"),
    include_human: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Pairwise Gwet AC1 agreement between every pair of raters (models) in
    `df`, computed only over the uris both raters rated in common.

    `df` should already be filtered to one (subset, prompt_mode) slice --
    pass e.g. df[(df.subset == "pilot_moderated") & (df.prompt_mode == "with_labels")].
    If include_human, a "Human" rater column is added from `annotator_label`
    (one value per uri, taken from the first row for that uri).

    Returns (matrix, rater_names): matrix is (n_raters x n_raters), diagonal
    1.0, lower triangle = AC1, upper triangle = NaN.
    """
    from irrCAC.raw import CAC

    pivot = df.pivot_table(index="uri", columns="model", values=value_col, aggfunc="first")
    if include_human:
        human = df.drop_duplicates("uri").set_index("uri")["annotator_label"]
        pivot["Human"] = human.reindex(pivot.index)

    raters = list(pivot.columns)
    n = len(raters)
    matrix = np.full((n, n), np.nan)
    np.fill_diagonal(matrix, 1.0)

    for i in range(n):
        for j in range(i):  # lower triangle only
            pair = pivot.iloc[:, [j, i]].dropna()
            if len(pair) < 2:
                continue
            try:
                matrix[i, j] = CAC(pair, categories=list(categories)).gwet()["est"]["coefficient_value"]
            except Exception as exc:
                print(f"[WARN] Gwet AC1 failed for ({raters[i]}, {raters[j]}): {exc}")

    return matrix, raters


def plot_gwet_heatmap(
    matrix: np.ndarray,
    rater_names: list[str],
    title: str | None = None,
    cmap: str = "RdYlGn",
    vmin: float = 0.0,
    vmax: float = 1.0,
    ax=None,
    save_path: str | None = None,
):
    """Draw one lower-triangle heatmap for a pairwise_gwet_matrix() result."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    n = len(rater_names)
    mat_df = pd.DataFrame(matrix, index=rater_names, columns=rater_names)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(1.2 * n + 1, 1.2 * n + 1))

    sns.heatmap(
        mat_df, ax=ax, mask=mask, cmap=cmap, vmin=vmin, vmax=vmax,
        annot=True, fmt=".2f", linewidths=0.5, linecolor="white",
        square=True, cbar=True,
    )
    if title:
        ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    ax.tick_params(axis="y", rotation=0)

    if created_fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
    return ax.figure
