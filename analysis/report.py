"""CLI entrypoint: load model results, print a Safe/Unsafe classification report,
and (optionally) save Gwet AC1 pairwise-agreement heatmaps.

    python -m analysis.report                                   # instruction-driven (default)
    python -m analysis.report -P example_driven                 # example-driven
    python -m analysis.report -P example_driven --subset-dirs my_dirs.json
    python -m analysis.report --plot-gwet

``--paradigm`` selects the default ``{subset: dir}`` layout and the report's
grouping columns:

  * ``instruction_driven`` -> dirs from analysis/subset_dirs.json
    (``outputs_instruction-driven/<subset>/``); grouped by subset x model x prompt_mode.
  * ``example_driven``      -> ``outputs_example-driven/<subset>/``; grouped by
    subset x in-context-select x example_per_group x use_safe_examples x model
    x prompt_mode.

``--subset-dirs`` is a JSON ``{subset: dir}`` merged over that default (only the
listed subsets are overridden).
"""
import argparse
import json
import os

from analysis.metrics import binary_classification_report, pairwise_gwet_matrix, plot_gwet_heatmap
from analysis.results import (
    DEFAULT_DIRS_BY_PARADIGM,
    GROUP_COLS_BY_PARADIGM,
    is_safety_model,
    load_results,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paradigm", "-P", default="instruction_driven",
                        choices=["instruction_driven", "example_driven"],
                        help="which pipeline's results to score (default: instruction_driven)")
    parser.add_argument("--subset-dirs", default=None,
                        help="path to a JSON {subset: output_dir} map, for subsets whose results "
                             "live somewhere other than the paradigm default. Only the listed "
                             "subsets are overridden; the rest keep their default directory.")
    parser.add_argument("--out-dir", default="analysis/reports",
                        help="where to save the classification report csv and heatmap pngs")
    parser.add_argument("--plot-gwet", action="store_true",
                        help="also save Gwet AC1 pairwise-agreement heatmaps")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    default_dirs = DEFAULT_DIRS_BY_PARADIGM[args.paradigm]
    subset_dirs = dict(default_dirs)
    if args.subset_dirs:
        with open(args.subset_dirs) as f:
            subset_dirs.update(json.load(f))

    df = load_results(subset_dirs, paradigm=args.paradigm)
    if df.empty:
        return

    group_cols = GROUP_COLS_BY_PARADIGM[args.paradigm]
    report = binary_classification_report(df, group_cols=group_cols)
    print(report.to_markdown(index=False))
    short = "example" if args.paradigm == "example_driven" else "instruction"
    report_path = os.path.join(args.out_dir, f"classification_report_{short}.csv")
    report.to_csv(report_path, index=False)
    print(f"Saved: {report_path}")

    if not args.plot_gwet:
        return

    # Safety-classifier models (Llama Guard, Shieldstral) are in the
    # classification report above, but excluded from pairwise agreement.
    vlm_df = df[~df["model"].apply(is_safety_model)]
    gwet_cols = (["subset", "incontext_select", "example_per_group"]
                 if args.paradigm == "example_driven" else ["subset", "prompt_mode"])
    for keys, subdf in vlm_df.groupby(gwet_cols):
        matrix, raters = pairwise_gwet_matrix(subdf)
        if len(raters) < 2:
            continue
        tag = "__".join(str(k) for k in (keys if isinstance(keys, tuple) else (keys,)))
        save_path = os.path.join(args.out_dir, f"gwet__{tag}.png")
        plot_gwet_heatmap(matrix, raters, title=tag, save_path=save_path)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
