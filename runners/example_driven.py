"""Example-driven inference runner: in-context demonstrations on top
of the instruction-driven pipeline, with ``random`` / ``prototypical`` / ``contextual``
example selection and an optional ``example_per_group`` chunked multi-call +
majority-vote mode."""

import os
import time

from tqdm import tqdm

from dataset_class import load_dataset
from prompt.example_driven import ExampleDrivenPrompt
from utils.helpers import parse_llm_response_2, read_jsonl
from utils.example_driven import (
    build_contextual_examples,
    build_contextual_groups,
    build_random_or_prototypical_examples,
    build_static_groups,
    log_media_stats_summary,
    majority_vote_with_first_run_fallback,
    reset_media_stats,
)
from utils.main_helpers import ResultWriter, build_vllm_model_kwargs, split_reasoning


def _apply_image_family_overrides(inference_kwargs, model_name):
    """Per-model-family image handling for example-driven's much larger prompts:
    gemma -> no resizing; qwen -> a max_pixels cap (independent of
    resize_images/resize_factor) so the combined text+image prompt fits
    max_model_len; everyone else (mistral3, magistral_small, llama4_scout,
    intern, llava) -> resize_images with resize_factor 0.5."""
    name = model_name.lower()
    if "gemma" in name:
        inference_kwargs["resize_images"] = False
        inference_kwargs["max_pixels"] = None
    elif "qwen" in name:
        inference_kwargs["resize_images"] = False
        inference_kwargs["max_pixels"] = 384000
    else:
        inference_kwargs["resize_images"] = True
        inference_kwargs["resize_factor"] = 0.5
        inference_kwargs["max_pixels"] = None


def run(cfg):
    from models import load_model_vllm

    print("Processing dataset")
    dataset_test = load_dataset(cfg)

    batch_size = cfg["dataset"].get("batch_size", 1)
    number_example = cfg["prompt"].get("in_context_num", 10)
    select = cfg["prompt"]["in_context_select"]
    num_frames = cfg["model"].get("media_options", {}).get("num_frames", 16)
    output_confidence = cfg["prompt"]["output_confidence"]

    prompt = ExampleDrivenPrompt(
        mode=cfg["prompt"]["mode"],
        platform=cfg["prompt"].get("platform", "bluesky"),
        use_safe_examples=cfg["prompt"].get("use_safe_examples", True),
    )
    prompt_builder = prompt.build_chat_messages

    model_kwargs, inference_kwargs, video_as_frames = build_vllm_model_kwargs(
        cfg, max_images=250, max_images_native=250,
    )
    model = load_model_vllm(**model_kwargs)
    model.video_as_frames = video_as_frames
    _apply_image_family_overrides(inference_kwargs, model_kwargs["model_name"])

    example_per_group = cfg["prompt"].get("example_per_group", False)
    group_batch_size = cfg["dataset"].get("group_batch_size", 5)
    safe_path = cfg["dataset"].get("safe_path", "")

    # -ics contextual: the per-query nearest neighbours come from a side file
    # (`-nb`), joined to the query set by uri -- so `--dataset` is always a plain
    # *_metadata.jsonl, same as every other run.
    neighbors_index = None
    if select == "contextual":
        neighbors_path = cfg["dataset"].get("neighbors_path", "")
        if not neighbors_path:
            raise ValueError(
                "in_context_select='contextual' requires -nb/--neighbors_path -- an (enriched) "
                "data_files/example_driven/contextual_test_sets/<subset>_top10_per_label_flattened.jsonl "
                "of {uri, neighbors_per_label}, joined to --dataset by uri."
            )
        neighbors_index = {
            r["uri"]: (r.get("neighbors_per_label") or {})
            for r in read_jsonl(neighbors_path) if r.get("uri")
        }
        print(f"Loaded neighbours for {len(neighbors_index)} query uris from {neighbors_path}")

    reset_media_stats()

    # Precompute the static (random/prototypical) pool once -- identical for
    # every query item.
    static_flat = None
    static_groups = None
    if select in ("random", "prototypical"):
        if example_per_group:
            static_groups = build_static_groups(select, number_example)
        else:
            static_flat = build_random_or_prototypical_examples(select, number_example)

    all_data = dataset_test.data
    len_data = len(all_data["text"])

    def neighbors_for(idx_in_batch):
        return neighbors_index.get(batch_slices["uri"][idx_in_batch], {})

    dataset_tag = os.path.splitext(os.path.basename(cfg["dataset"]["test_path"]))[0]
    use_safe_examples = cfg["prompt"].get("use_safe_examples", True)
    per_group_tag = "_per_group" if example_per_group else ""
    jsonl_name = (
        f"example_driven{per_group_tag}_{dataset_tag}_{select}"
        f"_usesafe_{cfg['prompt'].get('use_safe_examples', True)}"
        f"_{cfg['model']['name']}-{cfg['model']['size']}"
        f"_{cfg['prompt']['mode']}_conf-{output_confidence}.jsonl"
    )
    output_dir = cfg.get("output_dir") or "outputs_example-driven"
    jsonl_path = os.path.join(output_dir, jsonl_name)

    # Same columns the instruction-driven runner reads, plus the two the
    # example-driven path needs on top (`label` as an annotator_label fallback
    # for consolidated test sets).
    keys = ["text", "images", "video", "platform_label", "annotator_label",
            "cid", "uri", "is_reply", "embed_type", "label"]

    def base_row(idx_in_batch):
        # Common fields -- identical to runners/instruction_driven.py's row_dict.
        row = {
            "input_id": batch_start + idx_in_batch,
            "input_cid": batch_slices["cid"][idx_in_batch],
            "input_uri": batch_slices["uri"][idx_in_batch],
            "input_is_reply": batch_slices["is_reply"][idx_in_batch],
            "input_etype": batch_slices["embed_type"][idx_in_batch],
            "text": batch_slices["text"][idx_in_batch].replace("\n", "\\n"),
            "image_paths": batch_slices["images"][idx_in_batch],
            "video_path": batch_slices["video"][idx_in_batch],
            "platform_label": batch_slices["platform_label"][idx_in_batch],
            # consolidated test sets carry the human label under `label`, not
            # `annotator_label` -- prefer the latter, fall back to the former.
            "annotator_label": (batch_slices["annotator_label"][idx_in_batch]
                                or batch_slices["label"][idx_in_batch]),
            "model": f"{cfg['model']['name']}-{cfg['model']['size']}",
            "prompt_type": cfg["prompt"]["type"],
            "prompt_mode": cfg["prompt"]["mode"],
            "num_frames": num_frames,
            "incontext_select": select,
        }
        # Example-driven-only fields.
        row.update({
            "example_per_group": example_per_group,
            "use_safe_examples": use_safe_examples,
        })
        return row

    with ResultWriter(jsonl_path) as writer:
        for batch_start in tqdm(range(0, len_data, batch_size),
                                total=(len_data + batch_size - 1) // batch_size,
                                desc="Processing"):
            batch_end = min(batch_start + batch_size, len_data)
            batch_slices = {k: all_data[k][batch_start:batch_end] for k in keys}
            batch_n = batch_end - batch_start

            # ---------------------------------------------------------------
            # example_per_group: one model call per example chunk, majority vote
            # ---------------------------------------------------------------
            if example_per_group:
                for idx_in_batch in range(batch_n):
                    if select in ("random", "prototypical"):
                        item_groups = static_groups
                    else:
                        item_groups = build_contextual_groups(
                            neighbors_for(idx_in_batch), safe_path, number_example
                        )

                    item_start = time.time()
                    group_preds, group_confs, group_outputs_log = [], [], []
                    prompt_len = None
                    for g_start in range(0, len(item_groups), group_batch_size):
                        g_end = min(g_start + group_batch_size, len(item_groups))
                        g_n = g_end - g_start
                        try:
                            g_outputs = model.predict_with_chat(
                                batch_texts=[batch_slices["text"][idx_in_batch]] * g_n,
                                batch_image_paths=[batch_slices["images"][idx_in_batch]] * g_n,
                                batch_video_paths=[batch_slices["video"][idx_in_batch]] * g_n,
                                prompt_builder=prompt_builder,
                                example_k_texts=[item_groups[g][0] for g in range(g_start, g_end)],
                                example_k_images=[item_groups[g][1] for g in range(g_start, g_end)],
                                example_k_labels=[item_groups[g][2] for g in range(g_start, g_end)],
                                **inference_kwargs,
                            )
                        except Exception as e:
                            print(f"Error during per-group model prediction (item "
                                  f"{batch_start + idx_in_batch}, groups {g_start}:{g_end}):\n{e}")
                            continue
                        for out_item in g_outputs:
                            if prompt_len is None:
                                prompt_len = len(out_item.prompt_token_ids)
                            out_txt = out_item.outputs[0].text.strip()
                            reasoning, answer = split_reasoning(out_txt, model.simple_name)
                            parsed = parse_llm_response_2(
                                answer, confidence=output_confidence, two_stage=False, severity=False,
                            )
                            pred = parsed.get("PREDICTED_CATEGORY_ID") if isinstance(parsed, dict) else None
                            conf = parsed.get("CONFIDENCE_SCORE") if isinstance(parsed, dict) else None
                            group_preds.append(pred)
                            group_confs.append(conf)
                            group_outputs_log.append({
                                "PREDICTED_CATEGORY_ID": pred,
                                "CONFIDENCE_SCORE": conf,
                                "REASONING": reasoning,
                                "raw": out_txt.replace("\n", "\\n"),
                            })

                    voted_label, majority_used, fallback_used = majority_vote_with_first_run_fallback(group_preds)
                    voted_conf = next(
                        (c for p, c in zip(group_preds, group_confs) if p == voted_label and c is not None),
                        None,
                    )
                    row_dict = base_row(idx_in_batch)
                    row_dict.update({
                        "incontext_num": sum(len(g[2]) for g in item_groups),
                        "num_groups": len(item_groups),
                        "output": {
                            "PREDICTED_CATEGORY_ID": voted_label,
                            "CONFIDENCE_SCORE": voted_conf,
                            "PER_GROUP_PREDICTED_CATEGORY_IDS": group_preds,
                            "MAJORITY_VOTE_USED": majority_used,
                            "FALLBACK_TO_FIRST_RUN": fallback_used,
                            "GROUP_OUTPUTS": group_outputs_log,
                        },
                        "processing_time": time.time() - item_start,
                        "input_tokenized_length": prompt_len,
                    })
                    writer.add(row_dict)

                writer.end_batch()
                continue

            # ---------------------------------------------------------------
            # flat: one combined prompt per query item
            # ---------------------------------------------------------------
            example_k_texts, example_k_images, example_k_labels = [], [], []
            if select in ("random", "prototypical"):
                example_k_texts = [static_flat[0]] * batch_n
                example_k_images = [static_flat[1]] * batch_n
                example_k_labels = [static_flat[2]] * batch_n
            elif select == "contextual":
                for idx_in_batch in range(batch_n):
                    t, i, l = build_contextual_examples(neighbors_for(idx_in_batch), safe_path, number_example)
                    example_k_texts.append(t)
                    example_k_images.append(i)
                    example_k_labels.append(l)
            else:
                raise ValueError(f"Unsupported in_context_select: {select}")

            start = time.time()
            try:
                outputs = model.predict_with_chat(
                    batch_texts=batch_slices["text"],
                    batch_image_paths=batch_slices["images"],
                    batch_video_paths=batch_slices["video"],
                    prompt_builder=prompt_builder,
                    example_k_texts=example_k_texts,
                    example_k_images=example_k_images,
                    example_k_labels=example_k_labels,
                    **inference_kwargs,
                )
            except Exception as e:
                print(f"Error during model prediction at batch starting index {batch_start}:\n{e}")
                continue
            time_per_item = (time.time() - start) / batch_n

            for idx_in_batch, output_item in enumerate(outputs):
                output_txt = output_item.outputs[0].text.strip()
                reasoning, answer = split_reasoning(output_txt, model.simple_name)
                parsed = parse_llm_response_2(
                    answer, confidence=output_confidence, two_stage=False, severity=False,
                )
                row_dict = base_row(idx_in_batch)
                row_dict.update({
                    "incontext_num": len(example_k_labels[idx_in_batch]),
                    "output": {},
                    "processing_time": time_per_item,
                    "input_tokenized_length": len(output_item.prompt_token_ids),
                })
                if isinstance(parsed, dict):
                    row_dict["output"].update(parsed)
                if reasoning is not None:
                    row_dict["output"]["REASONING"] = reasoning
                row_dict["output"]["raw"] = output_txt.replace("\n", "\\n")
                writer.add(row_dict)

            writer.end_batch()

    log_media_stats_summary()
    print(f"Results saved at: {jsonl_path}")
