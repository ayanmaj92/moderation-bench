"""Instruction-driven inference runner: the model gets only the
written policy, no worked examples."""

import os
import time

from tqdm import tqdm

from dataset_class import load_dataset
from prompt.instruction_driven import InstructionDrivenPrompt
from utils.helpers import parse_llm_response_2
from utils.main_helpers import ResultWriter, build_vllm_model_kwargs, split_reasoning


def run(cfg):
    from models import load_model_vllm

    print("Processing dataset")
    dataset_test = load_dataset(cfg)

    batch_size = cfg["dataset"].get("batch_size", 1)
    num_frames = cfg["model"].get("media_options", {}).get("num_frames", 16)

    prompt = InstructionDrivenPrompt(
        mode=cfg["prompt"]["mode"],
        platform=cfg["prompt"].get("platform", "bluesky"),
    )
    prompt_builder = prompt.build_chat_messages

    model_kwargs, inference_kwargs, video_as_frames = build_vllm_model_kwargs(
        cfg, max_images=num_frames, max_images_native=4,
    )
    model = load_model_vllm(**model_kwargs)
    model.video_as_frames = video_as_frames

    frame_text = (
        f"_frames-{num_frames}"
        if cfg["dataset"]["posts_with_video"] and cfg["model"].get("media_options", {}).get("num_frames")
        else ""
    )
    all_data = dataset_test.data
    len_data = len(all_data["text"])
    jsonl_name = (
        f"{prompt.platform}__{cfg['model']['name']}-{cfg['model']['size']}"
        f"__{cfg['prompt']['mode']}__conf-{cfg['prompt']['output_confidence']}{frame_text}.jsonl"
    )
    output_dir = cfg.get("output_dir") or "outputs_instruction-driven"
    jsonl_path = os.path.join(output_dir, jsonl_name)

    keys = ["text", "images", "video", "platform_label", "annotator_label",
            "cid", "uri", "is_reply", "embed_type"]

    with ResultWriter(jsonl_path) as writer:
        for batch_start in tqdm(range(0, len_data, batch_size),
                                total=(len_data + batch_size - 1) // batch_size,
                                desc="Processing"):
            batch_end = min(batch_start + batch_size, len_data)
            batch_slices = {k: all_data[k][batch_start:batch_end] for k in keys}

            start = time.time()
            try:
                outputs = model.predict_with_chat(
                    batch_texts=batch_slices["text"],
                    batch_image_paths=batch_slices["images"],
                    batch_video_paths=batch_slices["video"],
                    prompt_builder=prompt_builder,
                    **inference_kwargs,
                )
            except Exception as e:
                print(f"Error during model prediction at batch starting index {batch_start}:\n{e}")
                continue
            time_per_item = (time.time() - start) / (batch_end - batch_start)

            for idx_in_batch, output_item in enumerate(outputs):
                output_txt = output_item.outputs[0].text.strip()
                reasoning, answer = split_reasoning(output_txt, model.simple_name)
                parsed = parse_llm_response_2(
                    answer,
                    confidence=cfg["prompt"]["output_confidence"],
                    two_stage=False,
                    severity=False,
                )
                row_dict = {
                    "input_id": batch_start + idx_in_batch,
                    "input_cid": batch_slices["cid"][idx_in_batch],
                    "input_uri": batch_slices["uri"][idx_in_batch],
                    "input_is_reply": batch_slices["is_reply"][idx_in_batch],
                    "input_etype": batch_slices["embed_type"][idx_in_batch],
                    "text": batch_slices["text"][idx_in_batch].replace("\n", "\\n"),
                    "image_paths": batch_slices["images"][idx_in_batch],
                    "video_path": batch_slices["video"][idx_in_batch],
                    "platform_label": batch_slices["platform_label"][idx_in_batch],
                    "annotator_label": batch_slices["annotator_label"][idx_in_batch],
                    "model": f"{cfg['model']['name']}-{cfg['model']['size']}",
                    "prompt_type": cfg["prompt"]["type"],
                    "prompt_mode": cfg["prompt"]["mode"],
                    "num_frames": num_frames,
                    "incontext_select": None,
                    "output": {},
                    "processing_time": time_per_item,
                    "input_tokenized_length": len(output_item.prompt_token_ids),
                }
                if isinstance(parsed, dict):
                    row_dict["output"].update(parsed)
                if reasoning is not None:
                    row_dict["output"]["REASONING"] = reasoning
                row_dict["output"]["raw"] = output_txt.replace("\n", "\\n")
                writer.add(row_dict)

            writer.end_batch()

    print(f"Results saved at: {jsonl_path}")
