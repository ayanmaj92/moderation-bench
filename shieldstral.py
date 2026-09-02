import os

from utils.runtime import setup_cuda_env

setup_cuda_env(ld_library_path=True)  # locate a recent CUDA toolkit before importing torch/vLLM

import argparse
import json
import time

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Shieldstral-1.0-3B with the Bluesky custom policy."
    )
    parser.add_argument('--dataset', '-ds', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, default="outputs_instruction-driven/")
    parser.add_argument('--batch_size', '-bs', type=int, default=50)
    parser.add_argument('--max_model_len', '-mlen', type=int, default=32768)
    parser.add_argument('--gpu_memory_utilization', '-gmem', type=float, default=0.8)
    parser.add_argument('--tensor_parallel_size', '-tps', type=int, default=1)
    parser.add_argument('--posts_with_video', '-pv', action='store_true', default=False)
    parser.add_argument('--num_frames', '-nf', type=int, default=8)
    parser.add_argument('--prompt_mode', '-p', type=str, default='with_labels_rationale_details',
                        choices=['with_labels', 'with_labels_details', 'with_labels_rationale', 'with_labels_rationale_details'],
                        help='Policy file stem to use as the <Instruct> block.')
    return parser.parse_args()

args = parse_args()

hf_token     = os.getenv("HF_TOKEN", "")
hf_hub_cache = os.getenv("HF_HUB_CACHE", "")

os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]     = "expandable_segments:True"

##############################################################
# Heavy imports
##############################################################
# Note: Shieldstral requires vllm>=0.26.0 and mistral-common>=1.11.5
# (https://huggingface.co/mistralai/Shieldstral-1.0-3B)
from dataset_class import load_dataset
from utils.media import _load_single_media, _load_single_video_frames_base64
from vllm import LLM, SamplingParams
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch

##############################################################
# Constants
##############################################################
MODEL_ID     = "mistralai/Shieldstral-1.0-3B"
POLICY_DIR   = "prompt/policies/bluesky"

_BSKY_CATEGORIES = {
    "S1":  "porn",
    "S2":  "sexual",
    "S3":  "sexual-figurative",
    "S4":  "self-harm",
    "S5":  "nudity",
    "S6":  "intolerant",
    "S7":  "graphic-media",
    "S8":  "rude",
    "S9":  "threat",
    "S10": "other-unsafe",
}

SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruction provided. Note that the answer can only be "yes" or "no".'
)

TASK_TEXT = (
    "You are an expert content moderator for a social media platform. Your task is to determine whether "
    "a given social media post requires moderation and what labels need to be applied based on the "
    "labeling policy provided to you. Next, you will be given the particular social media post and its "
    "contents (text and visual) that need to be analyzed. Examine all provided content (text and visual) "
    "alongside the labeling policy before making a decision."
)

_YES = ("yes", "yes.", '"yes"', "'yes'")
_NO  = ("no",  "no.",  '"no"',  "'no'")

THRESHOLD = 0.5


##############################################################
# Helpers
##############################################################
def stable_softmax(x):
    x = np.array(x, dtype=float)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def _build_policy_instruct(policy_file):
    with open(policy_file) as f:
        policy_text = f.read().strip()
    return f"{TASK_TEXT}\n\n{policy_text}"


def build_category_query(key, name):
    return (
        f"Should this post be labeled '{key}: {name}' according to the "
        "Labeling Policy Details provided in the Instruction?"
    )


##############################################################
# Message builder
##############################################################
def build_shieldstral_messages(instruct, query, text, images):
    prefix = f"<Instruct>: {instruct}\n\n<Query>: {query}\n\n<Document>: "
    if images:
        content = [{"type": "text", "text": prefix}]
        for img in images:
            if img is not None:
                content.append({"type": "image_url", "image_url": {"url": img}})
        content.append({"type": "text", "text": f" {text or ''}"})
    else:
        content = prefix + (text or "")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": content},
    ]


##############################################################
# Score extraction
##############################################################
def extract_yes_score(output_item):
    if output_item is None:
        return None
    step_logprobs = output_item.outputs[0].logprobs
    if not step_logprobs:
        return None
    top = step_logprobs[0]
    z_yes, z_no = -1e9, -1e9
    for lp in top.values():
        tok = (lp.decoded_token or "").strip().lower()
        if tok in _YES:
            z_yes = max(z_yes, lp.logprob)
        elif tok in _NO:
            z_no = max(z_no, lp.logprob)
    return float(stable_softmax([z_yes, z_no])[0])


##############################################################
# Main
##############################################################
def main():
    unsafe_keys = sorted(_BSKY_CATEGORIES.keys(), key=lambda k: int(k[1:]))
    n_cats = len(unsafe_keys)

    policy_file      = os.path.join(POLICY_DIR, f"{args.prompt_mode}.md")
    policy_instruct  = _build_policy_instruct(policy_file)
    category_queries = {k: build_category_query(k, _BSKY_CATEGORIES[k]) for k in unsafe_keys}
    print(f"Bluesky policy loaded from {policy_file} ({n_cats} unsafe categories, instruct length: {len(policy_instruct)} chars).")

    cfg = {
        "dataset": {
            "name": "bluesky_final",
            "test_path": args.dataset,
            "posts_with_video": args.posts_with_video,
        },
        "model": {
            "media_options": {"num_frames": args.num_frames},
        },
    }
    print("Loading dataset...")
    all_data = load_dataset(cfg).data
    keys = list(all_data.keys())
    len_data = len(all_data["text"])
    print(f"Total samples: {len_data}")

    print(f"Loading {MODEL_ID}...")
    # When processing video posts, frames are extracted and passed as images.
    # Set the image limit to the larger of 4 (max images per post) or num_frames.
    max_images = args.num_frames if args.posts_with_video else 4
    load_kwargs = dict(
        model=MODEL_ID,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.batch_size * n_cats,
        limit_mm_per_prompt={"image": max_images, "video": 0, "audio": 0},
        disable_log_stats=True,
        tensor_parallel_size=args.tensor_parallel_size,
        max_logprobs=20,
    )
    if hf_hub_cache:
        load_kwargs["download_dir"] = hf_hub_cache
    if hf_token:
        load_kwargs["hf_token"] = hf_token
    llm = LLM(**load_kwargs)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        top_p=1.0,
        logprobs=20,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    frame_text = f"_frames-{args.num_frames}" if args.posts_with_video and args.num_frames else ""
    jsonl_name = (
        f"bluesky__shieldstral_1.0-3B"
        f"__{args.prompt_mode}__batch-{args.batch_size}{frame_text}.jsonl"
    )
    jsonl_path = os.path.join(args.output_dir, jsonl_name)

    buffer = []
    batch_counter = 0

    with open(jsonl_path, "w", encoding="utf-8") as outfile:
        for batch_start in tqdm(range(0, len_data, args.batch_size),
                                total=(len_data + args.batch_size - 1) // args.batch_size,
                                desc="Processing"):
            batch_end   = min(batch_start + args.batch_size, len_data)
            n_in_batch  = batch_end - batch_start
            batch = {k: all_data[k][batch_start:batch_end] for k in keys}

            tasks = []
            for idx, image_paths in enumerate(batch["images"]):
                if image_paths:
                    for j, p in enumerate(image_paths):
                        tasks.append({
                            "kind": "image", "batch_idx": idx, "pos": j, "path": p,
                            # Force decode+re-encode through PIL to ensure valid base64 payload
                            "kwargs": {"resize_images": True, "resize_factor": 1.0},
                        })

            results = [None] * len(tasks)
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {
                    executor.submit(_load_single_media, t["kind"], t["path"], **t["kwargs"]): i
                    for i, t in enumerate(tasks)
                }
                for future in as_completed(futures):
                    results[futures[future]] = future.result()

            batch_media = defaultdict(lambda: {"query_images": []})
            for task, res in zip(tasks, results):
                batch_media[task["batch_idx"]]["query_images"].append((task["pos"], res))
            for idx in batch_media:
                batch_media[idx]["query_images"] = [
                    x[1] for x in sorted(batch_media[idx]["query_images"], key=lambda x: x[0])
                ]

            # Video posts: extract frames manually and treat as images
            # (Shieldstral has no native video support)
            for idx, video_path in enumerate(batch["video"]):
                if video_path:
                    frames = _load_single_video_frames_base64(
                        video_path, num_frames=args.num_frames
                    )
                    if frames:
                        batch_media[idx]["query_images"] = frames

            all_messages = []
            for idx in range(n_in_batch):
                images = batch_media[idx]["query_images"] or None
                for k in unsafe_keys:
                    all_messages.append(build_shieldstral_messages(
                        instruct=policy_instruct,
                        query=category_queries[k],
                        text=batch["text"][idx],
                        images=images,
                    ))

            start = time.time()
            try:
                outputs = llm.chat(all_messages, sampling_params=sampling_params, use_tqdm=False)
            except Exception as e:
                tqdm.write(
                    f"Batch {batch_start} failed ({e!r}); retrying {len(all_messages)} requests individually..."
                )
                outputs = [None] * len(all_messages)
                for i, msg in enumerate(all_messages):
                    idx_in_batch, cat_pos = divmod(i, n_cats)
                    try:
                        outputs[i] = llm.chat([msg], sampling_params=sampling_params, use_tqdm=False)[0]
                    except Exception as e_single:
                        tqdm.write(
                            f"  FAILED input_id={batch_start + idx_in_batch} "
                            f"category={unsafe_keys[cat_pos]} "
                            f"cid={batch['cid'][idx_in_batch]!r} "
                            f"n_images={len(batch['images'][idx_in_batch] or [])} "
                            f"err={e_single!r}"
                        )
            elapsed = time.time() - start

            for idx_in_batch in range(n_in_batch):
                post_outputs = outputs[idx_in_batch * n_cats: (idx_in_batch + 1) * n_cats]
                scores_list  = [extract_yes_score(o) for o in post_outputs]

                flagged = [s is not None and s > THRESHOLD for s in scores_list]
                if any(flagged):
                    best_idx = max(
                        (i for i, s in enumerate(scores_list) if s is not None),
                        key=lambda i: scores_list[i],
                    )
                    predicted_category = unsafe_keys[best_idx]
                else:
                    predicted_category = "S0"

                input_lens = [
                    o.prompt_token_ids for o in post_outputs
                    if o is not None and o.prompt_token_ids is not None
                ]
                input_len = max((len(l) for l in input_lens), default=None)

                row_dict = {
                    "input_id":       batch_start + idx_in_batch,
                    "input_cid":      batch["cid"][idx_in_batch],
                    "input_uri":      batch["uri"][idx_in_batch],
                    "input_is_reply": batch["is_reply"][idx_in_batch],
                    "input_etype":    batch["embed_type"][idx_in_batch],
                    "text":           batch["text"][idx_in_batch].replace("\n", "\\n"),
                    "image_paths":    batch["images"][idx_in_batch],
                    "video_path":     batch["video"][idx_in_batch],
                    "platform_label": batch["platform_label"][idx_in_batch],
                    "annotator_label": batch["annotator_label"][idx_in_batch],
                    "model":          "shieldstral_1.0-3B",
                    "prompt_type":    "shieldstral_bsky_policy",
                    "prompt_mode":    args.prompt_mode,
                    "output": {
                        "predicted_category":      predicted_category,
                        "predicted_category_name": _BSKY_CATEGORIES.get(predicted_category, "no-moderation"),
                        "scores":                  scores_list,
                    },
                    "processing_time":        elapsed / n_in_batch,
                    "input_tokenized_length": input_len,
                }
                buffer.append(json.dumps(row_dict))

            batch_counter += 1
            if batch_counter % 10 == 0 and buffer:
                outfile.write("\n".join(buffer) + "\n")
                outfile.flush()
                buffer.clear()
                torch.cuda.empty_cache()

        if buffer:
            outfile.write("\n".join(buffer) + "\n")
            outfile.flush()

    print(f"Results saved at: {jsonl_path}")


if __name__ == "__main__":
    main()
