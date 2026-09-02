import os

from utils.runtime import setup_cuda_env

setup_cuda_env()  # locate a recent CUDA toolkit before importing torch/vLLM

import argparse
import json
import time

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Llama Guard 4 on Bluesky posts with either its own built-in policy or the Bluesky custom policy."
    )
    parser.add_argument('--dataset', '-ds', type=str, required=True)
    parser.add_argument('--policy', '-p', choices=['own', 'bluesky'], default='own',
                        help="'own' uses Llama Guard's built-in taxonomy; "
                             "'bluesky' injects the custom Bluesky policy via chat_template_kwargs.")
    parser.add_argument('--output_dir', '-o', type=str, default="outputs_instruction-driven/")
    parser.add_argument('--batch_size', '-bs', type=int, default=10)
    parser.add_argument('--max_model_len', '-mlen', type=int, default=32768)
    parser.add_argument('--gpu_memory_utilization', '-gmem', type=float, default=0.95)
    parser.add_argument('--tensor_parallel_size', '-tps', type=int, default=1)
    parser.add_argument('--posts_with_video', '-pv', action='store_true', default=False)
    parser.add_argument('--num_frames', '-nf', type=int, default=16)
    return parser.parse_args()

args = parse_args()

from dataset_class import load_dataset
from utils.media import _load_single_media
from vllm import LLM, SamplingParams
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm
import torch

MODEL_ID    = "meta-llama/Llama-Guard-4-12B"
POLICY_FILE = "prompt/policies/bluesky/with_labels_details.md"

# Llama Guard 4 built-in hazard taxonomy
_META_CATEGORIES = {
    "S1":  "Violent Crimes",
    "S2":  "Non-Violent Crimes",
    "S3":  "Sex-Related Crimes",
    "S4":  "Child Sexual Exploitation",
    "S5":  "Defamation",
    "S6":  "Specialized Advice",
    "S7":  "Privacy",
    "S8":  "Intellectual Property",
    "S9":  "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Suicide & Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections",
    "S14": "Code Interpreter Abuse",
}

# Bluesky moderation taxonomy
_BSKY_CATEGORIES = {
    "S0":  "no-moderation",
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


def _load_bluesky_categories(path):
    """Parse policy markdown into {S1: full_block, ...} for chat_template_kwargs."""
    with open(path) as f:
        text = f.read()
    categories = {}
    sections = re.split(r'\n(?=## \*\*S\d+\*\*)', text)
    for section in sections:
        m = re.match(r'## \*\*(S\d+)\*\*: (.+?)\n(.*)', section, re.DOTALL)
        if not m:
            continue
        cat_id = m.group(1)
        name   = m.group(2).strip()
        body   = m.group(3).rstrip('-').strip()
        categories[cat_id] = f"{name}\n{body}"
    return categories


def build_guard_messages(text, images, video, video_as_frames=True):
    content = []
    if images:
        for img in images:
            if img is not None:
                content.append({"type": "image_url", "image_url": {"url": img}})
    elif video is not None:
        if video_as_frames and isinstance(video, list):
            for frame in video:
                if frame is not None:
                    content.append({"type": "image_url", "image_url": {"url": frame}})
        else:
            content.append({"type": "video_url", "video_url": {"url": video}})
    content.append({"type": "text", "text": text if text else ""})
    return [{"role": "user", "content": content}]


def parse_guard_output(text: str):
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if not lines:
        return "unknown", None
    verdict = lines[0].lower()
    # Take first item (template may return a comma-separated list)
    category = lines[1].split(',')[0].strip() if len(lines) > 1 else None
    return verdict, category


def main():
    # Load policy if bluesky
    if args.policy == 'bluesky':
        bluesky_categories_dict = _load_bluesky_categories(POLICY_FILE)
        print(f"Bluesky policy loaded ({len(bluesky_categories_dict)} categories).")
        category_lookup = _BSKY_CATEGORIES
        prompt_type = "llama_guard_bsky_policy"
        prompt_mode = "with_details"
        filename_tag = "bsky_policy"
    else:
        bluesky_categories_dict = None
        category_lookup = _META_CATEGORIES
        prompt_type = "llama_guard_own"
        prompt_mode = "own"
        filename_tag = "own"

    print("Loading dataset...")
    cfg = {
        "dataset": {
            "name": "bluesky_final",
            "test_path": args.dataset,
            "posts_with_video": args.posts_with_video,
        }
    }
    dataset = load_dataset(cfg)
    all_data = dataset.data
    len_data = len(all_data['text'])
    print(f"Total samples: {len_data}")

    # Frames are passed as images; allow up to num_frames images when processing video.
    max_images = args.num_frames if args.posts_with_video else 4
    print(f"Loading {MODEL_ID}...")
    llm = LLM(
        model=MODEL_ID,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.batch_size,
        limit_mm_per_prompt={"image": max_images, "video": 0, "audio": 0},
        download_dir=os.getenv("HF_HUB_CACHE", ""),
        hf_token=os.getenv("HF_TOKEN", ""),
        disable_log_stats=True,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=20, top_p=1.0)

    os.makedirs(args.output_dir, exist_ok=True)
    frame_text = f"_frames-{args.num_frames}" if args.posts_with_video and args.num_frames else ""
    jsonl_name = (
        f"bluesky__llama_guard_4-12B"
        f"__{prompt_mode}__{filename_tag}_batch-{args.batch_size}{frame_text}.jsonl"
    )
    jsonl_path = os.path.join(args.output_dir, jsonl_name)

    keys = ['text', 'images', 'video', 'platform_label', 'annotator_label', 'cid', 'uri', 'is_reply', 'embed_type']
    buffer = []
    batch_counter = 0

    with open(jsonl_path, "w", encoding="utf-8") as outfile:
        for batch_start in tqdm(range(0, len_data, args.batch_size),
                                total=(len_data + args.batch_size - 1) // args.batch_size,
                                desc="Processing"):
            batch_end = min(batch_start + args.batch_size, len_data)
            batch = {k: all_data[k][batch_start:batch_end] for k in keys}

            # Load media in parallel
            tasks = []
            for idx, (image_paths, video_path) in enumerate(zip(batch['images'], batch['video'])):
                if image_paths:
                    for j, p in enumerate(image_paths):
                        tasks.append({
                            "kind": "image", "batch_idx": idx,
                            "slot": "query_images", "pos": j,
                            "path": p, "kwargs": {"resize_images": False},
                        })
                if video_path and args.posts_with_video:
                    tasks.append({
                        "kind": "video_frames", "batch_idx": idx,
                        "slot": "query_video", "pos": 0,
                        "path": video_path, "kwargs": {"num_frames": args.num_frames},
                    })

            results = [None] * len(tasks)
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {
                    executor.submit(_load_single_media, task["kind"], task["path"], **task["kwargs"]): i
                    for i, task in enumerate(tasks)
                }
                for future in as_completed(futures):
                    results[futures[future]] = future.result()

            batch_media = defaultdict(lambda: {"query_images": [], "query_video": None})
            for task, res in zip(tasks, results):
                idx = task["batch_idx"]
                if task["slot"] == "query_video":
                    batch_media[idx]["query_video"] = res
                else:
                    batch_media[idx]["query_images"].append((task["pos"], res))
            for idx in batch_media:
                batch_media[idx]["query_images"] = [
                    x[1] for x in sorted(batch_media[idx]["query_images"], key=lambda x: x[0])
                ]

            # Build messages
            all_messages = []
            for idx in range(batch_end - batch_start):
                all_messages.append(build_guard_messages(
                    text=batch['text'][idx],
                    images=batch_media[idx]["query_images"] or None,
                    video=batch_media[idx]["query_video"],
                ))

            start = time.time()
            try:
                chat_kwargs = {}
                if bluesky_categories_dict is not None:
                    chat_kwargs["chat_template_kwargs"] = {"categories": bluesky_categories_dict}
                outputs = llm.chat(
                    all_messages,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                    **chat_kwargs,
                )
            except Exception as e:
                print(f"Error during inference at batch {batch_start}: {e}")
                continue
            elapsed = time.time() - start

            for idx_in_batch, output_item in enumerate(outputs):
                raw_output = output_item.outputs[0].text.strip()
                verdict, category = parse_guard_output(raw_output)
                row_dict = {
                    "input_id":       batch_start + idx_in_batch,
                    "input_cid":      batch['cid'][idx_in_batch],
                    "input_uri":      batch['uri'][idx_in_batch],
                    "input_is_reply": batch['is_reply'][idx_in_batch],
                    "input_etype":    batch['embed_type'][idx_in_batch],
                    "text":           batch['text'][idx_in_batch].replace("\n", "\\n"),
                    "image_paths":    batch['images'][idx_in_batch],
                    "video_path":     batch['video'][idx_in_batch],
                    "platform_label": batch['platform_label'][idx_in_batch],
                    "annotator_label": batch['annotator_label'][idx_in_batch],
                    "model":          "llama_guard_4-12B",
                    "prompt_type":    prompt_type,
                    "prompt_mode":    prompt_mode,
                    "output": {
                        "raw":                     raw_output.replace("\n", "\\n"),
                        "safety_verdict":          verdict,
                        "predicted_category":      category,
                        "predicted_category_name": category_lookup.get(category) if category else None,
                    },
                    "processing_time":        elapsed / (batch_end - batch_start),
                    "input_tokenized_length": len(output_item.prompt_token_ids),
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
