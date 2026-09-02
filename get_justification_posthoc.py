import os, re, subprocess

def cuda_version(nvcc_path):
    try:
        out = subprocess.run([nvcc_path, "--version"], capture_output=True,
                              text=True, timeout=10).stdout
        m = re.search(r"release (\d+)\.(\d+)", out)
        return tuple(map(int, m.groups())) if m else (0, 0)
    except Exception:
        return (0, 0)

result = subprocess.run(
    ["find", "/usr", "/opt", "/cm", "/software", "-name", "nvcc", "-type", "f"],
    capture_output=True, text=True, timeout=60
)

candidates = [p for p in result.stdout.strip().splitlines() if p]
if not candidates:
    print("nvcc not found")
else:
    ranked = sorted(((cuda_version(p), p) for p in candidates), reverse=True)
    for ver, p in ranked:
        print(f"  found {p} -> CUDA {ver[0]}.{ver[1]}")

    best_ver, best_path = ranked[0]
    if best_ver < (12, 0):
        print(f"WARNING: newest nvcc found is CUDA {best_ver[0]}.{best_ver[1]}, "
              f"insufficient for sm_90a (Hopper). Need >=12.0, ideally >=12.4.")

    cuda_home = os.path.dirname(os.path.dirname(best_path))
    os.environ["CUDA_HOME"] = cuda_home
    os.environ["PATH"] = f"{cuda_home}/bin:{os.environ.get('PATH', '')}"
    print(f"Using nvcc at {best_path} (CUDA {best_ver[0]}.{best_ver[1]})")

import argparse
import json
import time

_DEFAULT_MODEL_SIZES = {
    "gemma4_thinking": "31B",
    "gemma4":          "31B",
    "qwen3_5_thinking": "27B",
    "qwen3_5":         "27B",
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Justification inference over all instances in a data file."
    )
    parser.add_argument('--data_file', '-df', required=True,
                        help='Path to the source data JSONL (default: labeled pilot 1k)')
    parser.add_argument('--results_file', '-r', required=True,
                        help='Path to results JSONL from the original inference run')
    parser.add_argument('--output_dir', '-o', required=True,
                        help='Directory to save output JSONL')
    parser.add_argument('--model_name', '-mn', required=True,
                        help='Model key, e.g. gemma4_thinking, qwen3_5_thinking')
    parser.add_argument('--model_size', '-ms', type=str, default=None,
                        help='Model size override (default: inferred from model_name)')
    parser.add_argument('--batch_size', '-bs', type=int, default=10)
    parser.add_argument('--max_model_len', '-mlen', type=int, default=32768)
    parser.add_argument('--gpu_memory_utilization', '-gmem', type=float, default=0.9)
    parser.add_argument('--tensor_parallel_size', '-tps', type=int, default=1)
    parser.add_argument('--device', '-d', type=str, default='cuda:0')
    return parser.parse_args()

args = parse_args()

if args.device == 'cuda:0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
elif args.device == 'cuda:1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if "gemma4" in args.model_name.lower():
    os.environ["VLLM_USE_DEEP_GEMM"] = "0"

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from models import load_model_vllm
from utils.media import _load_single_image_as_base64
from prompt.instruction_driven import InstructionDrivenPrompt
from vllm import SamplingParams

_REASONING_DELIMITERS = {
    'qwen3_vl_thinking':     ('</think>',   '<think>'),
    'qwen3_5_thinking':      ('</think>',   '<think>'),
    'intern3_5_vl_thinking': ('</think>',   '<think>'),
    'magistral_small':       ('[/THINK]',   '[THINK]'),
    'gemma4_thinking':       ('<channel|>', '<|channel>thought'),
}

JUSTIFICATION_PROMPT = (
    "**EXPLANATION**: Provide a concise 2–3 sentence explanation of your final decision. Note:\n"
    "  First, briefly describe what specific element of the post (text, image, or both) triggered the decision.\n"
    "  Then, state the policy rule or criterion it violates, using the language of the policy.\n"
    "  If the post requires no moderation (S0), briefly state what the content is and why it does not meet the threshold for any category.\n"
    "  Do not describe your reasoning process or alternatives you considered. State only the justification for your final conclusion."
)

POLICY_QUOTES_PROMPT = (
    '**POLICY_QUOTES**: List exact verbatim excerpts from the policy text that are directly attributed to your final '
    'labeling decision. Each excerpt must be copied exactly as it appears in the policy, enclosed in quotes ("..."). '
    'This applies even when predicting no moderation (S0). For S0, quote specific text from the policy and community '
    'guidelines explaining why the content is allowed, not the S0 label definition itself.'
)

JUSTIFICATION_JSON_PROMPT = (
    "Provide your justification as a JSON object with exactly this structure:\n\n"
    "```json\n"
    "{\n"
    '    "POLICY_QUOTES": ["verbatim excerpt from policy", "..."],\n'
    '    "EXPLANATION": "2-3 sentences justifying your decision"\n'
    "}\n"
    "```\n\n"
    + JUSTIFICATION_PROMPT + "\n\n"
    + POLICY_QUOTES_PROMPT
)


def _extract_answer(raw: str, model_name: str) -> str:
    """Strip thinking tokens from a stored raw output and return the answer portion."""
    text = raw.replace('\\n', '\n')
    if model_name not in _REASONING_DELIMITERS:
        return text
    end_tok, _ = _REASONING_DELIMITERS[model_name]
    parts = text.split(end_tok)
    return parts[1].strip() if len(parts) > 1 else text


def load_images_for_row(image_paths):
    if not image_paths:
        return None
    tasks = [(i, p) for i, p in enumerate(image_paths) if p]
    if not tasks:
        return None
    results = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as executor:
        futures = {
            executor.submit(_load_single_image_as_base64, p, resize_images=False): i
            for i, p in tasks
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    loaded = [img for img in results if img is not None]
    return loaded if loaded else None


def build_justification_messages(prompt, row, model_name):
    """Build a 4-turn conversation: system → user (original prompt) → assistant (prev answer) → user (justify)."""
    text = row.get('text', '').replace('\\n', '\n')
    loaded_images = load_images_for_row(row.get('image_paths'))

    messages = prompt.build_chat_messages(
        query_text=text,
        query_images=loaded_images,
        query_video=None,
        video_as_frames=False,
    )

    prev_answer = _extract_answer(row['output'].get('raw', ''), model_name)
    messages.append({"role": "assistant", "content": prev_answer})
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": JUSTIFICATION_JSON_PROMPT}],
    })
    return messages


def parse_justification_response(text: str) -> dict:
    """Extract POLICY_QUOTES and EXPLANATION from the model's JSON response."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    obj_match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if obj_match:
        try:
            parsed = json.loads(obj_match.group())
            return {
                "policy_quotes": parsed.get("POLICY_QUOTES", []),
                "explanation":   parsed.get("EXPLANATION", ""),
            }
        except json.JSONDecodeError:
            pass
    return {"policy_quotes": [], "explanation": text.strip()}


def main():
    # Collect ordered CIDs from the data file
    data_cids = []
    with open(args.data_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = item.get('cid')
            if cid:
                data_cids.append(cid)

    cid_set = set(data_cids)
    print(f"Loaded {len(data_cids)} CIDs from data file ({len(cid_set)} unique)")

    # Index results file by CID
    rows_by_cid = {}
    with open(args.results_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = row.get('input_cid')
            if cid in cid_set:
                rows_by_cid[cid] = row

    print(f"Found {len(rows_by_cid)} matching rows in results file")
    missing = cid_set - set(rows_by_cid)
    if missing:
        print(f"Warning: {len(missing)} CIDs from data file not found in results file")

    rows_list = [rows_by_cid[cid] for cid in data_cids if cid in rows_by_cid]
    print(f"Processing {len(rows_list)} instances")

    prompt = InstructionDrivenPrompt(mode='with_labels_rationale_details', platform='bluesky')

    model_name = args.model_name
    model_size = args.model_size or _DEFAULT_MODEL_SIZES.get(model_name, "unknown")

    native_video_models = {"qwen3_vl", "qwen3_5", "gemma4", "llava"}
    native_video = any(x in model_name.lower() for x in native_video_models)

    model = load_model_vllm(
        model_name,
        model_size=model_size,
        device=args.device,
        hf_token=os.getenv("HF_TOKEN", ""),
        hf_cache=os.getenv("HF_HUB_CACHE", ""),
        max_logprobs=20,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        num_video_frames=16,
        max_images=4 if native_video else 16,
        max_videos=1 if native_video else 0,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=32768,
        top_p=1.0,
        top_k=1,
        repetition_penalty=1.0,
        skip_special_tokens=False,
    )

    chat_template_kwargs = {"enable_thinking": False} if model.enable_thinking_in_chat else None

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{model_name}_{model_size}_justification.jsonl")

    buffer = []

    with open(out_path, 'w', encoding='utf-8') as outfile:
        for batch_start in tqdm(
            range(0, len(rows_list), args.batch_size),
            total=(len(rows_list) + args.batch_size - 1) // args.batch_size,
            desc="Processing",
        ):
            batch = rows_list[batch_start: batch_start + args.batch_size]

            batch_messages = []
            for row in batch:
                try:
                    msgs = build_justification_messages(prompt, row, model_name)
                except Exception as e:
                    print(f"Error building messages for {row.get('input_cid')}: {e}")
                    msgs = []
                batch_messages.append(msgs)

            valid = [(msg, row) for msg, row in zip(batch_messages, batch) if msg]
            if not valid:
                continue
            valid_messages, valid_rows = zip(*valid)

            start = time.time()
            try:
                chat_kwargs = {}
                if chat_template_kwargs is not None:
                    chat_kwargs["chat_template_kwargs"] = chat_template_kwargs
                outputs = model.llm.chat(
                    list(valid_messages),
                    sampling_params,
                    use_tqdm=False,
                    **chat_kwargs,
                )
            except Exception as e:
                print(f"Error during inference at batch {batch_start}: {e}")
                continue
            elapsed = time.time() - start

            for row, output_item in zip(valid_rows, outputs):
                raw_response = output_item.outputs[0].text.strip()
                answer = _extract_answer(raw_response, model_name)
                parsed = parse_justification_response(answer)
                result_row = {
                    "input_cid":            row.get('input_cid'),
                    "input_uri":            row.get('input_uri'),
                    "text":                 row.get('text'),
                    "image_paths":          row.get('image_paths'),
                    "video_path":           row.get('video_path'),
                    "annotator_label":      row.get('annotator_label'),
                    "model":                row.get('model'),
                    "original_prediction":  row['output'].get('PREDICTED_CATEGORY_ID'),
                    "original_confidence":  row['output'].get('CONFIDENCE_SCORE'),
                    "policy_quotes":        parsed["policy_quotes"],
                    "explanation":          parsed["explanation"],
                    "justification_raw":    raw_response.replace('\n', '\\n'),
                    "processing_time":      elapsed / len(valid_rows),
                }
                buffer.append(json.dumps(result_row))

            if len(buffer) >= 50:
                outfile.write('\n'.join(buffer) + '\n')
                outfile.flush()
                buffer.clear()

        if buffer:
            outfile.write('\n'.join(buffer) + '\n')
            outfile.flush()

    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
