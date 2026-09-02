import os
import yaml
import argparse
import copy
import json

DEFAULT_MODEL_SIZES_DICT = {
    "gemma3": "27b", "qwen_vl": "32B", "llava": "72b",
    "intern3_vl": "38B", "aya_vision": "32b",
    "intern3_5_vl": "38B", "intern3_5_vl_thinking": "38B",
    "mistral3": "24B", "aria": "25B", "ovis2": "34B",
    "deepseek_vl2": "large", "qwen3_vl": "32B",
    "qwen3_vl_thinking": "32B",
    "magistral_small": "24B",
    "gemma4": "31B", "gemma4_thinking": "31B",
    "qwen3_5": "27B", "qwen3_5_thinking": "27B",
    "llama4_scout": "17B-16E",
    "llama4_scout_fp8": "17B-16E-FP8"
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return None


def load_config(args):
    with open(args.base_config, "r") as f:
        cfg = yaml.safe_load(f)

    # Deep copy to avoid mutating original dict
    cfg = copy.deepcopy(cfg)
    
    if "media_options" not in cfg["model"]:
        cfg["model"]["media_options"] = {}

    # Override top-level values
    # NOTE: Add others if needed!
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    # Override nested values
    if args.model_name:
        cfg["model"]["name"] = args.model_name
    if args.dataset:
        cfg['dataset']['test_path'] = args.dataset
    if args.neighbors_path:
        cfg['dataset']['neighbors_path'] = args.neighbors_path
    if args.safe_path:
        cfg['dataset']['safe_path'] = args.safe_path
    if args.model_size:
        cfg["model"]["size"] = args.model_size
    else:
        cfg["model"]["size"] = DEFAULT_MODEL_SIZES_DICT[cfg["model"]["name"]]
    if args.prompt_type:
        cfg["prompt"]["type"] = args.prompt_type
    if args.prompt_mode:
        cfg["prompt"]["mode"] = args.prompt_mode
    if args.device:
        cfg["model"]["device"] = args.device
    if args.max_new_tokens is not None:
        cfg["model"]["inference"]["max_new_tokens"] = args.max_new_tokens
    if args.resize_images is not None:
        cfg["model"]["inference"]["resize_images"] = args.resize_images
        # if args.resize_factor:
        #     cfg["model"]["inference"]["resize_factor"] = args.resize_factor
    if args.with_confidence is not None:
        cfg["prompt"]["output_confidence"] = args.with_confidence
    else:
        # Default this to True
        cfg["prompt"]["output_confidence"] = True
    if args.in_context_num:
        cfg["prompt"]["in_context_num"] = args.in_context_num
    if args.in_context_select:
        cfg["prompt"]["in_context_select"] = args.in_context_select
    if args.example_per_group is not None:
        cfg["prompt"]["example_per_group"] = args.example_per_group
    if args.group_batch_size is not None:
        cfg["dataset"]["group_batch_size"] = args.group_batch_size
    if args.model_name == "qwen_vl":
        cfg["model"]["inference"]["resize_images"] = True
    if args.num_frames:
        cfg["model"]["media_options"]["num_frames"] = args.num_frames
    
    # Set defaults if missing in config
    cfg["model"].setdefault("max_model_len", 8192)
    cfg["model"].setdefault("gpu_memory_utilization", 0.95)
    cfg["model"].setdefault("max_num_seqs", 1)
    cfg["model"].setdefault("tensor_parallel_size", 1)
    cfg["dataset"].setdefault("batch_size", 1)
    cfg["prompt"].setdefault("output_confidence", True)
    cfg["model"]["media_options"].setdefault("num_frames", 16)
    cfg["dataset"].setdefault("posts_with_video", False)
    cfg["dataset"].setdefault("safe_path", "")
    cfg["dataset"].setdefault("neighbors_path", "")
    cfg["prompt"].setdefault("example_per_group", False)
    cfg["dataset"].setdefault("group_batch_size", 5)

    # Override from CLI args when provided
    if args.max_model_len is not None:
        cfg["model"]["max_model_len"] = args.max_model_len
    if args.gpu_memory_utilization is not None:
        cfg["model"]["gpu_memory_utilization"] = args.gpu_memory_utilization
    # if args.max_num_seqs is not None:
    #     cfg["model"]["max_num_seqs"] = args.max_num_seqs
    if args.tensor_parallel_size is not None:
        cfg["model"]["tensor_parallel_size"] = args.tensor_parallel_size
    if args.batch_size is not None:
        cfg["dataset"]["batch_size"] = args.batch_size
        cfg["model"]["max_num_seqs"] = args.batch_size
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Bluesky labels for images using selected model.")

    # Select the function based on model choice
    parser.add_argument('--base_config', '-c', default="configs/inference_instruction_driven.yaml")
    parser.add_argument('--model_name', '-mn', default=None, help='Choose the model')
    parser.add_argument('--model_size', '-ms', default=None, help='Choose appropriate size')
    parser.add_argument('--prompt_type', '-pt', default=None, help='Choose the prompt type')
    parser.add_argument('--prompt_mode', '-p', default=None, help='Choose the mode')
    parser.add_argument('--output_dir', '-o', type=str, default=None)
    parser.add_argument('--device', '-d', type=str, default=None, help='Device to run the model on')
    parser.add_argument('--dataset', '-ds', type=str, default=None, help="path to the query test set")
    parser.add_argument('--neighbors_path', '-nb', type=str, default=None,
                         help="JSONL mapping each query uri -> its 'neighbors_per_label', joined to "
                              "--dataset by uri. Required for -ics contextual; ignored otherwise.")
    parser.add_argument('--safe_path', '-sp', type=str, default=None,
                         help="path to the safe/contrast example pool used for contextual in-context selection")
    parser.add_argument('--resize_images', '-ri', type=str2bool, default=None, help='Whether to resize images')
    # parser.add_argument('--resize_factor', type=float, default=None, help='Resize factor for images')
    parser.add_argument('--in_context_num', '-icn', type=int, default=None, help='Number of in-context examples to use')
    parser.add_argument('--in_context_select', '-ics', type=str, default=None, help='Method to select in-context examples')
    parser.add_argument('--max_model_len', '-mlen', type=int, default=None, help='Maximum model context length (e.g. 8192)')
    parser.add_argument('--gpu_memory_utilization', '-gmem', type=float, default=None, help='Fraction of GPU memory to utilize (e.g. 0.95)')
    # parser.add_argument('--max_num_seqs', '-mns', type=int, default=None, help='Maximum number of sequences (e.g. 1)')
    parser.add_argument('--max_new_tokens', '-mnt', type=int, default=None, help='Maximum new tokens to generate (e.g. 256)')
    parser.add_argument('--tensor_parallel_size', '-tps', type=int, default=None, help='Tensor parallel size (e.g. 1)')
    parser.add_argument('--batch_size', '-bs', type=int, default=None, help='Batch size for inference')
    parser.add_argument('--with_confidence', '-wc', type=str2bool, default=None, help='Whether to output confidence scores')
    parser.add_argument('--num_frames', '-nf', type=int, default=None, help='Number of frames to use for video inputs (if applicable)')
    parser.add_argument('--example_per_group', '-epg', type=str2bool, default=None,
                         help='Split example-driven examples into in_context_num-sized chunks and run one '
                              'model call per chunk, aggregating via majority vote, instead of one '
                              'combined mega-prompt')
    parser.add_argument('--group_batch_size', '-gbs', type=int, default=None,
                         help='Number of per-item example-group calls to batch together in one '
                              'vLLM call when --example_per_group is set')
    return parser.parse_args()

def load_examples(path):
    examples = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            if "images" not in obj:
                raise ValueError(
                    f"'{path}' has no 'images' field -- it looks like a uri-only seed file that "
                    f"hasn't been enriched yet. Run download_media_for_batch.py on it first."
                )
            data = {
                'text': obj.get('text'),
                'images': [i.get('file') for i in obj.get('images')],
                'label': obj.get('label', 'safe'),
                'video': obj.get('video'),
            }
            examples.append(data)
    return examples


############################################################
# Shared inference-runner glue (kept torch-free: ResultWriter
# imports torch lazily, so this module can be imported before
# utils.runtime.setup_cuda_env() runs).
############################################################

REASONING_DELIMITERS = {
    "qwen3_vl_thinking":     ("</think>",   "<think>"),
    "qwen3_5_thinking":      ("</think>",   "<think>"),
    "intern3_5_vl_thinking": ("</think>",   "<think>"),
    "magistral_small":       ("[/THINK]",   "[THINK]"),
    "gemma4_thinking":       ("<channel|>", "<|channel>thought"),
}

NATIVE_VIDEO_MODELS = ("qwen3_vl", "qwen3_5", "gemma4", "llava")

_GATED_MODELS = ("gemma3", "mistral3", "magistral_small", "llama4_scout")


def split_reasoning(output_txt, simple_name):
    """Split a thinking model's raw output into ``(reasoning, answer)``. Returns
    ``(None, output_txt)`` for non-thinking models or on any parse failure."""
    if simple_name not in REASONING_DELIMITERS:
        return None, output_txt
    end_tok, start_tok = REASONING_DELIMITERS[simple_name]
    try:
        parts = output_txt.split(end_tok)
        reasoning = parts[0].strip()
        if reasoning.startswith(start_tok):
            reasoning = reasoning[len(start_tok):].strip()
        reasoning = reasoning.replace("\n", "\\n")
        return reasoning, parts[1].strip()
    except Exception as e:
        print(f"Error splitting reasoning and answer: {e}")
        return None, output_txt


def build_vllm_model_kwargs(cfg, *, max_images, max_images_native=None,
                            native_video_models=NATIVE_VIDEO_MODELS):
    """Assemble kwargs for ``load_model_vllm(**model_kwargs)`` and the
    ``inference_kwargs`` for ``predict_with_chat`` from ``cfg``.

    Returns ``(model_kwargs, inference_kwargs, video_as_frames)``.

    ``max_images`` / ``max_images_native`` size the per-paradigm image budget:
    instruction-driven passes ``(num_frames, 4)``; example-driven passes
    ``(250, 250)`` to fit many demonstration images.
    """
    model_kwargs = {k: v for k, v in cfg["model"].items() if not isinstance(v, dict)}
    model_kwargs["model_name"] = model_kwargs.pop("name")
    model_kwargs["model_size"] = model_kwargs.pop("size")

    if cfg["model"]["name"] in _GATED_MODELS:
        model_kwargs["hf_token"] = os.getenv("HF_TOKEN", None)
    model_kwargs["hf_cache"] = os.getenv("HF_HUB_CACHE", None)

    num_frames = cfg["model"].get("media_options", {}).get("num_frames", 16)
    name_lower = model_kwargs["model_name"].lower()
    is_native = any(m in name_lower for m in native_video_models)

    if is_native and max_images_native is not None:
        model_kwargs["max_images"] = max_images_native
    else:
        model_kwargs["max_images"] = max_images
    model_kwargs["max_videos"] = 1 if is_native else 0
    if is_native:
        model_kwargs["num_video_frames"] = num_frames

    inference_kwargs = dict(cfg["model"]["inference"])
    inference_kwargs["num_video_frames"] = num_frames

    return model_kwargs, inference_kwargs, (not is_native)


class ResultWriter:
    """Buffered JSONL writer. Flushes every ``flush_every`` batches (emptying the
    CUDA cache on flush) plus a final flush on exit -- matching the original
    runner loops. Use as a context manager; call ``add(row)`` per result and
    ``end_batch()`` once per processed batch."""

    def __init__(self, path, flush_every=10):
        self.path = path
        self.flush_every = flush_every
        self._buf = []
        self._batches = 0
        self._fh = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")
        return self

    def add(self, row):
        self._buf.append(json.dumps(row))

    def end_batch(self):
        self._batches += 1
        if self._batches % self.flush_every == 0 and self._buf:
            self._flush()
            import torch
            torch.cuda.empty_cache()

    def _flush(self):
        if self._buf and self._fh is not None:
            self._fh.write("\n".join(self._buf) + "\n")
            self._fh.flush()
            self._buf.clear()

    def __exit__(self, *exc):
        self._flush()
        if self._fh is not None:
            self._fh.close()

