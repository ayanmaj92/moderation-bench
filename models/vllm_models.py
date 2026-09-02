from .base_model import BaseModel
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from utils.media import load_batch_media


def _apply_reasoning_system_prompt(messages, model_id, simple_name):
    has_system = messages and messages[0]["role"] == "system"
    if "magistral" in model_id.lower():
        existing_content = messages[0]["content"] if has_system else ""
        system_msg_magistral = f"""
# HOW YOU SHOULD THINK AND ANSWER

First draft your thinking process (inner monologue) until you arrive at a response. Format your response using Markdown, and use LaTeX for any mathematical equations. Write both your thoughts and the response in the same language as the input.

Your thinking process must follow the template below:[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate the response to the user.[/THINK]Here, provide a self-contained response.\n\n{existing_content}"""
        index_begin_think = system_msg_magistral.find("[THINK]")
        index_end_think = system_msg_magistral.find("[/THINK]")
        magistral_system = {
            "role": "system",
            "content": [
                {"type": "text", "text": system_msg_magistral[:index_begin_think]},
                {
                    "type": "thinking",
                    "thinking": system_msg_magistral[index_begin_think + len("[THINK]"):index_end_think],
                    "closed": True,
                },
                {"type": "text", "text": system_msg_magistral[index_end_think + len("[/THINK]"):]},
            ],
        }
        return [magistral_system] + messages
    elif simple_name == "intern3_5_vl_thinking":
        R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()
        if has_system:
            existing_content = messages[0]["content"]
            combined = (
                R1_SYSTEM_PROMPT + "\n\n" + existing_content
                if isinstance(existing_content, str)
                else [{"type": "text", "text": R1_SYSTEM_PROMPT + "\n\n"}] + existing_content
            )
            messages[0] = {"role": "system", "content": combined}
        else:
            messages = [{"role": "system", "content": R1_SYSTEM_PROMPT}] + messages
    return messages


def _build_sampling_params(do_sample, max_new_tokens, temperature, top_p, reasoning_model=False, **extra):
    return SamplingParams(
        temperature=temperature if do_sample else 0.0,
        max_tokens=max_new_tokens,
        top_p=top_p if do_sample else 1.0,
        top_k=0 if do_sample else 1,
        **({"repetition_penalty": 1.2,
            "thinking_token_budget": 3000,
            "skip_special_tokens": False} if reasoning_model else {}),
        **extra,
    )


############################################################
# Base VLLM model
############################################################
class VLLMModel(BaseModel):
    """Base for vLLM-backed models with standard inference."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simple_name = None
        self.enable_thinking_in_chat = False
        self.is_reasoning_model = False

    def predict_with_chat(self, batch_texts, batch_image_paths, prompt_builder, batch_video_paths=None, **kwargs):
        resize_images = kwargs.pop("resize_images", False)
        resize_factor = kwargs.pop("resize_factor", 0.5)
        max_pixels = kwargs.pop("max_pixels", None)
        max_new_tokens = kwargs.pop("max_new_tokens", 100)
        do_sample = kwargs.pop("do_sample", False)
        temperature = kwargs.pop("temperature", 0.4)
        top_p = kwargs.pop("top_p", 0.9)
        example_k_texts_for_batch = kwargs.pop("example_k_texts", None)
        example_k_images_for_batch = kwargs.pop("example_k_images", None)
        example_k_labels_for_batch = kwargs.pop("example_k_labels", None)
        reasoning_model = kwargs.pop("reasoning_model", False)
        num_frames = kwargs.pop("num_video_frames", 16)
        self.video_as_frames = kwargs.pop("video_as_frames", False)

        if batch_video_paths is None:
            batch_video_paths = [None] * len(batch_texts)

        batch_media = load_batch_media(
            self.model_id, batch_image_paths, batch_video_paths,
            resize_images, resize_factor, self.video_as_frames,
            num_frames, example_k_images_for_batch,
            max_pixels=max_pixels,
        )

        all_messages = []
        for idx in range(len(batch_texts)):
            data = batch_media[idx]
            query_images_loaded = data["query_images"] or None
            query_video_loaded = data["query_video"]
            example_driven_image_loaded = data["example_images"] or None
            example_k_texts = example_k_texts_for_batch[idx] if example_k_texts_for_batch is not None else None
            example_k_labels = example_k_labels_for_batch[idx] if example_k_labels_for_batch is not None else None

            assert not (query_images_loaded and query_video_loaded), \
                "Currently does not support both images and video in a single instance since Bluesky does not either."

            prompt_kwargs = dict(
                query_text=batch_texts[idx],
                query_images=query_images_loaded,
                query_video=query_video_loaded,
                video_as_frames=self.video_as_frames,
            )
            if example_driven_image_loaded is not None and example_k_texts is not None and example_k_labels is not None:
                prompt_kwargs.update(
                    example_k_texts=example_k_texts,
                    example_k_images=example_driven_image_loaded,
                    example_k_labels=example_k_labels,
                )
            messages = prompt_builder(**prompt_kwargs)

            messages = _apply_reasoning_system_prompt(messages, self.model_id, self.simple_name)

            all_messages.append(messages)

        reasoning_model = reasoning_model or self.is_reasoning_model
        sampling_params = _build_sampling_params(
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_model=reasoning_model,
            **kwargs,
        )

        if "Qwen3.5" in self.model_id and not self.enable_thinking_in_chat:
            chat_template_kwargs = {"enable_thinking": False}
        elif self.enable_thinking_in_chat and "Qwen3.5" not in self.model_id:
            chat_template_kwargs = {"enable_thinking": True}
        else:
            chat_template_kwargs = None
        return self.llm.chat(
            all_messages,
            sampling_params=sampling_params,
            use_tqdm=False,
            **({"chat_template_kwargs": chat_template_kwargs} if chat_template_kwargs else {}),
        )
            
############################################################
# Standard subclasses
############################################################
class VLLMGemma3Model(VLLMModel):
    def __init__(self, model_name="google/gemma-3", model_size="27b", device="cuda:0", max_images=5, max_videos=0):
        model_size = model_size.lower()
        ALLOWED = ["270m", "1b", "4b", "12b", "27b"]
        if model_size not in ALLOWED:
            raise ValueError(f"Invalid Gemma3 size {model_size}, allowed: {ALLOWED}")
        self.model_id = f"{model_name}-{model_size}-it"
        super().__init__(model_name, model_size, device)
        self.max_images, self.max_videos = max_images, max_videos

    def load(self, **kwargs):
        self.llm = LLM(
            model=self.model_id,
            max_model_len=kwargs.get("max_model_len", 8192),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.9),
            max_num_seqs=kwargs.get("max_num_seqs", 1),
            dtype="bfloat16",
            limit_mm_per_prompt={"image": self.max_images, "video": 0, "audio": 0},
            download_dir=kwargs["cache_dir"],
            hf_token=kwargs["hf_token"],
            disable_log_stats=True,
            max_logprobs=kwargs["max_logprobs"],
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=kwargs["cache_dir"])


class VLLMGemma4Model(VLLMModel):
    def __init__(self, model_name="google/gemma-4", model_size="31B", device="cuda:0", max_images=5, max_videos=0):
        ALLOWED = ["31B", "26B-A4B", "E4B", "E2B"]
        if model_size not in ALLOWED:
            raise ValueError(f"Invalid Gemma4 size {model_size}, allowed: {ALLOWED}")
        self.model_id = f"{model_name}-{model_size}-it"
        super().__init__(model_name, model_size, device)
        self.max_images, self.max_videos = max_images, max_videos

    def load(self, **kwargs):
        from vllm.config import ReasoningConfig
        self.llm = LLM(
            model=self.model_id,
            max_model_len=kwargs.get("max_model_len", 32768),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.95),
            max_num_seqs=kwargs.get("max_num_seqs", 1),
            limit_mm_per_prompt={"image": self.max_images, "video": self.max_videos, "audio": 0},
            media_io_kwargs={"video": {"num_frames": kwargs.get("num_video_frames", 16)}},
            download_dir=kwargs["cache_dir"],
            disable_log_stats=True,
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
            trust_remote_code=True,
            reasoning_config=ReasoningConfig(
                reasoning_start_str="<|channel>thought\n",
                reasoning_end_str="\nI have limited thinking budget, I have to give the final solution based on the current thinking directly now.<channel|>"
            )
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=kwargs["cache_dir"])


class VLLMQwen3_5Model(VLLMModel):
    def __init__(self, model_name="Qwen/Qwen3.5", model_size="27B", device="cuda:0", max_images=5, max_videos=0):
        model_size = model_size.upper()
        ALLOWED_SIZES = ["0.8B", "2B", "4B", "9B", "27B", "35B-A3B", "122B-A10B", "397-A17B"]
        if model_size not in ALLOWED_SIZES:
            raise ValueError(
                f"Invalid model size '{model_size}' for {model_name}. "
                f"Allowed sizes: {ALLOWED_SIZES}"
            )
        self.model_id = f"{model_name}-{model_size}"
        super().__init__(model_name, model_size, device)
        self.max_images, self.max_videos = max_images, max_videos

    def load(self, **kwargs):
        from vllm.config import ReasoningConfig
        self.llm = LLM(
            model=self.model_id,
            max_model_len=kwargs.get("max_model_len", 32768),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.95),
            max_num_seqs=kwargs.get("max_num_seqs", 1),
            limit_mm_per_prompt={"image": self.max_images, "video": self.max_videos, "audio": 0},
            media_io_kwargs={"video": {"num_frames": kwargs.get("num_video_frames", 16)}},
            download_dir=kwargs["cache_dir"],
            disable_log_stats=True,
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
            trust_remote_code=True,
            reasoning_config=ReasoningConfig(
                reasoning_start_str="<think>",
                reasoning_end_str="\nI have limited thinking budget, I have to give the final solution based on the current thinking directly now.</think>"
            )
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=kwargs["cache_dir"])


class VLLMQwenVLModel(VLLMModel):
    def __init__(self, model_name="Qwen/Qwen3-VL", model_size="32B", device="cuda:0", max_images=5, max_videos=0):
        model_size = model_size.upper()
        ALLOWED_SIZES = ["2B", "4B", "8B", "32B", "30B-A3B", "235B-A22B"]
        if model_size not in ALLOWED_SIZES:
            raise ValueError(
                f"Invalid model size '{model_size}' for {model_name}. "
                f"Allowed sizes: {ALLOWED_SIZES}"
            )
        self.model_id = f"{model_name}-{model_size}-Instruct"
        super().__init__(model_name, model_size, device)
        self.max_images, self.max_videos = max_images, max_videos

    def load(self, **kwargs):
        self.llm = LLM(
            model=self.model_id,
            max_model_len=kwargs.get("max_model_len", 32768),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.95),
            max_num_seqs=kwargs.get("max_num_seqs", 1),
            limit_mm_per_prompt={"image": self.max_images, "video": self.max_videos, "audio": 0},
            download_dir=kwargs["cache_dir"],
            disable_log_stats=True,
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=kwargs["cache_dir"])


class VLLMQwenVLThinkingModel(VLLMModel):
    def __init__(self, model_name="Qwen/Qwen3-VL", model_size="32B", device="cuda:0", max_images=5, max_videos=0):
        model_size = model_size.upper()
        ALLOWED_SIZES = ["2B", "4B", "8B", "32B", "30B-A3B", "235B-A22B"]
        if model_size not in ALLOWED_SIZES:
            raise ValueError(
                f"Invalid model size '{model_size}' for {model_name}. "
                f"Allowed sizes: {ALLOWED_SIZES}"
            )
        self.model_id = f"{model_name}-{model_size}-Thinking"
        super().__init__(model_name, model_size, device)
        self.max_images, self.max_videos = max_images, max_videos
        self.is_reasoning_model = True

    def load(self, **kwargs):
        from vllm.config import ReasoningConfig
        self.llm = LLM(
            model=self.model_id,
            max_model_len=kwargs.get("max_model_len", 32768),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.95),
            max_num_seqs=kwargs.get("max_num_seqs", 1),
            limit_mm_per_prompt={"image": self.max_images, "video": self.max_videos, "audio": 0},
            media_io_kwargs={"video": {"num_frames": kwargs.get("num_video_frames", 16)}},
            download_dir=kwargs["cache_dir"],
            disable_log_stats=True,
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
            reasoning_config=ReasoningConfig(
                reasoning_start_str="<think>",
                reasoning_end_str="\nI have limited thinking budget, I have to give the final solution based on the current thinking directly now.</think>"
            )
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=kwargs["cache_dir"])


class VLLMMistral3Model(VLLMModel):
    def __init__(self, model_name="mistralai/Mistral-Small-3.2-24B-Instruct-2506", model_size=None, device="cuda:0", max_images=5, max_videos=0):
        self.max_images, self.max_videos = max_images, max_videos
        super().__init__(model_name, None, device)
        self.model_id = model_name

    def load(self, **kwargs):
        self.llm = LLM(
            model=self.model_id,
            max_model_len=kwargs.get("max_model_len", 32768),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.95),
            max_num_seqs=kwargs.get("max_num_seqs", 1),
            limit_mm_per_prompt={"image": self.max_images, "video": 0, "audio": 0},
            download_dir=kwargs["cache_dir"],
            hf_token=kwargs["hf_token"],
            disable_log_stats=True,
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
            tokenizer_mode="mistral", config_format="mistral", load_format="mistral"
        )


class VLLMMagistralSmallReasoningModel(VLLMModel):
    def __init__(self, model_name="mistralai/Magistral-Small-2509", model_size=None, device="cuda:0", max_images=5, max_videos=0):
        self.max_images, self.max_videos = max_images, max_videos
        super().__init__(model_name, None, device)
        self.model_id = model_name
        self.is_reasoning_model = True

    def load(self, **kwargs):
        from vllm.config import ReasoningConfig
        self.llm = LLM(
            model=self.model_id,
            max_model_len=kwargs.get("max_model_len", 32768),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.9),
            max_num_seqs=kwargs.get("max_num_seqs", 1),
            limit_mm_per_prompt={"image": self.max_images, "video": 0, "audio": 0},
            download_dir=kwargs["cache_dir"],
            hf_token=kwargs["hf_token"],
            disable_log_stats=True,
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
            tokenizer_mode="mistral", config_format="mistral", load_format="mistral",
            reasoning_config=ReasoningConfig(
                reasoning_start_str="[THINK]",
                reasoning_end_str="\nI have limited thinking budget, I have to give the final solution based on the current thinking directly now.[/THINK]"
            )
        )


class VLLMLlama4ScoutModel(VLLMModel):
    def __init__(self, model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct", model_size=None, device="cuda:0", max_images=5, max_videos=0):
        self.max_images, self.max_videos = max_images, max_videos
        super().__init__(model_name, None, device)
        self.model_id = model_name

    def load(self, **kwargs):
        self.llm = LLM(
            model=self.model_id,
            max_model_len=kwargs.get("max_model_len", 32768),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.9),
            max_num_seqs=kwargs.get("max_num_seqs", 1),
            limit_mm_per_prompt={"image": self.max_images, "video": 0, "audio": 0},
            download_dir=kwargs["cache_dir"],
            hf_token=kwargs["hf_token"],
            disable_log_stats=True,
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
        )


class VLLMLlavaModel(VLLMModel):
    def __init__(self, model_name="llava-hf/llava-onevision-qwen2", model_size="72b", device="cuda:0", max_images=5, max_videos=0):
        self.max_images, self.max_videos = max_images, max_videos
        ALLOWED_SIZES = ["7b", "72b"]
        if model_size not in ALLOWED_SIZES:
            raise ValueError(
                f"Invalid model size '{model_size}' for {model_name}. "
                f"Allowed sizes: {ALLOWED_SIZES}"
            )
        self.model_id = f"{model_name}-{model_size}-ov-hf"
        super().__init__(model_name, None, device)

    def load(self, **kwargs):
        self.llm = LLM(
            model=self.model_id,
            max_model_len=kwargs.get("max_model_len", 32768),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.95),
            max_num_seqs=kwargs.get("max_num_seqs", 1),
            limit_mm_per_prompt={"image": self.max_images, "video": self.max_videos, "audio": 0},
            media_io_kwargs={"video": {"num_frames": kwargs.get("num_video_frames", 16)}},
            download_dir=kwargs["cache_dir"],
            disable_log_stats=True,
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 2),
        )



class VLLMInternVL3Model(VLLMModel):
    def __init__(self, model_name="OpenGVLab/InternVL3_5", model_size="38B", device="cuda:0", max_images=5, max_videos=0):
        model_size = model_size.upper()
        if "InternVL3_5" in model_name:
            print("Using InternVL3_5 model")
            ALLOWED_SIZES = ["1B", "2B", "4B", "8B", "14B", "38B", "30B-A3B", "241B-A28B"]
        else:
            ALLOWED_SIZES = ["1B", "2B", "8B", "14B", "38B", "78B"]
        if model_size not in ALLOWED_SIZES:
            raise ValueError(
                f"Invalid model size '{model_size}' for {model_name}. "
                f"Allowed sizes: {ALLOWED_SIZES}"
            )
        super().__init__(model_name, model_size, device)
        self.model_id = f"{self.model_name}-{self.model_size}"
        self.max_images, self.max_videos = max_images, max_videos


    def load(self, **kwargs):
        from vllm.config import ReasoningConfig
        self.llm = LLM(
            model=self.model_id,
            max_model_len=kwargs.get("max_model_len", 32768),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.95),
            max_num_seqs=kwargs.get("max_num_seqs", 1),
            limit_mm_per_prompt={"image": self.max_images, "video": 0, "audio": 0},
            download_dir=kwargs["cache_dir"],
            disable_log_stats=True,
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
            trust_remote_code=True,
            reasoning_config=ReasoningConfig(
                reasoning_start_str="<think>",
                reasoning_end_str="\nI have limited thinking budget, I have to give the final solution based on the current thinking directly now.</think>"
            )
        )
