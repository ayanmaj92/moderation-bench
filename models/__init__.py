import importlib


def load_model_vllm(model_name: str, **kwargs):
    """
    Dynamically load a vLLM model by name.

    Args:
        model_name (str): Model name key.
        **kwargs: Arguments for model initialization.

    Returns:
        Instantiated vLLM model.
    """
    # Map model_name → (module_path, class_name)
    registry = {
        "gemma3": ("models.vllm_models", "VLLMGemma3Model"),
        "gemma4": ("models.vllm_models", "VLLMGemma4Model"),
        "gemma4_thinking": ("models.vllm_models", "VLLMGemma4Model"),
        "mistral3": ("models.vllm_models", "VLLMMistral3Model"),
        "llava": ("models.vllm_models", "VLLMLlavaModel"),
        "intern3_5_vl": ("models.vllm_models", "VLLMInternVL3Model"),
        "intern3_5_vl_thinking": ("models.vllm_models", "VLLMInternVL3Model"),
        "qwen_vl": ("models.vllm_models", "VLLMQwenVLModel"),
        "qwen3_vl": ("models.vllm_models", "VLLMQwenVLModel"),
        "qwen3_vl_thinking": ("models.vllm_models", "VLLMQwenVLThinkingModel"),
        "qwen3_5": ("models.vllm_models", "VLLMQwen3_5Model"),
        "qwen3_5_thinking": ("models.vllm_models", "VLLMQwen3_5Model"),
        "magistral_small": ("models.vllm_models", "VLLMMagistralSmallReasoningModel"),
        "llama4_scout": ("models.vllm_models", "VLLMLlama4ScoutModel"),
    }

    key = model_name.lower()
    if key not in registry:
        raise ValueError(
            f"Unknown vLLM model '{model_name}'. "
            f"Available: {list(registry.keys())}"
        )
    if "qwen3_vl" in key:
        kwargs["model_name"] = "Qwen/Qwen3-VL"
    elif "intern3_5_vl" in key:
        kwargs["model_name"] = "OpenGVLab/InternVL3_5"
    module_name, class_name = registry[key]
    module = importlib.import_module(module_name)
    ModelClass = getattr(module, class_name)

    # Adjust kwargs for special cases
    if key in ["mistral3", "magistral_small", "aria", "llama4_scout", "llama4_scout_fp8"]:
        kwargs["model_size"] = None  # not needed for these

    load_kwargs = {
        "cache_dir": kwargs.pop("hf_cache", ""),
        "max_logprobs": kwargs.pop("max_logprobs", 20),
        "max_model_len": kwargs.pop("max_model_len", 32768),
        "gpu_memory_utilization": kwargs.pop("gpu_memory_utilization", 0.95),
        "max_num_seqs": kwargs.pop("max_num_seqs", 10),
        "tensor_parallel_size": kwargs.pop("tensor_parallel_size", 1),
        "num_video_frames": kwargs.pop("num_video_frames", 16),
    }
    hf_token = kwargs.pop("hf_token", "")
    if key in ["gemma3", "mistral3", "magistral_small", "aya_vision", "llama4_scout", "llama4_scout_fp8"] and hf_token:
        load_kwargs["hf_token"] = hf_token

    model = ModelClass(**kwargs)
    model.simple_name = key
    model.enable_thinking_in_chat = key in ["gemma4_thinking", "qwen3_5_thinking"]
    model.is_reasoning_model = model.is_reasoning_model or key in ["gemma4_thinking", "qwen3_5_thinking", "intern3_5_vl_thinking"]
    print(model.model_id)
    model.load(**load_kwargs)
    return model
