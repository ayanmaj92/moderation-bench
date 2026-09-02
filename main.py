"""Single entry point for open-weight local (vLLM) moderation inference.

NOTE: Not for API-based inference or safety models, e.g., llama-guard or shieldstral.

Dispatches on the paradigm (``prompt.type`` in the config, overridable with
``-pt/--prompt_type``):

    instruction_driven  -> runners.instruction_driven   (instruction-driven: policy only)
    example_driven   -> runners.example_driven     (example-driven: in-context demonstrations)

    python main.py -c configs/inference_instruction_driven.yaml -mn gemma4_thinking ...
    python main.py -c configs/inference_example_driven.yaml -mn gemma4_thinking -ics contextual ...
"""

# Parse + configure BEFORE any heavy import: setup_cuda_env() exports
# CUDA_HOME/PATH/CUDA_VISIBLE_DEVICES/VLLM_* which only take effect if set
# before torch/vLLM are imported. Only stdlib + these two torch-free modules
# are imported at this point; the runner (which pulls in torch/vLLM) is
# imported lazily afterwards.
from utils.runtime import setup_cuda_env
from utils.main_helpers import load_config, parse_args


def main():
    args = parse_args()
    cfg = load_config(args)
    setup_cuda_env(cfg)

    from runners import get_runner
    get_runner(cfg["prompt"]["type"]).run(cfg)


if __name__ == "__main__":
    main()
