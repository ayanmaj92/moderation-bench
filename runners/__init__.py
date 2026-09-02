"""Per-paradigm inference runners for open-weight local (vLLM) models.

Dispatched from ``main.py`` on ``cfg["prompt"]["type"]``. Import the runner
modules ONLY through ``get_runner()`` -- they pull in torch/vLLM, and
``utils.runtime.setup_cuda_env()`` must have run first.
"""

import importlib

_RUNNERS = {
    "instruction_driven": "runners.instruction_driven",
    "example_driven": "runners.example_driven",
}


def get_runner(paradigm):
    key = (paradigm or "").lower()
    if key not in _RUNNERS:
        raise SystemExit(
            f"unknown paradigm {paradigm!r}; expected one of {sorted(_RUNNERS)} "
            f"(set prompt.type in the config or pass -pt/--prompt_type)"
        )
    return importlib.import_module(_RUNNERS[key])
