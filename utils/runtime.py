"""Runtime environment bootstrap shared by every local-vLLM entry point.

`setup_cuda_env()` must run *before* any `import torch` / `import vllm`, so this
module is deliberately dependency-free (stdlib only). Entry points call it right
after parsing args/config and before the heavy imports.
"""

import os
import re
import subprocess


def cuda_version(nvcc_path):
    try:
        out = subprocess.run(
            [nvcc_path, "--version"], capture_output=True, text=True, timeout=10
        ).stdout
        m = re.search(r"release (\d+)\.(\d+)", out)
        return tuple(map(int, m.groups())) if m else (0, 0)
    except Exception:
        return (0, 0)


def _discover_cuda_home(set_ld_library_path=False):
    """Find the newest CUDA toolkit on the machine and point CUDA_HOME / PATH
    (and, only if ``set_ld_library_path``, LD_LIBRARY_PATH) at it. Needed on
    clusters where the default nvcc is too old for Hopper (sm_90a)."""
    try:
        result = subprocess.run(
            ["find", "/usr", "/opt", "/cm", "/software", "-name", "nvcc", "-type", "f"],
            capture_output=True, text=True, timeout=60,
        )
    except Exception as e:
        print(f"nvcc discovery failed: {e}")
        return

    candidates = [p for p in result.stdout.strip().splitlines() if p]
    if not candidates:
        print("nvcc not found")
        return

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
    if set_ld_library_path:
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_home}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
    print(f"Using nvcc at {best_path} (CUDA {best_ver[0]}.{best_ver[1]})")


def setup_cuda_env(cfg=None, *, device=None, model_name=None, ld_library_path=False):
    """Bootstrap CUDA/vLLM environment.

    Always: locate a recent CUDA toolkit and export CUDA_HOME / PATH (this is
    exactly what the pre-refactor entry point did). Pass ``ld_library_path=True``
    to also prepend ``$CUDA_HOME/lib64`` to LD_LIBRARY_PATH -- only
    ``shieldstral.py`` needs that (newer vLLM).

    If a device is given (directly or via ``cfg['model']['device']``): pin
    CUDA_VISIBLE_DEVICES. If a gemma4 model is given (directly or via
    ``cfg['model']['name']``): set VLLM_USE_DEEP_GEMM=0.

    The safety-model scripts call this with no cfg (they manage device placement
    through the environment themselves).
    """
    _discover_cuda_home(set_ld_library_path=ld_library_path)

    if cfg is not None:
        device = device or cfg.get("model", {}).get("device")
        model_name = model_name or cfg.get("model", {}).get("name")

    if device == "cuda:0":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif device == "cuda:1":
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    if model_name and "gemma4" in model_name.lower():
        os.environ["VLLM_USE_DEEP_GEMM"] = "0"
