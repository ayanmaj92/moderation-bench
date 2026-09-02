<h1><img src="logo.png" alt="logo" height="35" style="vertical-align:middle;"/> ModerationBench</h1>

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![HF Dataset](https://img.shields.io/badge/Dataset-ModerationBench--4K-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/ayanmaj/ModerationBench-4K)
[![Project Website](https://img.shields.io/badge/Project-Website-1f6feb)](https://moderation-bench.github.io/)
<!-- [![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) -->

A multimodal content moderation benchmark for evaluating vision-language models in real-world content moderation in the context of the Bluesky social media platform. 

## Environment Setup

### 1. Environment variables

Add to `~/.bashrc`:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
export HF_HOME=/path/to/huggingface/cache
export HF_HUB_CACHE=/path/to/huggingface/hub/cache
```

### 2. Create the venv

If uv is not installed, check this [link](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv venv /path/to/venvs/vlm_modbench --python 3.12
source /path/to/venvs/vlm_modbench/bin/activate
```

### 3. Install dependencies

Run `setup.sh` to setup the environment (creates the venv and installs everything):

```bash
bash setup.sh /path/to/venvs/vlm_modbench
```

#### Tested versions

```
torch                2.10.0+cu128
transformers         5.6.2
vllm                 0.19.1
mistral-common       1.11.0
```

### 4. Get the data

`data_files/` ships four `*_uri.jsonl` files (one per subset) with the post URI, `platform_label`, and `annotator_label`, e.g. `moderated_uri.jsonl`.

#### Option A — reconstruct from URIs (faster; no approval needed)

Set the code env and run `bash scripts/get_data_files.sh`. This fetches each post and its media from Bluesky directly into `*_metadata.jsonl` files.

**Note:** Some posts may have been deleted or taken down since collection and will fail to retrieve.

All downstream scripts read `data_files/<subset>_metadata.jsonl` and must be run **from the repo root** so the relative `data_files/...` media paths resolve.


#### Option A — pre-fetched dataset from Hugging Face (complete; gated access)

The full benchmark (post content + all images/videos, including posts since deleted from Bluesky) is released as a **gated** dataset, [`ayanmaj/ModerationBench-4K`](https://huggingface.co/datasets/ayanmaj/ModerationBench-4K).

1. Sign in to Hugging Face, open the dataset page, click **"Agree and access repository"**, and submit the request. Wait for approval to access the dataset.
2. Create a **read** access token at <https://huggingface.co/settings/tokens>.
3. From the repo root, with the venv active:

   ```bash
   export HF_TOKEN=hf_your_read_token   # or pass --token, or use `huggingface-cli login`
   python utils/fetch_hf_dataset.py
   ```

   This downloads `data_files/<subset>_metadata.jsonl` plus `data_files/images/` and `data_files/videos/`, then verifies every media path resolves. Re-run to resume an interrupted download; `pip install hf_transfer` makes the media download faster.



## Bluesky Moderation Categories

| ID  | Name               |
|-----|--------------------|
| S0  | no-moderation      |
| S1  | porn               |
| S2  | sexual             |
| S3  | sexual-figurative  |
| S4  | self-harm          |
| S5  | nudity             |
| S6  | intolerant         |
| S7  | graphic-media      |
| S8  | rude               |
| S9  | threat             |
| S10 | other-unsafe       |

## Supported Models

| Key                    | Class                              | Default Size  | Gated |
|------------------------|------------------------------------|---------------|-------|
| `gemma3`               | VLLMGemma3Model                    | 27b           |  yes  |
| `gemma4`               | VLLMGemma4Model                    | 31B           |       |
| `gemma4_thinking`      | VLLMGemma4Model                    | 31B           |       |
| `mistral3`             | VLLMMistral3Model                  | 24B           |  yes  |
| `magistral_small`      | VLLMMagistralSmallReasoningModel   | 24B           |  yes  |
| `llama4_scout`         | VLLMLlama4ScoutModel               | 17B-16E       |  yes  |
| `qwen3_vl`             | VLLMQwenVLModel                    | 32B           |       |
| `qwen3_vl_thinking`    | VLLMQwenVLThinkingModel            | 32B           |       |
| `qwen3_5`              | VLLMQwen3_5Model                   | 27B           |       |
| `qwen3_5_thinking`     | VLLMQwen3_5Model                   | 27B           |       |
| `intern3_5_vl`         | VLLMInternVL3Model                 | 38B           |       |
| `intern3_5_vl_thinking`| VLLMInternVL3Model                 | 38B           |       |
| `llava`                | VLLMLlavaModel                     | 72b           |       |

Thinking variants (`gemma4_thinking`, `qwen3_5_thinking`, `qwen3_vl_thinking`, `magistral_small`, `intern3_5_vl_thinking`) automatically use reasoning sampling parameters and strip thinking tokens from the output.

## Prompt Modes

Policy files live in `prompt/policies/bluesky/`. Pass the filename stem as the `mode`:

| Mode                         | Description                                              |
|------------------------------|----------------------------------------------------------|
| `with_labels`                | Label names and one-line definitions only                |
| `with_labels_details`        | Labels + detailed scope descriptions                     |
| `with_labels_rationale`      | Labels + rationale for each decision                     |
| `with_labels_rationale_details` | Labels + rationale + full scope details (Full platform policy) |

---

## Dataset

**Code**: `dataset_class/bluesky_dataset.py`

`BlueskyDataset` is the single class that loads data for every subset and both
paradigms; it
fills a superset of columns and leaves the ones a given file doesn't have empty.
It is loaded via `load_dataset(cfg)` from `dataset_class/__init__.py`. Set
`posts_with_video: true` in the config to additionally use video posts during VLM analysis.

---

## Inference (`main.py`)

`main.py` is the single entry point for open-weight local (vLLM) inference. It
dispatches on the paradigm — `prompt.type` in the config (`instruction_driven` /
`example_driven`), overridable with `-pt/--prompt_type`:

| `prompt.type` | Runner | Paradigm |
|---|---|---|
| `instruction_driven` | `runners/instruction_driven.py` | Instruction-driven: policy only, no examples |
| `example_driven`  | `runners/example_driven.py`   | Example-driven: in-context demonstrations |

### Instruction-driven

All settings live in a YAML config. Use `configs/inference_instruction_driven.yaml` as the base.

CLI args override YAML values. Run `python main.py --help` for the full list.

Results are stored as JSONL in `{output_dir}/`.

**Example run** (`random` subset). Output directory should be ensured per subset (`outputs_instruction-driven/<subset>/`) so provided analysis code can find them; swap `random` in both `-ds` and `-o` for the other subsets:

```bash
python main.py -c configs/inference_instruction_driven.yaml \
  -mn gemma4_thinking -ms 31B -p with_labels_rationale_details \
  -ds data_files/random_metadata.jsonl \
  -o outputs_instruction-driven/random -d cuda:0 \
  -bs 50 -mlen 32768 -mnt 8192 -gmem 0.95 -nf 16
```

### Example-driven

The `example_driven` paradigm (`prompt.type: example_driven`, dispatched to
`runners/example_driven.py`): in-context learning on top of the same pipeline, with
three example-selection strategies — `random`, `prototypical`, `contextual`.
Each builds one flat prompt per query post by default (a single model call),
with up to `in_context_num` examples **per label** plus a safe/contrast pool (default: 10).
The grouped/multi-call mode (`example_per_group`, below) splits those examples
across several smaller calls instead. This is essential for open-weight models with limited context.

**Data prerequisites.** The pools and test sets under `data_files/example_driven/` ship
as uri-only seeds and must be hydrated before use:

- `data_files/example_driven/example_pools/` (random / prototypical demos) — run
  `scripts/enrich_example_pools.sh` (~200 posts).
- `data_files/example_driven/contextual_test_sets/` (the `-nb` neighbour files for
  `contextual`) — run `scripts/enrich_contextual_test_sets.sh [concurrency]` (all
  four; hours each — ~90k posts per set once neighbours are included — resumable).
  For one set only: `utils/enrich_contextual_test_set.py <seed_uri.jsonl> <out.jsonl>`.
  The **query** set (`-ds`) is just the ordinary `data_files/<subset>_metadata.jsonl`.

See the `README.md` in each of those folders for the seed/row schema.

**Selection strategies.** The file that provides the *demonstrations* differs:

- **`random`** / **`prototypical`** — a fixed pool, the same set for every query
  post, loaded from
  `data_files/example_driven/example_pools/examples_{random,prototypical}(_safe).jsonl`.
  **Note**: These paths are hardcoded in `utils/example_driven.py` (`_EXAMPLE_POOL_DIR`),
  not configurable via YAML/CLI. If any changes are manually made during data download, *edit that constant*.
- **`contextual`** — per-query semantically nearest neighbours from a **required file**
  (`-nb` / `dataset.neighbors_path`): a JSONL of `{uri, neighbors_per_label}`
  (nearest-neighbour examples per label with a `cosine_similarity` score), joined
  to the query set by `uri`. The safe/contrast portion comes from
  `-sp` / `dataset.safe_path` (falling back to `examples_prototypical_safe.jsonl`).

**Grouped / multi-call mode (`example_per_group`).** For open models, set
`prompt.example_per_group: true` (or `-epg true`) to split a query's examples
into `in_context_num`-sized chunks and run one model call per chunk:
`contextual` pools this item's neighbours across *all* unsafe labels and
ranks them globally; `random` / `prototypical` chunk the static per-label pool.
The final label is the majority vote across chunks, falling back to the first
chunk's own prediction on a tie. 

**Config.** Use `configs/inference_example_driven.yaml`. Beyond the instruction-driven CLI it
adds `-ics`/`--in_context_select`, `-icn`/`--in_context_num`, `-sp`/`--safe_path`,
`-nb`/`--neighbors_path`, `-epg`/`--example_per_group`, `-gbs`/`--group_batch_size`.

Example runs on the `random` subset (swap `random` in `-ds` / `-nb` / `-o` for other subsets). Output goes to `outputs_example-driven/<subset>/` so `analysis.report -P example_driven` finds it.

All three use the **grouped / majority-vote mode** (`-epg true`) — the recommended default for open models: it splits the demonstrations into `in_context_num`-sized chunks and runs one call per chunk (`-gbs` chunks per vLLM call), so no single prompt has to hold ~100 mixed-label examples. It's ~`in_context_num / group_batch_size`× more calls, so batch size is lower. Drop `-epg true -gbs 5` (and raise `-bs`) for the single-mega-prompt variant.

```bash
# contextual: needs -nb (the enriched neighbour file), joined to -ds by uri
python main.py -c configs/inference_example_driven.yaml -mn qwen3_5 -ms 27B \
  -pt example_driven -p with_labels \
  -ds data_files/random_metadata.jsonl \
  -nb data_files/example_driven/contextual_test_sets/random_top10_per_label_flattened.jsonl \
  -sp "" -o outputs_example-driven/random -d cuda:0 \
  -mlen 81920 -tps 1 -bs 4 -mnt 2048 -gmem 0.85 -nf 16 -icn 10 -ics contextual -epg true -gbs 5

# random: demos are the static random pool -- no -nb
python main.py -c configs/inference_example_driven.yaml -mn qwen3_5 -ms 27B \
  -pt example_driven -p with_labels \
  -ds data_files/random_metadata.jsonl \
  -sp "" -o outputs_example-driven/random -d cuda:0 \
  -mlen 81920 -tps 1 -bs 4 -mnt 2048 -gmem 0.85 -nf 16 -icn 10 -ics random -epg true -gbs 5

# prototypical: demos are the static prototypical pool -- no -nb
python main.py -c configs/inference_example_driven.yaml -mn qwen3_5 -ms 27B \
  -pt example_driven -p with_labels \
  -ds data_files/random_metadata.jsonl \
  -sp "" -o outputs_example-driven/random -d cuda:0 \
  -mlen 81920 -tps 1 -bs 4 -mnt 2048 -gmem 0.85 -nf 16 -icn 10 -ics prototypical -epg true -gbs 5
```

---

## AI Safety Models
Analyze specialized fine-tuned AI safety models in the instruction-driven paradigm.

### Llama Guard (`llama_guard.py`)

Runs Meta's Llama Guard 4 (12B) with either its built-in safety taxonomy or the custom Bluesky policy. Results can be written into `outputs_instruction-driven/<subset>/` so `analysis.report` (default `-P instruction_driven`) scores it alongside other instruction-driven VLM runs.

```bash
# Llama Guard's own built-in taxonomy   (prompt_mode "own")
python llama_guard.py --dataset data_files/random_metadata.jsonl \
  --policy own --output_dir outputs_instruction-driven/random -bs 10

# Bluesky custom policy, injected via chat_template_kwargs   (prompt_mode "with_details")
python llama_guard.py --dataset data_files/random_metadata.jsonl \
  --policy bluesky --output_dir outputs_instruction-driven/random -bs 10
```

The Bluesky policy is loaded from `prompt/policies/bluesky/with_labels_details.md` and injected directly into the Llama Guard chat template. Run `python llama_guard.py --help` for all flags.

---

### Shieldstral (`shieldstral.py`)

Runs Mistral's Shieldstral-1.0-3B model with the Bluesky policy. Each post is evaluated against all unsafe categories (S1–S10) with a binary yes/no query per category, batched as a single vLLM call. The predicted label is the highest-scoring flagged category, or S0 if none exceed the threshold.

> **Note:** Shieldstral requires a newer vLLM than the main environment. Set up a separate venv:
> ```bash
> uv venv shieldstral
> source shieldstral/bin/activate
> uv pip install vllm --upgrade
> ```

```bash
python shieldstral.py --dataset data_files/random_metadata.jsonl \
  -p with_labels_rationale_details \
  --output_dir outputs_instruction-driven/random -bs 50
```

Run `python shieldstral.py --help` for all flags.

---

## Frontier API Models (`run_api.py`)

Runs the Bluesky policies against commercial LLM APIs, e.g., GPT and Gemini.
Both paradigms, chosen with `-pt/--prompt_type` (same as `main.py`):
`instruction_driven` (default) or `example_driven`.

> **Note:** Provider SDKs aren't in `requirements.txt`. Install whichever ones you need into the existing venv:
> ```bash
> uv pip install openai         # for gpt-* models
> uv pip install google-genai   # for gemini-* models
> ```

Set exactly one of these environment variables, matching `--model`'s prefix:

```bash
export OPENAI_API_KEY=sk-...          # for gpt-* models
export GOOGLE_API_KEY=...             # for gemini-* models
```

### Instruction-driven

```bash
python run_api.py --model gpt-5.6-terra --dataset data_files/moderated_metadata.jsonl
python run_api.py --model gpt-5.6-terra --dataset data_files/safe_metadata.jsonl --limit 20
python run_api.py --model gemini-3.5-flash --dataset data_files/random_metadata.jsonl --output outputs_instruction-driven/random/gemini_random.jsonl
```

`--prompt_mode` selects the same policy modes as `main.py` (default `with_labels_rationale_details`, see Prompt Modes above). Results are written to `{output_dir}/bluesky__{model}__{prompt_mode}__conf-true.jsonl` (default `output_dir`: `outputs_instruction-driven/`) unless `--output` is given.

### Example-driven

In-context demonstrations on top of the same pipeline, selected the same way as `main.py`'s example-driven runner (`-ics random` / `prototypical` static pools, or `-ics contextual` per-query nearest neighbours). **API example-driven is flat only** — every demonstration goes in a single call, there is no `-epg` grouped / majority-vote mode. Auto output name is `bluesky__{model}__{in_context_select}__{prompt_mode}__conf-true.jsonl` (default `output_dir`: `outputs_example-driven/`).

```bash
# static random pool
python run_api.py --model gpt-5.6-terra --dataset data_files/random_metadata.jsonl \
  -pt example_driven -ics random -icn 10

# static prototypical pool
python run_api.py --model gpt-5.6-terra --dataset data_files/random_metadata.jsonl \
  -pt example_driven -ics prototypical -icn 10

# contextual: needs -nb (the enriched neighbour file), joined to --dataset by uri
python run_api.py --model gemini-3.5-flash --dataset data_files/random_metadata.jsonl \
  -pt example_driven -ics contextual -icn 10 \
  -nb data_files/example_driven/contextual_test_sets/random_top10_per_label_flattened.jsonl
```

Beyond the instruction-driven flags it adds `-ics`/`--in_context_select`, `-icn`/`--in_context_num`, `-nb`/`--neighbors_path` (required for `contextual`), `-sp`/`--safe_path`, and `--no_safe_examples`. A flat example-driven prompt carries ~`in_context_num` × (number of unsafe labels) demonstrations plus a safe pool, each with its images, in one request — mind provider request-size / token limits and lower `-icn` to shrink it.

To feed either paradigm's results into `analysis/report.py`, point `--output_dir` at (or move the output file into) one of the directories listed in `analysis/subset_dirs.json` for the matching subset -- same as any other model's results.

Run `python run_api.py --help` for all flags, including resume (`--overwrite`, `--retry_refusals`), sampling (`--cids`, `--cids_file`, `--limit`), and provider-specific reasoning controls (`--thinking_budget`, `--reasoning_effort`, `--gemini_thinking_level`).

---

## Justification Inference (`get_justification_posthoc.py`)

Given a results JSONL from a previous instruction-driven run, this script asks the same open-weight model to provide a policy-grounded justification for each prediction — as a multi-turn conversation (original prompt → model answer → justification request).

```bash
python get_justification_posthoc.py \
  --results_file outputs_instruction-driven/random/bluesky__gemma4_thinking-31B__with_labels_rationale_details__conf-True.jsonl \
  --output_dir outputs_instruction-driven/justifications/ \
  --model_name gemma4_thinking \
  --data_file data_files/random_metadata.jsonl
```

Output adds `policy_quotes`, `explanation`, and `justification_raw` fields per row. Run `python get_justification_posthoc.py --help` for all flags.

---

## Analysis (`analysis/`)

Once you have result JSONLs from `main.py` (and optionally `llama_guard.py`/`shieldstral.py`), `analysis/` loads them, joins them against human-annotated ground truth, and computes a Safe/Unsafe classification report and pairwise inter-rater (Gwet AC1) agreement.

**Annotator labels** are taken from `data_files/<subset>_uri.jsonl` for each of the four subsets (`moderated`, `near-moderated`, `random`, `safe`). **Configure which result paths map to which data subset** via `analysis/subset_dirs.json`. **A value can be a single directory, or a list of directories if a subset's results are scattered across more than one location**. For example, if the instruction-guided results of standard VLMs write to `outputs_instruction-driven/safe/` but a Llama Guard run for the same subset landed in `outputs_other_place/safe/`:

```json
{
  "moderated": "outputs_instruction-driven/moderated",
  "near-moderated": "outputs_instruction-driven/near-moderated",
  "random": "outputs_instruction-driven/random",
  "safe": ["outputs_instruction-driven/safe", "outputs_other_place/safe"]
}
```

One entry point, both paradigms — `--paradigm` / `-P`:

```bash
python -m analysis.report                                # instruction-driven (default)
python -m analysis.report -P example_driven              # example-driven
python -m analysis.report -P example_driven --subset-dirs <(echo '{"moderated":"outputs/my_run"}')
python -m analysis.report --plot-gwet
```

`--paradigm` picks the default `{subset: dir}` layout and the report's grouping:

| `--paradigm` | default dirs | grouped by |
|---|---|---|
| `instruction_driven` | `analysis/subset_dirs.json` (`outputs_instruction-driven/<subset>/`) | subset × model × prompt_mode |
| `example_driven` | `outputs_example-driven/<subset>/` | + in-context-select × example_per_group × use_safe_examples |

`--subset-dirs` is a JSON `{subset: dir}` merged over that default (only the listed subsets are overridden). Ground truth is the same `data_files/<subset>_uri.jsonl` join for both. Output: `analysis/reports/classification_report_{instruction,example}.csv` plus (with `--plot-gwet`) `gwet__<...>.png`.


For interactive use (in notebooks):

```python
from analysis.results import load_results
from analysis.metrics import binary_classification_report, pairwise_gwet_matrix, plot_gwet_heatmap

df = load_results()                                  # instruction-driven default
# df = load_results(paradigm="example_driven")       # example-driven (outputs_example-driven/<subset>/)
# df = load_results({"moderated": "outputs/my_run"}, paradigm="example_driven")
from analysis.results import GROUP_COLS_BY_PARADIGM
report = binary_classification_report(df, group_cols=GROUP_COLS_BY_PARADIGM["instruction_driven"])
```

---

## Citation

If you use ModerationBench, please cite our paper:

```bibtex
@misc{majumdar2026moderation,
  title         = {Can Foundation Models Moderate Online Content? Evaluating Instruction- vs. Example-Driven Policy Operationalization},
  author        = {Ayan Majumdar and Shounak Paul and Pushpdeep Singh and Ines Abdelaziz and Sayeh Jarollahi and Seungeon Lee and Krishna P. Gummadi and Ingmar Weber and Abhisek Dash},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

---

## Contact

For any queries and access request for pre-fetched data, contact `ayanm[at]protonmail.com` or `psingh[at]mpi-sws.org`.

---

**Logo attribution**

<a href="https://www.flaticon.com/free-icons/protection" title="protection icons">Protection icons created by Magnific - Flaticon</a>