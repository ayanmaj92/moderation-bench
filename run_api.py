#!/usr/bin/env python3
"""
run_api.py

Content-moderation runner using commercial LLM APIs (GPT, Gemini), adapted to
this repository's dataset/prompt/output conventions so its results are directly
comparable to main.py's vLLM-based runs and drop straight into
analysis/report.py.

Runs both paradigms, chosen with --prompt_type / -pt (same as main.py):
  * instruction_driven (default) -- the model gets only the written policy.
  * example_driven               -- in-context demonstrations (random /
    prototypical static pools, or contextual per-query nearest neighbours via
    -nb). Sent FLAT: one API call per row with every demonstration in a single
    prompt (no grouped / majority-vote mode -- that's local-vLLM only).

- Dataset loading uses dataset_class.load_dataset() / BlueskyDataset --
  the same loader main.py and llama_guard.py use -- so it expects
  the same *_metadata.jsonl files produced by scripts/get_data_files.sh.
- Prompts are built with prompt/instruction_driven.py's InstructionDrivenPrompt against
  prompt/policies/bluesky/<mode>.md, i.e. the exact same system prompt,
  policy text, and output-format instructions the vLLM models see.
- Response parsing uses utils/helpers.py's parse_llm_response_2, and the
  output JSONL uses the same row-schema fields as main.py /
  llama_guard.py (input_cid, input_uri, output.PREDICTED_CATEGORY_ID,
  output.CONFIDENCE_SCORE, processing_time, ...).

Set exactly one of these environment variables, matching --model's prefix:
    OPENAI_API_KEY      (for gpt-* models)
    GOOGLE_API_KEY       (for gemini-* models)

To feed results into analysis/report.py, point --output_dir at (or move the
output file into) one of the directories listed in analysis/subset_dirs.json
for the matching subset, e.g. `outputs_instruction-driven/moderated/` -- same as any
other model's results.

REFUSAL / CONTENT-BLOCK TRACKING
----------------------------------
See the module-level comments in call_gpt/call_gemini and
call_model() for exactly what each provider returns when it refuses, and
the note below for the important caveat: this only catches EXPLICIT
refusals (native signal or refusal-shaped text). A model that silently
mislabels difficult content instead of refusing will not be flagged here.
"""

import argparse
import base64
import io
import json
import mimetypes
import os
import re
import time
from functools import lru_cache

from dataset_class import load_dataset
from prompt.instruction_driven import InstructionDrivenPrompt
from prompt.example_driven import ExampleDrivenPrompt
from utils.helpers import parse_llm_response_2, read_jsonl
from utils.example_driven import (
    build_contextual_examples,
    build_random_or_prototypical_examples,
    log_media_stats_summary,
    reset_media_stats,
)


# ============================================================
# Category id <-> name mapping -- kept local to this script, same pattern
# llama_guard.py uses for its own _BSKY_CATEGORIES copy.
# ============================================================

CATEGORY_ID_TO_NAME = {
    "S0": "no-moderation", "S1": "porn", "S2": "sexual", "S3": "sexual-figurative",
    "S4": "self-harm", "S5": "nudity", "S6": "intolerant", "S7": "graphic-media",
    "S8": "rude", "S9": "threat", "S10": "other-unsafe",
}


# ============================================================
# Media loading
# ============================================================

@lru_cache(maxsize=4096)
def load_image_as_data_url(path):
    """Reads a local image file and returns it as a base64 data URL, or None
    (with a printed warning) if the file can't be read OR can't be decoded as
    an image.

    Memoized: the static random/prototypical demo pool (~100 images) is
    identical for every row, so each file is read + decoded + encoded once, and
    the resulting base64 bytes are byte-stable across rows (keeps the provider
    prompt-cache prefix stable). Contextual / query images are bounded by the
    LRU. The PIL decode-check means a single corrupt demo image is dropped here
    with a warning instead of failing the whole API request."""
    try:
        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type:
            mime_type = "image/jpeg"
        with open(path, "rb") as f:
            raw = f.read()
        from PIL import Image  # lazy, same pattern as the provider SDKs below
        Image.open(io.BytesIO(raw)).load()
        encoded = base64.b64encode(raw).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"
    except Exception as e:
        print(f"  Warning: couldn't load image '{path}': {e}")
        return None


# ============================================================
# Prompt building: runs the repo's own InstructionDrivenPrompt.build_chat_messages
# (identical wording to what the vLLM models see) and turns its
# text/image_url content blocks into provider-agnostic blocks, loading
# each referenced local image file as base64 right here. The counts let
# the caller record in the output whether every requested image actually
# made it into the request -- a failed image load (bad path, corrupt file,
# permissions) previously only printed a warning and silently proceeded
# as if the image had never been there, indistinguishable in the output
# from a normal text-only row.
# ============================================================

def build_provider_blocks(prompt, text, image_paths, example_kwargs=None):
    """Returns (blocks, images_requested, images_loaded).

    For example-driven runs, ``example_kwargs`` carries
    ``example_k_texts`` / ``example_k_images`` / ``example_k_labels``; the demo
    images ride through the same ``image_url`` -> base64 loop as the query
    images, so the two counts cover query + demo images together (not split,
    matching runners/example_driven.py)."""
    build_kwargs = {"query_text": text, "query_images": image_paths, "query_video": None}
    if example_kwargs:
        build_kwargs.update(example_kwargs)
    messages = prompt.build_chat_messages(**build_kwargs)
    user_content = messages[1]["content"]

    blocks = []
    images_requested = 0
    images_loaded = 0
    for item in user_content:
        if item["type"] == "text":
            blocks.append({"kind": "text", "text": item["text"]})
        elif item["type"] == "image_url":
            images_requested += 1
            data_url = load_image_as_data_url(item["image_url"]["url"])
            if data_url:
                blocks.append({"kind": "image", "url": data_url})
                images_loaded += 1
    return blocks, images_requested, images_loaded


def to_gpt_responses_content(blocks):
    """Responses API shape: input_text/input_image, with image_url as a
    PLAIN STRING (not a nested {"url": ...} dict like Chat Completions)."""
    content = []
    for b in blocks:
        if b["kind"] == "text":
            content.append({"type": "input_text", "text": b["text"]})
        else:
            content.append({"type": "input_image", "image_url": b["url"]})
    return content


def to_gemini_parts(blocks):
    parts = []
    for b in blocks:
        if b["kind"] == "text":
            parts.append({"text": b["text"]})
        else:
            header, b64data = b["url"].split(",", 1)
            mime = header.split(";")[0].replace("data:", "")
            parts.append({"inline_data": {"mime_type": mime, "data": b64data}})
    return parts


# ============================================================
# Refusal detection
# ============================================================

class ContentPolicyBlocked(Exception):
    """Raised when a provider's SDK call fails specifically because of a
    content/safety policy block. Not retried."""
    pass


_EXCEPTION_BLOCK_MARKERS = [
    "content_filter", "content management policy", "safety system",
    "flagged as", "blocked", "prohibited_content", "responsible ai",
    "content policy",
]

_REFUSAL_TEXT_PATTERNS = [
    r"\bI can'?t (help|assist|provide|continue)\b",
    r"\bI'?m (not able|unable) to\b",
    r"\bI won'?t (be able to )?(help|assist|provide)\b",
    r"\bI cannot (help|assist|provide|continue|comply)\b",
    r"\bI'?m sorry,? but\b",
    r"\bas an AI\b.*\bcannot\b",
]
_REFUSAL_TEXT_RE = re.compile("|".join(_REFUSAL_TEXT_PATTERNS), re.IGNORECASE)


def is_policy_block_message(msg):
    msg_lower = str(msg).lower()
    return any(marker in msg_lower for marker in _EXCEPTION_BLOCK_MARKERS)


def looks_like_refusal(text):
    """Heuristic fallback: no JSON object present AND text pattern-matches
    a refusal opening. Only used when no native provider signal fired.
    See module docstring for what this does and does NOT catch."""
    if "{" in text and "PREDICTED_CATEGORY_ID" in text.upper():
        return False
    return bool(_REFUSAL_TEXT_RE.search(text))


# ============================================================
# Provider calls
# ============================================================

def _field(obj, name, default=None):
    """Attribute access with a dict-key fallback. Response SDK objects are
    normally attribute-accessible (Pydantic-style), but if any nested item
    ever comes through as a plain dict instead (a real possibility depending
    on SDK version / code path), getattr(obj, name, None) SILENTLY returns
    None even though the data is genuinely present under obj[name] --
    indistinguishable from the field truly being absent. This closes that
    gap: try attribute access first, fall back to dict-style lookup."""
    if obj is None:
        return default
    val = getattr(obj, name, None)
    if val is not None:
        return val
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def call_gpt(client, model, system_text, content_blocks, max_tokens, reasoning_effort=None, reasoning_mode=None):
    """Uses the Responses API (client.responses.create), not Chat
    Completions -- OpenAI's own docs recommend this for reasoning models:
    it uses a single max_output_tokens parameter (no max_tokens vs
    max_completion_tokens ambiguity), gives an explicit
    status=="incomplete" / incomplete_details.reason=="max_output_tokens"
    truncation signal, and can return an actual reasoning SUMMARY (real
    text, not just a token count) via reasoning={"summary": "auto"}.

    NOTE: some of this (e.g. whether refusal content items in the output
    array look exactly like this) is based on documented Responses API
    behavior I could confirm via OpenAI's docs, but I have not been able to
    verify every field against a live response for whatever specific model
    string you're using. If any of the extraction below looks wrong against
    real output, that's the first place to check.
    """
    import openai

    input_content = to_gpt_responses_content(content_blocks)

    def _do_call(token_budget, include_reasoning_summary=True):
        call_kwargs = dict(
            model=model,
            instructions=system_text,
            input=[{"role": "user", "content": input_content}],
            max_output_tokens=token_budget,
        )
        if reasoning_effort or reasoning_mode or include_reasoning_summary:
            reasoning_cfg = {}
            if reasoning_effort:
                reasoning_cfg["effort"] = reasoning_effort
            if reasoning_mode:
                reasoning_cfg["mode"] = reasoning_mode
            if include_reasoning_summary:
                # Requests a real natural-language reasoning summary, not
                # just a token count. Requires org verification on some
                # accounts -- if that fails, the except-block below retries
                # without it, degrading gracefully to token-count-only.
                reasoning_cfg["summary"] = "auto"
            call_kwargs["reasoning"] = reasoning_cfg

        try:
            return client.responses.create(**call_kwargs)
        except openai.BadRequestError as e:
            err_msg = str(e)
            unsupported_param = getattr(e, "param", None)
            if "reasoning" in call_kwargs and (
                (unsupported_param and "reasoning" in str(unsupported_param))
                or "reasoning" in err_msg.lower()
            ):
                # Model doesn't support the reasoning param at all (e.g. a
                # non-reasoning model), or summary requires org verification
                # this account doesn't have. Retry once without it.
                call_kwargs.pop("reasoning", None)
                try:
                    return client.responses.create(**call_kwargs)
                except openai.BadRequestError as e2:
                    if is_policy_block_message(str(e2)):
                        raise ContentPolicyBlocked(str(e2))
                    raise
            elif is_policy_block_message(err_msg):
                raise ContentPolicyBlocked(err_msg)
            else:
                raise

    def _extract(response):
        thinking_parts = []
        for item in (_field(response, "output", None) or []):
            if _field(item, "type", None) == "reasoning":
                for s in (_field(item, "summary", None) or []):
                    s_text = _field(s, "text", None)
                    if s_text:
                        thinking_parts.append(s_text)

        text = _field(response, "output_text", None) or ""

        refusal_info = {"refused": False, "reason": None}
        for item in (_field(response, "output", None) or []):
            if _field(item, "type", None) == "message":
                for c in (_field(item, "content", None) or []):
                    if _field(c, "type", None) == "refusal":
                        refusal_text = _field(c, "refusal", "") or ""
                        refusal_info = {"refused": True, "reason": f"refusal: {refusal_text}"}
                        text = refusal_text or text

        status = _field(response, "status", None)
        incomplete_details = _field(response, "incomplete_details", None)
        incomplete_reason = _field(incomplete_details, "reason", None) if incomplete_details else None

        # status can also be "failed" (distinct from "incomplete"), with the
        # actual problem described in response.error rather than
        # incomplete_details. Not handling this separately previously meant
        # a failed response with empty text gave no diagnostic information
        # at all -- just an empty raw_output and no clue why.
        error_obj = _field(response, "error", None)
        error_message = None
        if status == "failed" and error_obj is not None:
            error_message = _field(error_obj, "message", None) or str(error_obj)

        if status == "failed":
            diagnostic_reason = f"status=failed: {error_message}"
        elif status == "incomplete":
            diagnostic_reason = incomplete_reason or "incomplete (no reason given)"
        else:
            diagnostic_reason = status

        if not refusal_info["refused"] and incomplete_reason == "content_filter":
            refusal_info = {"refused": True, "reason": "incomplete_details.reason=content_filter"}

        reasoning_tokens = None
        try:
            reasoning_tokens = response.usage.output_tokens_details.reasoning_tokens
        except Exception:
            usage = _field(response, "usage", None)
            details = _field(usage, "output_tokens_details", None)
            reasoning_tokens = _field(details, "reasoning_tokens", None)

        input_tokens = None
        try:
            input_tokens = response.usage.input_tokens
        except Exception:
            usage = _field(response, "usage", None)
            input_tokens = _field(usage, "input_tokens", None)

        # DIAGNOSTIC: if we have real reasoning tokens but got no summary
        # text, print the raw reasoning item so we can see definitively
        # whether the API sent an empty summary (org verification / model
        # limitation) or sent real content that our parsing still missed
        # (a remaining bug). This turns guessing into hard evidence on the
        # very next run.
        if not thinking_parts and reasoning_tokens:
            for item in (_field(response, "output", None) or []):
                if _field(item, "type", None) == "reasoning":
                    print(f"  DEBUG reasoning item raw repr: {item!r}")
                    print(f"  DEBUG reasoning item type(): {type(item)}")
                    raw_summary = _field(item, "summary", None)
                    print(f"  DEBUG summary field raw repr: {raw_summary!r}")

        return text, thinking_parts, reasoning_tokens, input_tokens, refusal_info, diagnostic_reason, status

    response = _do_call(max_tokens)
    text, thinking_parts, reasoning_tokens, input_tokens, refusal_info, diagnostic_reason, status = _extract(response)

    # IMPORTANT: OpenAI explicitly recommends reserving at least 25,000
    # tokens for reasoning + output when starting out with these models --
    # our earlier default of 100-300 was nowhere close. If the budget was
    # still too tight and nothing visible came back, retry once with a much
    # larger budget instead of silently returning an empty/unparseable
    # result with no explanation.
    #
    # Broadened to trigger on ANY non-"completed" status with empty text
    # (not just the specific incomplete_details.reason=="max_output_tokens"
    # case) -- there are other ways a response can come back empty
    # (status=="failed" with an error, an incomplete reason we don't
    # explicitly recognize, etc.) and all of them deserve the same retry +
    # clear diagnostic rather than a silent empty result.
    budget_used_for_retry = max_tokens
    if not text.strip() and status != "completed" and not refusal_info["refused"]:
        budget_used_for_retry = max(max_tokens * 10, (reasoning_tokens or 0) + 2000, 25000)
        print(f"  Note: empty response with status='{status}' ({diagnostic_reason}); "
              f"reasoning_tokens={reasoning_tokens}, budget was {max_tokens}. Retrying once "
              f"with max_output_tokens={budget_used_for_retry}.")
        response = _do_call(budget_used_for_retry)
        text, thinking_parts, reasoning_tokens, input_tokens, refusal_info, diagnostic_reason, status = _extract(response)
        if not text.strip() and status != "completed" and not refusal_info["refused"]:
            refusal_info = {
                "refused": False,
                "reason": f"empty_response ({diagnostic_reason}, budget_used={budget_used_for_retry})",
            }

    if thinking_parts:
        thinking_text = "\n\n".join(thinking_parts)
    elif reasoning_tokens:
        thinking_text = f"[reasoning summary not returned -- reasoning_tokens used: {reasoning_tokens}]"
    elif reasoning_tokens == 0:
        thinking_text = "[reasoning_tokens used: 0]"
    else:
        thinking_text = ""

    return text, thinking_text, input_tokens, refusal_info, diagnostic_reason


_GEMINI_BLOCK_FINISH_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "RECITATION", "BLOCKLIST", "SPII"}


def call_gemini(client, model, system_text, content_blocks, max_tokens, thinking_budget=None, thinking_level=None):
    """thinking_level (string: minimal/low/medium/high) is the recommended
    control for Gemini 3.x models (e.g. gemini-3.5-flash) -- the older
    numeric thinking_budget is "accepted for backwards compatibility" per
    Google's docs but not recommended for Gemini 3.x, and combining BOTH in
    one request returns a 400 error. So these stay mutually exclusive here:
    thinking_level takes priority if both are somehow passed.

    Gemini 3.5 Flash thinks by default (medium) even with neither set."""
    from google.genai import types
    from google.genai import errors as genai_errors

    parts = to_gemini_parts(content_blocks)

    def _do_call(token_budget):
        config_kwargs = dict(max_output_tokens=token_budget, system_instruction=system_text)
        # include_thoughts=True asks the API to return thought summaries as
        # separate parts (marked with part.thought = True) rather than just
        # spending the budget silently. This must be sent UNCONDITIONALLY --
        # Gemini 3.x models think by their own default effort regardless of
        # whether thinking_level/thinking_budget is explicitly set, but
        # summaries are only returned if include_thoughts is requested.
        # Previously this was nested inside the level/budget branches below,
        # so a run with neither flag set got real thinking (nonzero
        # thoughts_token_count) but never asked for the summary text at all.
        if thinking_level is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=thinking_level, include_thoughts=True
            )
        elif thinking_budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=thinking_budget, include_thoughts=True
            )
        else:
            config_kwargs["thinking_config"] = types.ThinkingConfig(include_thoughts=True)
        config = types.GenerateContentConfig(**config_kwargs)
        try:
            return client.models.generate_content(
                model=model, contents=[{"role": "user", "parts": parts}], config=config,
            )
        except genai_errors.ClientError as e:
            if is_policy_block_message(str(e)):
                raise ContentPolicyBlocked(str(e))
            raise

    def _extract(response):
        """Returns (text, thinking_text, thoughts_token_count, finish_reason,
        refusal_info) for a single response, or None-filled values if
        blocked before generation."""
        prompt_feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(prompt_feedback, "block_reason", None) if prompt_feedback else None
        if block_reason:
            return "", "", None, None, {"refused": True, "reason": f"prompt_feedback.block_reason={block_reason}"}

        refusal_info = {"refused": False, "reason": None}
        candidates = getattr(response, "candidates", None) or []
        finish_reason = str(getattr(candidates[0], "finish_reason", "")) if candidates else ""
        if any(r in finish_reason.upper() for r in _GEMINI_BLOCK_FINISH_REASONS):
            refusal_info = {"refused": True, "reason": f"finish_reason={finish_reason}"}

        # Manual part-by-part extraction (NOT the response.text quick-accessor):
        # thinking models return thought-summary parts alongside the final
        # answer, each part carrying a `thought` boolean. This is the
        # documented mechanism for GenerateContent (confirmed current for
        # Gemini 3.x too) -- separating them explicitly avoids relying on
        # response.text's undocumented behavior around whether it includes
        # thought parts or not.
        thinking_parts, answer_parts = [], []
        try:
            content_obj = getattr(candidates[0], "content", None) if candidates else None
            for part in (getattr(content_obj, "parts", None) or []):
                part_text = getattr(part, "text", None)
                if not part_text:
                    continue
                if getattr(part, "thought", False):
                    thinking_parts.append(part_text)
                else:
                    answer_parts.append(part_text)
        except Exception:
            pass

        thoughts_token_count = None
        try:
            thoughts_token_count = response.usage_metadata.thoughts_token_count
        except Exception:
            pass

        return "\n".join(answer_parts), "\n".join(thinking_parts), thoughts_token_count, finish_reason, refusal_info

    response = _do_call(max_tokens)
    input_tokens = None
    try:
        input_tokens = response.usage_metadata.prompt_token_count
    except Exception:
        pass

    text, thinking_parts_text, thoughts_token_count, finish_reason, refusal_info = _extract(response)

    # IMPORTANT: like OpenAI's reasoning models, thinking tokens count
    # against the SAME max_output_tokens budget as the visible answer. If
    # thinking consumed it all, the answer comes back empty with
    # finish_reason == "MAX_TOKENS" -- not a refusal, just no room left.
    # Detected here, this triggers ONE automatic retry with a larger budget.
    budget_used_for_retry = max_tokens
    if not text.strip() and finish_reason and "MAX_TOKENS" in finish_reason.upper() and not refusal_info["refused"]:
        budget_used_for_retry = max(max_tokens * 10, (thoughts_token_count or 0) + 2000, 8192)
        print(f"  Note: thinking consumed the entire token budget (budget was {max_tokens}), "
              f"leaving no room for the answer. Retrying once with max_output_tokens={budget_used_for_retry}.")
        response = _do_call(budget_used_for_retry)
        try:
            input_tokens = response.usage_metadata.prompt_token_count
        except Exception:
            pass
        text, thinking_parts_text, thoughts_token_count, finish_reason, refusal_info = _extract(response)
        if not text.strip() and finish_reason and "MAX_TOKENS" in finish_reason.upper():
            refusal_info = {
                "refused": False,
                "reason": f"empty_response_max_tokens_exhausted (budget_used={budget_used_for_retry})",
            }

    # Same pattern as GPT: real thought text if we got it, else a token-count
    # note (distinguishing 0 from "not reported" -- a genuine zero is still
    # informative), else genuinely nothing.
    if thinking_parts_text:
        thinking_text = thinking_parts_text
    elif thoughts_token_count is not None:
        thinking_text = f"[thought summary not returned -- thoughts_token_count: {thoughts_token_count}]"
    else:
        thinking_text = ""

    return text, thinking_text, input_tokens, refusal_info, finish_reason


def get_provider_and_client(model_name):
    name = model_name.lower()
    if name.startswith("gpt"):
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("Set OPENAI_API_KEY to use a gpt-* model.")
        return "gpt", OpenAI(api_key=api_key)
    elif name.startswith("gemini"):
        from google import genai
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit("Set GOOGLE_API_KEY to use a gemini-* model.")
        return "gemini", genai.Client(api_key=api_key)
    else:
        raise SystemExit(
            f"Can't infer a provider from model name '{model_name}'. "
            f"Expected it to start with 'gpt' or 'gemini'."
        )


def call_model(provider, client, model, system_text, content_blocks, max_tokens,
               max_retries=3, retry_backoff=2.0, thinking_budget=None, reasoning_effort=None,
               reasoning_mode=None, gemini_thinking_level=None):
    last_err = None
    for attempt in range(max_retries):
        try:
            if provider == "gpt":
                text, thinking_text, input_tokens, refusal_info, finish_reason = call_gpt(
                    client, model, system_text, content_blocks, max_tokens,
                    reasoning_effort=reasoning_effort, reasoning_mode=reasoning_mode)
            else:
                text, thinking_text, input_tokens, refusal_info, finish_reason = call_gemini(
                    client, model, system_text, content_blocks, max_tokens,
                    thinking_budget=thinking_budget, thinking_level=gemini_thinking_level)

            if not refusal_info["refused"] and looks_like_refusal(text):
                refusal_info = {"refused": True, "reason": "heuristic_text_match"}
            return text, thinking_text, input_tokens, refusal_info, finish_reason

        except ContentPolicyBlocked as e:
            print(f"  Content policy block (not retrying): {e}")
            return (
                '{"PREDICTED_CATEGORY_ID": null, "CONFIDENCE_SCORE": null}',
                "",
                None,
                {"refused": True, "reason": f"api_exception: {e}"},
                None,
            )

        except Exception as e:
            last_err = e
            wait = retry_backoff ** attempt
            print(f"  API call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait:.1f}s")
            time.sleep(wait)

    print(f"  All retries exhausted: {last_err}")
    return (
        '{"PREDICTED_CATEGORY_ID": null, "CONFIDENCE_SCORE": null}',
        "",
        None,
        {"refused": False, "reason": f"transient_error_max_retries: {last_err}"},
        None,
    )


# ============================================================
# Resume support: skip cids already processed in an existing output file
# ============================================================

def load_already_processed(output_path, retry_refusals=False):
    """Reads an existing output JSONL (if present) and returns
    (done_cids, n_transient_retry, n_refusal_retry). A row that only
    failed due to a transient API error (retries exhausted -- not a real
    answer) never counts as done, and is always retried. A refusal counts
    as done (a real, meaningful result) UNLESS retry_refusals=True, in
    which case it's also excluded so it gets one more attempt on resume --
    this is a manual, one-shot opt-in per invocation, not an automatic
    endless-retry loop; if it refuses again, that's the final answer."""
    done_cids = set()
    n_transient_retry = 0
    n_refusal_retry = 0
    if not os.path.exists(output_path):
        return done_cids, n_transient_retry, n_refusal_retry
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            reason = row.get("refusal_reason") or ""
            if reason.startswith("transient_error_max_retries"):
                n_transient_retry += 1
                continue  # not a real result -- always retried
            if retry_refusals and row.get("model_refused"):
                n_refusal_retry += 1
                continue  # explicitly opted into retrying refusals too
            cid = row.get("input_cid")
            if cid:
                done_cids.add(cid)
    return done_cids, n_transient_retry, n_refusal_retry


# ============================================================
# Main
# ============================================================

def _rows_from_dataset(dataset):
    """Flattens the column-oriented `dataset.data` dict (as produced by
    BlueskyDataset.load_all(), same as main.py/llama_guard.py consume) into a
    list of per-post row dicts."""
    d = dataset.data
    n = len(d["text"])
    rows = []
    for i in range(n):
        rows.append({
            "cid": d["cid"][i],
            "uri": d["uri"][i],
            "text": d["text"][i],
            "images": d["images"][i],
            "video": d["video"][i],
            "platform_label": d["platform_label"][i],
            "annotator_label": d["annotator_label"][i],
            # consolidated test sets carry the human label under `label`, not
            # `annotator_label` -- the example-driven path falls back to it.
            "label": d["label"][i],
            "is_reply": d["is_reply"][i],
            "embed_type": d["embed_type"][i],
        })
    return rows


def _load_and_filter_rows(input_path, output_path, args):
    """Shared pipeline front-half: make the output dir, load the dataset, flatten
    to per-post rows, print the video note, and apply --cids / --cids_file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cfg = {
        "dataset": {
            "name": "bluesky_final",
            "test_path": input_path,
            "posts_with_video": args.posts_with_video,
        },
    }
    dataset = load_dataset(cfg)
    rows = _rows_from_dataset(dataset)
    print(f"Loaded {len(rows)} rows from {input_path}.")

    n_with_video = sum(1 for r in rows if r["video"])
    if n_with_video:
        print(f"Note: {n_with_video} / {len(rows)} row(s) have a video attached. Video content itself is "
              f"NOT sent to any API provider by this script (image/text only) -- those rows are still "
              f"processed using whatever text/images they have.")

    # --- cid filtering (specific sample mode) ---
    target_cids = set()
    if args.cids:
        target_cids.update(c.strip() for c in args.cids.split(",") if c.strip())
    if args.cids_file:
        with open(args.cids_file, "r", encoding="utf-8") as f:
            target_cids.update(line.strip() for line in f if line.strip())
    if target_cids:
        found_cids = {r["cid"] for r in rows}
        missing = target_cids - found_cids
        if missing:
            preview = sorted(missing)[:10]
            print(f"Warning: {len(missing)} requested cid(s) not found in dataset: {preview}"
                  f"{' ...' if len(missing) > 10 else ''}")
        rows = [r for r in rows if r["cid"] in target_cids]
        print(f"Filtered to {len(rows)} row(s) matching the requested cids.")

    return rows


def _apply_resume(rows, output_path, args):
    """Shared pipeline front-half (cont.): on resume, drop cids already processed
    in an existing output file, then truncate to --limit. Returns
    (rows, file_mode)."""
    resuming = os.path.exists(output_path) and not args.overwrite
    if resuming:
        already_done, n_transient_retry, n_refusal_retry = load_already_processed(
            output_path, retry_refusals=args.retry_refusals
        )
        if already_done or n_transient_retry or n_refusal_retry:
            before = len(rows)
            rows = [r for r in rows if r["cid"] not in already_done]
            print(f"Resuming: found {len(already_done)} already-processed cid(s) in {output_path}. "
                  f"Skipping {before - len(rows)} row(s); {len(rows)} remaining.")
            if n_transient_retry:
                print(f"  {n_transient_retry} row(s) will be retried (previously failed with a transient error).")
            if n_refusal_retry:
                print(f"  {n_refusal_retry} row(s) will be retried (previously refused, --retry_refusals is set). "
                      f"If they refuse again this time, that's the final answer -- there's no automatic "
                      f"third attempt.")
            if n_transient_retry or n_refusal_retry:
                print("Note: retried rows' old entries stay in the file alongside the new ones -- "
                      "when analyzing, keep only the LAST entry per cid.")

    if args.limit:
        rows = rows[:args.limit]

    return rows, ("a" if resuming else "w")


def run_pipeline(provider, client, model, prompt, input_path, output_path, args):
    rows = _load_and_filter_rows(input_path, output_path, args)
    rows, file_mode = _apply_resume(rows, output_path, args)

    n_refused = 0
    n_images_requested_total = 0
    n_images_loaded_total = 0
    total_processing_time = 0.0
    with open(output_path, file_mode, encoding="utf-8") as outfile:
        for i, row in enumerate(rows):
            content_blocks, images_requested, images_loaded = build_provider_blocks(
                prompt, row["text"], row["images"]
            )

            start = time.time()
            raw_text, thinking_text, input_tokens, refusal_info, finish_reason = call_model(
                provider, client, model, prompt.system_text, content_blocks, args.max_tokens,
                thinking_budget=args.thinking_budget, reasoning_effort=args.reasoning_effort,
                reasoning_mode=args.reasoning_mode,
                gemini_thinking_level=args.gemini_thinking_level,
            )
            elapsed = time.time() - start

            if refusal_info["refused"]:
                n_refused += 1
            n_images_requested_total += images_requested
            n_images_loaded_total += images_loaded
            total_processing_time += elapsed

            parsed = parse_llm_response_2(raw_text, confidence=True, two_stage=False, severity=False) or {}

            escaped_raw = raw_text.replace("\n", "\\n")
            escaped_thinking = thinking_text.replace("\n", "\\n") if thinking_text else ""

            predicted_cat = parsed.get("PREDICTED_CATEGORY_ID")
            result = {
                "input_id": i,
                "input_cid": row["cid"],
                "input_uri": row["uri"],
                "input_is_reply": row["is_reply"],
                "input_etype": row["embed_type"],
                "text": row["text"],
                "image_paths": row["images"],
                "video_path": row["video"],
                "platform_label": row["platform_label"],
                "annotator_label": row["annotator_label"],
                "model": model,
                "prompt_type": "instruction_driven",
                "prompt_mode": args.prompt_mode,
                "images_requested": images_requested,
                "images_loaded": images_loaded,
                "output": {
                    "REASONING": parsed.get("REASONING"),
                    "PREDICTED_CATEGORY_ID": predicted_cat,
                    "PREDICTED_CATEGORY_NAME": CATEGORY_ID_TO_NAME.get(predicted_cat) if predicted_cat else None,
                    "CONFIDENCE_SCORE": parsed.get("CONFIDENCE_SCORE"),
                    "raw": escaped_raw,
                },
                "thinking_output": escaped_thinking,
                "finish_reason": finish_reason,
                "model_refused": refusal_info["refused"],
                "refusal_reason": refusal_info["reason"],
                "input_tokens": input_tokens,
                "processing_time": round(elapsed, 3),
            }
            outfile.write(json.dumps(result) + "\n")
            outfile.flush()

            flag = " [REFUSED]" if refusal_info["refused"] else ""
            print(f"[{i + 1}/{len(rows)}] cid={row['cid']} -> {predicted_cat} "
                  f"(annotator={row['annotator_label']}){flag}")

    print(f"\nDone. Results written to: {output_path}")
    if n_images_requested_total:
        print(f"Images: {n_images_loaded_total} / {n_images_requested_total} loaded successfully "
              f"and sent to the model.")
        if n_images_loaded_total < n_images_requested_total:
            print(f"WARNING: {n_images_requested_total - n_images_loaded_total} image(s) failed to load "
                  f"(see 'Warning: couldn't load image...' lines above for which ones) -- those rows "
                  f"were sent to the model as text-only, not with the image the label may depend on. "
                  f"Check images_requested/images_loaded in the output JSONL to find the affected rows.")
    if rows:
        print(f"Refused: {n_refused} / {len(rows)} ({100 * n_refused / len(rows):.1f}%)")
        print(f"Total processing time (sum of per-row processing_time): "
              f"{total_processing_time:.1f}s ({total_processing_time / 60:.2f} min)")
        print(f"Average time per sample: {total_processing_time / len(rows):.2f}s")

    return {
        "n_rows": len(rows),
        "n_refused": n_refused,
        "n_images_requested": n_images_requested_total,
        "n_images_loaded": n_images_loaded_total,
        "total_processing_time": total_processing_time,
    }


def run_pipeline_example(provider, client, model, prompt, input_path, output_path, args):
    """Example-driven counterpart of run_pipeline: one API call per row, the whole
    demonstration set sent flat in a single prompt (no grouped / majority-vote
    mode -- that's local-vLLM only). Selection reuses utils/example_driven.py, the
    same builders runners/example_driven.py uses."""
    select = args.in_context_select
    if select == "contextual" and not args.neighbors_path:
        raise SystemExit(
            "--in_context_select contextual requires -nb/--neighbors_path -- a JSONL of "
            "{uri, neighbors_per_label} (e.g. data_files/example_driven/contextual_test_sets/"
            "<subset>_top10_per_label_flattened.jsonl), joined to --dataset by uri."
        )

    rows = _load_and_filter_rows(input_path, output_path, args)
    reset_media_stats()

    # -ics contextual: per-query nearest neighbours come from a side file (-nb),
    # joined to --dataset by uri -- same construction as runners/example_driven.py.
    neighbors_index = None
    if select == "contextual":
        neighbors_index = {
            r["uri"]: (r.get("neighbors_per_label") or {})
            for r in read_jsonl(args.neighbors_path) if r.get("uri")
        }
        print(f"Loaded neighbours for {len(neighbors_index)} query uris from {args.neighbors_path}")

    # Static random/prototypical pool: identical for every row, built once.
    static_flat = None
    if select in ("random", "prototypical"):
        static_flat = build_random_or_prototypical_examples(select, args.in_context_num)

    rows, file_mode = _apply_resume(rows, output_path, args)

    use_safe_examples = not args.no_safe_examples
    n_refused = 0
    n_images_requested_total = 0
    n_images_loaded_total = 0
    total_processing_time = 0.0
    with open(output_path, file_mode, encoding="utf-8") as outfile:
        for i, row in enumerate(rows):
            if static_flat is not None:
                ex_t, ex_i, ex_l = static_flat
            else:
                ex_t, ex_i, ex_l = build_contextual_examples(
                    neighbors_index.get(row["uri"], {}), args.safe_path or "", args.in_context_num
                )
            example_kwargs = {
                "example_k_texts": ex_t,
                "example_k_images": ex_i,
                "example_k_labels": ex_l,
            }

            content_blocks, images_requested, images_loaded = build_provider_blocks(
                prompt, row["text"], row["images"], example_kwargs=example_kwargs
            )

            start = time.time()
            raw_text, thinking_text, input_tokens, refusal_info, finish_reason = call_model(
                provider, client, model, prompt.system_text, content_blocks, args.max_tokens,
                thinking_budget=args.thinking_budget, reasoning_effort=args.reasoning_effort,
                reasoning_mode=args.reasoning_mode,
                gemini_thinking_level=args.gemini_thinking_level,
            )
            elapsed = time.time() - start

            if refusal_info["refused"]:
                n_refused += 1
            n_images_requested_total += images_requested
            n_images_loaded_total += images_loaded
            total_processing_time += elapsed

            parsed = parse_llm_response_2(raw_text, confidence=True, two_stage=False, severity=False) or {}

            escaped_raw = raw_text.replace("\n", "\\n")
            escaped_thinking = thinking_text.replace("\n", "\\n") if thinking_text else ""

            predicted_cat = parsed.get("PREDICTED_CATEGORY_ID")
            result = {
                "input_id": i,
                "input_cid": row["cid"],
                "input_uri": row["uri"],
                "input_is_reply": row["is_reply"],
                "input_etype": row["embed_type"],
                "text": row["text"],
                "image_paths": row["images"],
                "video_path": row["video"],
                "platform_label": row["platform_label"],
                # consolidated test sets carry the human label under `label`.
                "annotator_label": row["annotator_label"] or row["label"],
                "model": model,
                "prompt_type": "example_driven",
                "prompt_mode": args.prompt_mode,
                "incontext_select": select,
                "use_safe_examples": use_safe_examples,
                "incontext_num": len(ex_l),
                "images_requested": images_requested,
                "images_loaded": images_loaded,
                "output": {
                    "REASONING": parsed.get("REASONING"),
                    "PREDICTED_CATEGORY_ID": predicted_cat,
                    "PREDICTED_CATEGORY_NAME": CATEGORY_ID_TO_NAME.get(predicted_cat) if predicted_cat else None,
                    "CONFIDENCE_SCORE": parsed.get("CONFIDENCE_SCORE"),
                    "raw": escaped_raw,
                },
                "thinking_output": escaped_thinking,
                "finish_reason": finish_reason,
                "model_refused": refusal_info["refused"],
                "refusal_reason": refusal_info["reason"],
                "input_tokens": input_tokens,
                "processing_time": round(elapsed, 3),
            }
            outfile.write(json.dumps(result) + "\n")
            outfile.flush()

            flag = " [REFUSED]" if refusal_info["refused"] else ""
            print(f"[{i + 1}/{len(rows)}] cid={row['cid']} -> {predicted_cat} "
                  f"(annotator={result['annotator_label']}){flag}")

    print(f"\nDone. Results written to: {output_path}")
    log_media_stats_summary()
    if n_images_requested_total:
        print(f"Images: {n_images_loaded_total} / {n_images_requested_total} loaded successfully "
              f"and sent to the model (query + demonstration images combined).")
        if n_images_loaded_total < n_images_requested_total:
            print(f"WARNING: {n_images_requested_total - n_images_loaded_total} image(s) failed to load "
                  f"(see 'Warning: couldn't load image...' lines above for which ones) -- the affected "
                  f"rows were sent with fewer images than intended. Check images_requested/images_loaded "
                  f"in the output JSONL.")
    if rows:
        print(f"Refused: {n_refused} / {len(rows)} ({100 * n_refused / len(rows):.1f}%)")
        print(f"Total processing time (sum of per-row processing_time): "
              f"{total_processing_time:.1f}s ({total_processing_time / 60:.2f} min)")
        print(f"Average time per sample: {total_processing_time / len(rows):.2f}s")

    return {
        "n_rows": len(rows),
        "n_refused": n_refused,
        "n_images_requested": n_images_requested_total,
        "n_images_loaded": n_images_loaded_total,
        "total_processing_time": total_processing_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Run commercial LLM content moderation over a Bluesky metadata JSONL.")
    parser.add_argument("--model", "-m", required=True, help="Exact model string, e.g. gpt-4o, gemini-2.5-pro.")
    parser.add_argument("--dataset", "-ds", required=True, help="Path to a *_metadata.jsonl file (e.g. data_files/moderated_metadata.jsonl), same format dataset_class.load_dataset() expects.")
    parser.add_argument("--prompt_mode", "-p", default="with_labels_rationale_details",
                        choices=["with_labels", "with_labels_details", "with_labels_rationale", "with_labels_rationale_details"],
                        help="Policy file stem under prompt/policies/bluesky/ -- same modes main.py uses (default matches configs/inference_instruction_driven.yaml).")
    parser.add_argument("--prompt_type", "-pt", default="instruction_driven",
                        choices=["instruction_driven", "example_driven"],
                        help="Moderation paradigm: 'instruction_driven' (policy only) or 'example_driven' (in-context demonstrations). Mirrors main.py's -pt. Default: instruction_driven.")
    parser.add_argument("--in_context_select", "-ics", default="random",
                        choices=["random", "prototypical", "contextual"],
                        help="Example-driven only: demonstration source -- 'random'/'prototypical' static pools under data_files/example_driven/example_pools/, or 'contextual' per-query nearest neighbours (needs -nb).")
    parser.add_argument("--in_context_num", "-icn", type=int, default=10,
                        help="Example-driven only: cap on demonstrations PER LABEL, plus up to this many safe examples (default: 10).")
    parser.add_argument("--neighbors_path", "-nb", default=None,
                        help="Example-driven only: JSONL of {uri, neighbors_per_label}, joined to --dataset by uri. REQUIRED for -ics contextual; ignored otherwise.")
    parser.add_argument("--safe_path", "-sp", default=None,
                        help="Example-driven only: override the safe/contrast example pool (JSONL). Falls back to data_files/example_driven/example_pools/examples_prototypical_safe.jsonl.")
    parser.add_argument("--no_safe_examples", action="store_true",
                        help="Example-driven only: drop safe (S0) demonstrations and add the 'still predict S0 if the query is safe' reminder. Default: safe examples are included.")
    parser.add_argument("--output_dir", "-o", default=None, help="Directory for the auto-computed output path (default: outputs_instruction-driven/ or outputs_example-driven/ by --prompt_type). To feed analysis/report.py, point this at (or move results into) a directory listed in analysis/subset_dirs.json.")
    parser.add_argument("--output", default=None, help="Override the auto-computed output path. Rarely needed.")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Only process the first N rows after any --cids filtering (useful for a quick test run).")
    parser.add_argument("--cids", default=None, help="Comma-separated list of cids to run on (runs ONLY these).")
    parser.add_argument("--cids_file", default=None, help="Path to a text file with one cid per line, to run ONLY these.")
    parser.add_argument("--posts_with_video", "-pv", action="store_true", help="Include posts with video in the dataset instead of skipping them (same flag name as llama_guard.py). Video content itself is still not sent to the API -- only the row is no longer skipped.")
    parser.add_argument("--max_tokens", type=int, default=2000, help="Max tokens per response (default: 2000). For GPT reasoning models, OpenAI recommends reserving at least 25000 for reasoning+output when starting out -- pass a higher value directly if you'd rather avoid the auto-retry. Not all values are meaningful for every provider/model.")
    parser.add_argument("--overwrite", action="store_true", help="Ignore any existing output file and start fresh instead of resuming.")
    parser.add_argument("--retry_refusals", action="store_true", help="On resume, ALSO retry rows that were previously refused (not just transient API failures). Off by default -- a refusal is treated as a final, meaningful result. This is a manual one-shot opt-in: if a row refuses again on this retry, that's the final answer, there's no automatic further retry.")
    parser.add_argument("--thinking_budget", type=int, default=None, help="Enable extended thinking with this numeric token budget (legacy Gemini 2.5 models only). NOT recommended for Gemini 3.x models (e.g. gemini-3.5-flash) -- use --gemini_thinking_level instead; combining both for a Gemini 3.x model returns a 400 error, so thinking_level takes priority if both are set.")
    parser.add_argument("--gemini_thinking_level", default=None, choices=["minimal", "low", "medium", "high"], help="Thinking level for Gemini 3.x models (e.g. gemini-3.5-flash) -- the recommended control, replacing the legacy numeric --thinking_budget. If unset, Gemini 3.5 Flash defaults to 'medium' on its own. 'minimal' does not guarantee thinking is fully off.")
    parser.add_argument("--reasoning_effort", default=None, choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"], help="Reasoning effort for GPT reasoning models via the Responses API (supported values are model-dependent -- 'max' is GPT-5.6-only). If unset, the model uses its own default effort (medium, for GPT-5.6). This is a guide, not a hard floor -- adaptive reasoning can still use ~0 tokens on trivial inputs even at high effort. A real reasoning summary is requested automatically when supported -- see thinking_output in the results.")
    parser.add_argument("--reasoning_mode", default=None, choices=["standard", "pro"], help="GPT-5.6 pro mode: applies more model work before returning a single final answer. Higher latency/token usage than effort alone. Only meaningful for GPT-5.6 models via the Responses API.")
    args = parser.parse_args()

    provider, client = get_provider_and_client(args.model)

    example_driven = args.prompt_type == "example_driven"
    if example_driven:
        prompt = ExampleDrivenPrompt(mode=args.prompt_mode, platform="bluesky",
                                     use_safe_examples=not args.no_safe_examples)
        default_dir = "outputs_example-driven/"
        auto_name = (f"bluesky__{args.model}__{args.in_context_select}"
                     f"__{args.prompt_mode}__conf-true.jsonl")
        print(f"Using example-driven prompt: select={args.in_context_select} "
              f"num={args.in_context_num} use_safe_examples={not args.no_safe_examples} "
              f"(policy: prompt/policies/bluesky/{args.prompt_mode}.md)")
    else:
        prompt = InstructionDrivenPrompt(mode=args.prompt_mode, platform="bluesky")
        default_dir = "outputs_instruction-driven/"
        auto_name = f"bluesky__{args.model}__{args.prompt_mode}__conf-true.jsonl"
        print(f"Using prompt mode: {args.prompt_mode} (policy: prompt/policies/bluesky/{args.prompt_mode}.md)")

    output_path = args.output or os.path.join(args.output_dir or default_dir, auto_name)

    print(f"\n{'=' * 60}\nInput:  {args.dataset}\nOutput: {output_path}\n{'=' * 60}")

    if example_driven:
        run_pipeline_example(provider, client, args.model, prompt, args.dataset, output_path, args)
    else:
        run_pipeline(provider, client, args.model, prompt, args.dataset, output_path, args)


if __name__ == "__main__":
    main()
