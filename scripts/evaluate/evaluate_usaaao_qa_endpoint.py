#!/usr/bin/env python3
"""Benchmark API endpoint models on USAAAO_QA.

This runner is for closed/API-hosted models such as GPT, Claude, Gemini via an
OpenAI-compatible endpoint, or another chat-completions compatible service. It
uses the same dataset loader, metrics, CSV schema, summaries, and optional
OpenAI judge as the local-model evaluator.

Examples:
    # OpenAI Responses API, full multimodal benchmark if the model supports images.
    python3 scripts/evaluate/evaluate_usaaao_qa_endpoint.py \
        --provider openai-responses \
        --model gpt-5.4 \
        --judge --judge-model gpt-5.6 \
        --years 2020 2021 2022 2023 2024 2025 2026

    # Anthropic Messages API.
    ANTHROPIC_API_KEY=... python3 scripts/evaluate/evaluate_usaaao_qa_endpoint.py \
        --provider anthropic-messages \
        --model claude-opus-4-8 \
        --text-only \
        --judge --judge-model gpt-5.6

    # Generic OpenAI-compatible chat-completions endpoint.
    python3 scripts/evaluate/evaluate_usaaao_qa_endpoint.py \
        --provider openai-chat \
        --endpoint-url https://example.invalid/v1/chat/completions \
        --api-key-env VENDOR_API_KEY \
        --model vendor-model-id \
        --text-only

    # Azure OpenAI chat-completions endpoint. Here --model is the deployment name.
    AZURE_OPENAI_API_KEY=... python3 scripts/evaluate/evaluate_usaaao_qa_endpoint.py \
        --provider azure-openai-chat \
        --endpoint-url "https://RESOURCE.openai.azure.com/openai/deployments/gpt-5.5/chat/completions?api-version=2026-05-01-preview" \
        --model gpt-5.5
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import config
from evaluate_usaaao_qa import (
    BERTScoreScorer,
    CSV_FIELDNAMES,
    CosineScorer,
    DEFAULT_DATASET_ROOT,
    OUTPUT_DIR,
    OpenAIJudge,
    TeeStream,
    append_csv_row,
    build_generation_prompt,
    build_output_paths,
    compute_bleu,
    compute_rouge_l,
    filter_examples,
    load_examples,
    load_rows_from_csv,
    make_run_stem,
    nested_import,
    print_summary,
    summarize,
    write_csv,
)


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Endpoint model id to evaluate.")
    parser.add_argument(
        "--provider",
        choices=[
            "openai-responses",
            "openai-chat",
            "azure-openai-chat",
            "anthropic-messages",
        ],
        default="openai-responses",
        help="API protocol to use for generation.",
    )
    parser.add_argument(
        "--endpoint-url",
        default=None,
        help="Full endpoint URL override. Use this for vendor/proxy endpoints.",
    )
    parser.add_argument(
        "--azure-endpoint",
        default=None,
        help=(
            "Azure OpenAI resource endpoint for generation, e.g. "
            "https://RESOURCE.openai.azure.com. Used with --provider azure-openai-chat "
            "when --endpoint-url is not supplied."
        ),
    )
    parser.add_argument(
        "--azure-api-version",
        default=None,
        help=(
            "Azure OpenAI API version for generation. Used with --provider "
            "azure-openai-chat when --endpoint-url is not supplied."
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable containing the generation API token.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Generation API token. Prefer --api-key-env or environment variables.",
    )
    parser.add_argument(
        "--anthropic-version",
        default="2023-06-01",
        help="Anthropic API version header for --provider anthropic-messages.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2017, 2018, 2019],
        help="Years to evaluate.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root directory containing root-level year JSONL files and images/.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional example cap.")
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Skip rows with linked images. Use this for text-only endpoint models.",
    )
    parser.add_argument(
        "--multimodal-only",
        action="store_true",
        help="Only score rows with linked images.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Enable OpenAI LLM-as-judge scoring.",
    )
    parser.add_argument(
        "--judge-model",
        default=config.OPENAI_JUDGE_MODEL,
        help="OpenAI judge model. Keep this fixed across compared models.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=config.DEFAULT_MAX_NEW_TOKENS,
        help="Maximum generated answer tokens.",
    )
    parser.add_argument(
        "--chat-token-param",
        choices=["auto", "max_tokens", "max_completion_tokens"],
        default="auto",
        help=(
            "Token-limit parameter for chat-completions providers. `auto` uses "
            "max_completion_tokens for Azure chat endpoints and max_tokens elsewhere."
        ),
    )
    parser.add_argument(
        "--omit-temperature",
        action="store_true",
        help="Do not send temperature to the generation endpoint.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="Optional reasoning effort for GPT reasoning-style chat endpoints.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=config.DEFAULT_TEMPERATURE,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between endpoint calls to reduce rate-limit pressure.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries for transient endpoint failures such as 429/5xx.",
    )
    parser.add_argument(
        "--retry-initial-seconds",
        type=float,
        default=10.0,
        help="Initial retry delay for transient endpoint failures.",
    )
    parser.add_argument(
        "--retry-max-seconds",
        type=float,
        default=180.0,
        help="Maximum retry delay for transient endpoint failures.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--endpoint-smoke-test",
        action="store_true",
        help="Send one short request to the generation endpoint, print the response, and exit.",
    )
    parser.add_argument("--run-stem", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing CSV for the same run stem.",
    )
    parser.add_argument(
        "--empty-answer-policy",
        choices=["fail", "score-zero"],
        default="fail",
        help=(
            "How to handle endpoint responses with no visible answer text. "
            "`fail` preserves strict behavior. `score-zero` records the example "
            "as an incorrect empty prediction and continues."
        ),
    )
    parser.add_argument(
        "--endpoint-error-policy",
        choices=["fail", "score-zero"],
        default="fail",
        help=(
            "How to handle generation endpoint errors after retries are exhausted. "
            "`fail` preserves strict behavior. `score-zero` records the example "
            "as an incorrect failed generation and continues."
        ),
    )
    return parser.parse_args()


def default_endpoint_url(provider: str) -> str:
    if provider == "openai-responses":
        return OPENAI_RESPONSES_URL
    if provider == "openai-chat":
        return OPENAI_CHAT_URL
    if provider == "azure-openai-chat":
        return ""
    if provider == "anthropic-messages":
        return ANTHROPIC_MESSAGES_URL
    raise ValueError(f"Unsupported provider: {provider}")


def build_azure_chat_url(endpoint: str, deployment: str, api_version: str) -> str:
    endpoint = endpoint.rstrip("/")
    deployment = urllib.parse.quote(deployment, safe="")
    return (
        f"{endpoint}/openai/deployments/{deployment}/chat/completions"
        f"?api-version={urllib.parse.quote(api_version, safe='')}"
    )


def resolve_endpoint_url(args: argparse.Namespace) -> str:
    if args.endpoint_url:
        return args.endpoint_url
    if args.provider == "azure-openai-chat":
        azure_endpoint = (
            args.azure_endpoint
            or os.getenv("CANDIDATE_AZURE_OPENAI_ENDPOINT")
            or os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        azure_api_version = (
            args.azure_api_version
            or os.getenv("CANDIDATE_OPENAI_API_VERSION")
            or os.getenv("AZURE_OPENAI_API_VERSION")
            or os.getenv("OPENAI_API_VERSION")
        )
        if not azure_endpoint or not azure_api_version:
            raise ValueError(
                "Azure generation endpoint setup is incomplete. Either pass "
                "--endpoint-url as the full deployment URL, or pass "
                "--azure-endpoint and --azure-api-version. You can also set "
                "CANDIDATE_AZURE_OPENAI_ENDPOINT and CANDIDATE_OPENAI_API_VERSION."
            )
        return build_azure_chat_url(azure_endpoint, args.model, azure_api_version)
    return default_endpoint_url(args.provider)


def default_api_key_env(provider: str) -> str:
    if provider in {"openai-responses", "openai-chat"}:
        return "OPENAI_API_KEY"
    if provider == "azure-openai-chat":
        return "AZURE_OPENAI_API_KEY"
    if provider == "anthropic-messages":
        return "ANTHROPIC_API_KEY"
    raise ValueError(f"Unsupported provider: {provider}")


def resolve_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key
    env_name = args.api_key_env or default_api_key_env(args.provider)
    api_key = os.getenv(env_name)
    if not api_key:
        raise ValueError(
            f"No generation API key found. Set {env_name}, pass --api-key-env, "
            "or pass --api-key."
        )
    return api_key


def image_to_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def image_to_anthropic_block(path: Path) -> Dict[str, Any]:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": mime_type,
            "data": payload,
        },
    }


def post_json(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: float,
    *,
    max_retries: int = 0,
    retry_initial_seconds: float = 10.0,
    retry_max_seconds: float = 180.0,
) -> Dict[str, Any]:
    headers = dict(headers)
    headers.setdefault("Accept", "application/json")
    headers.setdefault("User-Agent", "OpenAI/Python 1.0 AstroBench")
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    transient_codes = {408, 409, 425, 429, 500, 502, 503, 504}
    delay = retry_initial_seconds
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
                response_cost = response.headers.get("x-litellm-response-cost")
                try:
                    parsed = json.loads(body)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        "Endpoint returned non-JSON response: "
                        f"{body[:2000] if body else '<empty body>'}"
                    ) from exc
                if isinstance(parsed, dict) and response_cost:
                    parsed["_litellm_response_cost"] = response_cost
                return parsed
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code in transient_codes and attempt < max_retries:
                wait = min(delay, retry_max_seconds)
                print(
                    f"Endpoint HTTP {exc.code}; retrying in {wait:.1f}s "
                    f"({attempt + 1}/{max_retries})."
                )
                time.sleep(wait)
                delay = min(delay * 2, retry_max_seconds)
                continue
            raise RuntimeError(f"Endpoint HTTP {exc.code}: {body[:2000]}") from exc
        except urllib.error.URLError as exc:
            if attempt < max_retries:
                wait = min(delay, retry_max_seconds)
                print(
                    f"Endpoint connection error; retrying in {wait:.1f}s "
                    f"({attempt + 1}/{max_retries}): {exc}"
                )
                time.sleep(wait)
                delay = min(delay * 2, retry_max_seconds)
                continue
            raise RuntimeError(f"Endpoint connection error: {exc}") from exc


def extract_openai_chat_text(response: Dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return extract_openai_responses_text(response)
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        content = message.get("refusal") or message.get("reasoning_content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_bits = []
        for part in content:
            if isinstance(part, str):
                text_bits.append(part)
            elif isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    text_bits.append(part["text"])
                elif isinstance(part.get("content"), str):
                    text_bits.append(part["content"])
                elif isinstance(part.get("input_text"), str):
                    text_bits.append(part["input_text"])
                elif isinstance(part.get("output_text"), str):
                    text_bits.append(part["output_text"])
        return "\n".join(bit for bit in text_bits if bit)
    if isinstance(content, dict):
        for key in ("text", "content", "output_text"):
            value = content.get(key)
            if isinstance(value, str):
                return value
    return ""


def extract_openai_responses_text(response: Dict[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str):
        return direct
    text_bits: List[str] = []
    for item in response.get("output", []) or []:
        for part in item.get("content", []) or []:
            if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                text_bits.append(part.get("text", ""))
    return "\n".join(bit for bit in text_bits if bit)


def extract_anthropic_text(response: Dict[str, Any]) -> str:
    text_bits = []
    for part in response.get("content", []) or []:
        if isinstance(part, dict) and part.get("type") == "text":
            text_bits.append(part.get("text", ""))
    return "\n".join(bit for bit in text_bits if bit)


class EndpointGenerator:
    def __init__(self, args: argparse.Namespace):
        self.provider = args.provider
        self.model = args.model
        self.endpoint_url = resolve_endpoint_url(args)
        self.api_key = resolve_api_key(args)
        self.max_new_tokens = args.max_new_tokens
        self.temperature = None if args.omit_temperature else args.temperature
        self.chat_token_param = args.chat_token_param
        self.reasoning_effort = args.reasoning_effort
        self.timeout = args.timeout
        self.sleep_seconds = args.sleep_seconds
        self.max_retries = args.max_retries
        self.retry_initial_seconds = args.retry_initial_seconds
        self.retry_max_seconds = args.retry_max_seconds
        self.anthropic_version = args.anthropic_version

    def _post_json(self, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
        return post_json(
            self.endpoint_url,
            headers,
            payload,
            self.timeout,
            max_retries=self.max_retries,
            retry_initial_seconds=self.retry_initial_seconds,
            retry_max_seconds=self.retry_max_seconds,
        )

    def generate(self, example) -> str:
        prompt = build_generation_prompt(example)
        if self.provider == "openai-responses":
            prediction = self._generate_openai_responses(prompt, example.image_path)
        elif self.provider == "openai-chat":
            prediction = self._generate_openai_chat(prompt, example.image_path)
        elif self.provider == "azure-openai-chat":
            prediction = self._generate_openai_chat(prompt, example.image_path)
        elif self.provider == "anthropic-messages":
            prediction = self._generate_anthropic(prompt, example.image_path)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)
        prediction = prediction.strip()
        if not prediction:
            raise RuntimeError(
                f"Endpoint model `{self.model}` returned an empty answer. "
                "The response schema may need a new extractor. Raw response snippet: "
                f"{json.dumps(getattr(self, 'last_response', {}))[:2000]}"
            )
        return prediction

    def _generate_openai_responses(self, prompt: str, image_path: Optional[Path]) -> str:
        content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        if image_path is not None:
            content.append({"type": "input_image", "image_url": image_to_data_url(image_path)})
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": self.max_new_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = self._post_json(headers, payload)
        self.last_response = response
        return extract_openai_responses_text(response)

    def _generate_openai_chat(self, prompt: str, image_path: Optional[Path]) -> str:
        if self.provider == "azure-openai-chat" and not self.endpoint_url:
            raise ValueError(
                "--provider azure-openai-chat requires --endpoint-url. Use the Azure "
                "chat completions URL for the deployment, e.g. "
                "https://RESOURCE.openai.azure.com/openai/deployments/DEPLOYMENT/"
                "chat/completions?api-version=YYYY-MM-DD-preview"
            )
        content: Any = prompt
        if image_path is not None:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
            ]
        payload: Dict[str, Any] = {
            "messages": [{"role": "user", "content": content}],
        }
        token_param = self.chat_token_param
        if token_param == "auto":
            token_param = (
                "max_completion_tokens"
                if self.provider == "azure-openai-chat"
                else "max_tokens"
            )
        payload[token_param] = self.max_new_tokens
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        headers = {"Content-Type": "application/json"}
        if self.provider == "azure-openai-chat":
            headers["api-key"] = self.api_key
        else:
            payload["model"] = self.model
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = self._post_json(headers, payload)
        self.last_response = response
        return extract_openai_chat_text(response)

    def _generate_anthropic(self, prompt: str, image_path: Optional[Path]) -> str:
        content: List[Dict[str, Any]] = []
        if image_path is not None:
            content.append(image_to_anthropic_block(image_path))
        content.append({"type": "text", "text": prompt})
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": content}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "Content-Type": "application/json",
        }
        response = self._post_json(headers, payload)
        self.last_response = response
        return extract_anthropic_text(response)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = args.dataset_root.resolve()
    run_stem = make_run_stem(args.run_stem, args.model)
    paths = build_output_paths(args.output_dir, run_stem, 1, 0)
    log_path = paths["log"]
    csv_path = paths["csv"]
    json_path = paths["summary"]

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_handle = log_path.open("w")
    sys.stdout = TeeStream(original_stdout, log_handle)
    sys.stderr = TeeStream(original_stderr, log_handle)

    try:
        print(f"Logging to: {log_path}")
        examples = filter_examples(load_examples(args.years, dataset_root), args)
        if not examples:
            print("No examples matched the requested filters.", file=sys.stderr)
            return 1

        print(f"Loaded {len(examples)} examples.")
        print(f"Dataset root: {dataset_root}")
        print(f"Generation provider: {args.provider}")
        generation_endpoint = resolve_endpoint_url(args)
        print(f"Generation endpoint: {generation_endpoint}")
        print(f"Generation model: {args.model}")
        print(f"Judge enabled: {args.judge}")
        if args.judge and args.model == args.judge_model:
            print(
                "WARNING: generation model equals judge model. Judge scores for this run "
                "should be treated as self-judged and not directly comparable."
            )

        generator = EndpointGenerator(args)

        if args.endpoint_smoke_test:
            print("Running endpoint smoke test...")
            class SmokeExample:
                record_id = "endpoint_smoke_test"
                question = (
                    "Smoke test: answer in one short sentence. What is the nearest "
                    "star to Earth other than the Sun?"
                )
                image_path = None

            prediction = generator.generate(SmokeExample())
            print("\nSmoke-test response:")
            print(prediction)
            return 0

        cosine_scorer = None
        try:
            cosine_scorer = CosineScorer(config.EMBEDDING_MODEL_NAME)
        except Exception as exc:
            print(f"Cosine scorer unavailable: {exc}", file=sys.stderr)

        bertscore_scorer = None
        try:
            bertscore_scorer = BERTScoreScorer()
        except Exception as exc:
            print(f"BERTScore unavailable: {exc}", file=sys.stderr)

        judge = None
        if args.judge:
            judge = OpenAIJudge(args.judge_model)

        rows: List[Dict[str, Any]] = []
        completed_ids = set()
        if args.resume and csv_path.exists():
            rows = load_rows_from_csv(csv_path)
            completed_ids = {row["id"] for row in rows}
            print(f"Resume enabled: loaded {len(rows)} completed rows from {csv_path}")

        start = time.time()
        for index, example in enumerate(examples, start=1):
            if example.record_id in completed_ids:
                print(f"[{index}/{len(examples)}] {example.record_id} (already completed, skipping)")
                continue
            print(f"[{index}/{len(examples)}] {example.record_id}")
            generation_error = None
            try:
                prediction = generator.generate(example)
            except RuntimeError as exc:
                is_empty_answer = "returned an empty answer" in str(exc)
                can_score_zero = (
                    is_empty_answer
                    and args.empty_answer_policy == "score-zero"
                ) or (
                    not is_empty_answer
                    and args.endpoint_error_policy == "score-zero"
                )
                if not can_score_zero:
                    raise
                generation_error = str(exc)
                prediction = ""
                if is_empty_answer:
                    print(
                        "WARNING: endpoint returned an empty visible answer; "
                        "recording this example as incorrect and continuing."
                    )
                else:
                    print(
                        "WARNING: generation endpoint failed after retries; "
                        "recording this example as incorrect and continuing."
                    )

            row: Dict[str, Any] = {
                "id": example.record_id,
                "year": example.year,
                "question_length": example.question_length,
                "has_image": example.image_path is not None,
                "question": example.question,
                "reference_answer": example.reference_answer,
                "prediction": prediction,
                "bleu": compute_bleu(example.reference_answer, prediction),
                "rouge_l": compute_rouge_l(example.reference_answer, prediction),
                "bertscore_f1": None,
                "cosine_similarity": None,
                "judge_verdict": None,
                "judge_score": None,
                "judge_rationale": None,
            }

            if generation_error is not None:
                row["bleu"] = 0.0
                row["rouge_l"] = 0.0
                row["judge_verdict"] = "incorrect"
                row["judge_score"] = 0.0
                row["judge_rationale"] = (
                    "Generation endpoint did not produce a scorable answer. "
                    f"{generation_error[:500]}"
                )

            if generation_error is None and bertscore_scorer is not None:
                row["bertscore_f1"] = bertscore_scorer.score(
                    example.reference_answer, prediction
                )

            if generation_error is None and cosine_scorer is not None:
                row["cosine_similarity"] = cosine_scorer.score(
                    example.reference_answer, prediction
                )

            if generation_error is None and judge is not None:
                judged = judge.judge(example.question, example.reference_answer, prediction)
                row["judge_verdict"] = judged.get("verdict")
                judge_score = judged.get("score")
                if isinstance(judge_score, str):
                    try:
                        judge_score = float(judge_score)
                    except ValueError:
                        pass
                row["judge_score"] = judge_score
                row["judge_rationale"] = judged.get("rationale")

            rows.append(row)
            append_csv_row(csv_path, row)
            completed_ids.add(example.record_id)

        summary = summarize(rows)
        summary["elapsed_seconds"] = round(time.time() - start, 2)
        summary["generation_model"] = args.model
        summary["generation_provider"] = args.provider
        summary["generation_endpoint"] = generation_endpoint
        summary["judge_model"] = args.judge_model if args.judge else None
        summary["self_judged"] = bool(args.judge and args.model == args.judge_model)
        summary["log_file"] = str(log_path)

        write_csv(csv_path, rows)
        json_path.write_text(json.dumps(summary, indent=2))

        print_summary(summary)
        print(f"\nWrote per-example results to: {csv_path}")
        print(f"Wrote summary to: {json_path}")
        print(f"Wrote log to: {log_path}")
        return 0
    except Exception:
        print("\nBenchmark run failed with an exception:", file=sys.stderr)
        traceback.print_exc()
        print(f"\nPartial log captured at: {log_path}", file=sys.stderr)
        return 1
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
