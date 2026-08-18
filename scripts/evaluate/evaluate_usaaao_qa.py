#!/usr/bin/env python3
"""Benchmark USAAAO_QA with classic metrics and optional LLM-as-judge.

Example:
    python3 scripts/evaluate/evaluate_usaaao_qa.py \
        --model google/gemma-3-4b-it \
        --judge \
        --years 2017 2018 2019

Recommended packages:
    pip install transformers torch pillow pandas nltk rouge-score bert-score \
        sentence-transformers scikit-learn openai
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import config


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "usaaao_qa_local"
OUTPUT_DIR = REPO_ROOT / "benchmark_results"


def nested_import(module_name: str, attr_name: str):
    try:
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    except (ImportError, AttributeError):
        return None


def resolve_pipeline_device_index(require_gpu: bool = True) -> int:
    torch = __import__("torch")
    if torch.cuda.is_available():
        return 0
    if require_gpu:
        raise RuntimeError(
            "GPU execution was requested, but PyTorch could not initialize CUDA/ROCm. "
            "Check your driver and torch build compatibility."
        )
    return -1


@dataclass
class Example:
    record_id: str
    year: int
    question_length: str
    parent_id: Optional[str]
    question: str
    reference_answer: str
    image_path: Optional[Path]
    raw: Dict[str, Any]


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams
        self._closed = False

    def write(self, data: str) -> int:
        if self._closed:
            return 0
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        if self._closed:
            return
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        if self._closed:
            return False
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    def close(self) -> None:
        if self._closed:
            return
        for stream in self.streams:
            close_method = getattr(stream, "close", None)
            if close_method is not None:
                close_method()
        self._closed = True


def parse_parallel_devices(parallel_devices: Optional[str]) -> List[str]:
    if not parallel_devices:
        return []
    return [part.strip() for part in parallel_devices.split(",") if part.strip()]


def shard_examples(
    examples: List[Example], num_shards: int, shard_index: int
) -> List[Example]:
    if num_shards < 1:
        raise ValueError("--num-shards must be at least 1.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards.")
    return examples[shard_index::num_shards]


def model_slug(model_name: str) -> str:
    slug = model_name.split("/")[-1].lower()
    allowed = []
    previous_dash = False
    for char in slug:
        if char.isalnum() or char == ".":
            allowed.append(char)
            previous_dash = False
        else:
            if not previous_dash:
                allowed.append("-")
                previous_dash = True
    slug = "".join(allowed).strip("-")
    slug = slug.replace("-instruct", "")
    slug = slug.replace("-it", "")
    slug = slug.replace("-chat", "")
    return slug or "model"


def make_run_stem(run_stem: Optional[str], model_name: str) -> str:
    if run_stem:
        return run_stem
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"usaaao_{model_slug(model_name)}_eval_{timestamp}"


def worker_suffix(num_shards: int, shard_index: int) -> str:
    if num_shards <= 1:
        return ""
    return f"_worker{shard_index:02d}"


def build_output_paths(
    output_dir: Path, run_stem: str, num_shards: int, shard_index: int
) -> Dict[str, Path]:
    suffix = worker_suffix(num_shards, shard_index)
    return {
        "log": output_dir / f"{run_stem}{suffix}.log",
        "csv": output_dir / f"{run_stem}{suffix}.csv",
        "summary": output_dir / f"{run_stem}{suffix}_summary.json",
    }


CSV_FIELDNAMES = [
    "id",
    "year",
    "question_length",
    "has_image",
    "question",
    "reference_answer",
    "prediction",
    "bleu",
    "rouge_l",
    "bertscore_f1",
    "cosine_similarity",
    "judge_verdict",
    "judge_score",
    "judge_rationale",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=config.DEFAULT_GENERATION_MODEL,
        help="Local Hugging Face generation model.",
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
        help="Dataset root directory containing `data/` and optionally `images/`.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of examples.",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Skip any rows with linked images.",
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
        help="OpenAI judge model override.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=config.DEFAULT_MAX_NEW_TOKENS,
        help="Generation length cap.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=config.DEFAULT_TEMPERATURE,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--parallel-devices",
        default=None,
        help=(
            "Comma-separated device ids to use in parallel, e.g. `0,1,2,3`. "
            "On Frontier, these are the ROCm-visible GPU/GCD ids."
        ),
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--run-stem",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing CSV for the same run stem instead of starting over.",
    )
    return parser.parse_args()


def load_examples(years: Iterable[int], dataset_root: Path) -> List[Example]:
    examples: List[Example] = []
    data_dir = dataset_root
    for year in years:
        path = data_dir / f"{year}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                image_rel = row.get("image")
                image_path = None
                if image_rel:
                    image_path = (dataset_root / image_rel).resolve()
                examples.append(
                    Example(
                        record_id=row["id"],
                        year=row["year"],
                        question_length=row["question_length"],
                        parent_id=row.get("parent_id"),
                        question=row["question"],
                        reference_answer=row["answer"],
                        image_path=image_path,
                        raw=row,
                    )
                )
    return examples


def filter_examples(examples: List[Example], args: argparse.Namespace) -> List[Example]:
    filtered = []
    for example in examples:
        has_image = example.image_path is not None
        if args.text_only and has_image:
            continue
        if args.multimodal_only and not has_image:
            continue
        filtered.append(example)
    if args.limit is not None:
        filtered = filtered[: args.limit]
    return filtered


def build_generation_prompt(example: Example) -> str:
    return (
        "Answer the astronomy/astrophysics question directly.\n"
        "Provide a concise final answer with essential reasoning only.\n"
        "If the question is quantitative, show the key calculation.\n\n"
        f"Question:\n{example.question}"
    )


class GemmaGenerator:
    def __init__(self, model_name: str, max_new_tokens: int, temperature: float):
        transformers_pipeline = nested_import("transformers", "pipeline")
        if transformers_pipeline is None:
            raise ImportError(
                "transformers is required to run local generation. "
                "Install it with `pip install transformers torch pillow`."
            )
        self.PIL_Image = nested_import("PIL.Image", "open")
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.multimodal_capable = False
        self.pipeline_device = resolve_pipeline_device_index(require_gpu=True)

        # Prefer a multimodal pipeline when the model supports it, but fall back
        # to a text-generation pipeline for text-only chat models.
        try:
            candidate_pipe = transformers_pipeline(
                "image-text-to-text",
                model=model_name,
                device=self.pipeline_device,
            )
            processor = getattr(candidate_pipe, "processor", None)
            if processor is not None and hasattr(processor, "apply_chat_template"):
                self.pipe = candidate_pipe
                self.multimodal_capable = True
            else:
                self.pipe = transformers_pipeline(
                    "text-generation",
                    model=model_name,
                    device=self.pipeline_device,
                )
        except Exception:
            self.pipe = transformers_pipeline(
                "text-generation",
                model=model_name,
                device=self.pipeline_device,
            )

    def generate(self, example: Example) -> str:
        prompt = build_generation_prompt(example)
        multimodal_message: Dict[str, Any] = {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
        text_chat_message: Dict[str, Any] = {
            "role": "user",
            "content": prompt,
        }
        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "max_length": None,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature

        if example.image_path is not None:
            if not self.multimodal_capable:
                raise ValueError(
                    f"Model `{self.model_name}` is text-only and cannot process image-linked "
                    "examples. Re-run with `--text-only` or choose a multimodal model."
                )
            if self.PIL_Image is None:
                raise ImportError(
                    "Pillow is required for multimodal evaluation. "
                    "Install it with `pip install pillow`."
                )
            image = self.PIL_Image(example.image_path)
            multimodal_message["content"].insert(0, {"type": "image", "image": image})

        if self.multimodal_capable:
            result = self.pipe(
                [multimodal_message],
                generate_kwargs=generate_kwargs,
                processor_kwargs={},
            )
        else:
            tokenizer = getattr(self.pipe, "tokenizer", None)
            has_chat_template = bool(
                tokenizer is not None and getattr(tokenizer, "chat_template", None)
            )
            text_input: Any = [text_chat_message] if has_chat_template else prompt
            result = self.pipe(text_input, **generate_kwargs)
        return extract_text_from_generation(result).strip()


def extract_text_from_generation(result: Any) -> str:
    if isinstance(result, list) and result:
        item = result[0]
        if isinstance(item, dict):
            generated = item.get("generated_text")
            if isinstance(generated, list) and generated:
                last = generated[-1]
                if isinstance(last, dict):
                    content = last.get("content")
                    if isinstance(content, list):
                        text_bits = []
                        for piece in content:
                            if isinstance(piece, dict) and piece.get("type") == "text":
                                text_bits.append(piece.get("text", ""))
                        if text_bits:
                            return "\n".join(text_bits)
                    if isinstance(content, str):
                        return content
            if isinstance(generated, str):
                return generated
    return str(result)


def simple_tokenize(text: str) -> List[str]:
    return text.lower().split()


def compute_bleu(reference: str, prediction: str) -> Optional[float]:
    sentence_bleu = nested_import("nltk.translate.bleu_score", "sentence_bleu")
    smoothing_fn = nested_import("nltk.translate.bleu_score", "SmoothingFunction")
    if sentence_bleu is None or smoothing_fn is None:
        return None
    reference_tokens = [simple_tokenize(reference)]
    prediction_tokens = simple_tokenize(prediction)
    smoothie = smoothing_fn().method1
    return float(sentence_bleu(reference_tokens, prediction_tokens, smoothing_function=smoothie))


def compute_rouge_l(reference: str, prediction: str) -> Optional[float]:
    rouge_scorer = nested_import("rouge_score.rouge_scorer", "RougeScorer")
    if rouge_scorer is None:
        return None
    scorer = rouge_scorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return float(scores["rougeL"].fmeasure)


class BERTScoreScorer:
    def __init__(self):
        scorer_cls = nested_import("bert_score", "BERTScorer")
        if scorer_cls is None:
            raise ImportError(
                "bert-score is required for BERTScore. Install it with `pip install bert-score`."
            )
        self.scorer = scorer_cls(lang="en", rescale_with_baseline=True)

    def score(self, reference: str, prediction: str) -> float:
        _, _, f1 = self.scorer.score([prediction], [reference])
        return float(f1[0].item())


class CosineScorer:
    def __init__(self, model_name: str):
        sentence_transformers = nested_import("sentence_transformers", "SentenceTransformer")
        cosine_similarity = nested_import("sklearn.metrics.pairwise", "cosine_similarity")
        if sentence_transformers is None or cosine_similarity is None:
            raise ImportError(
                "sentence-transformers and scikit-learn are required for cosine similarity. "
                "Install with `pip install sentence-transformers scikit-learn`."
            )
        self.model = sentence_transformers(model_name)
        self.cosine_similarity = cosine_similarity

    def score(self, reference: str, prediction: str) -> float:
        embeddings = self.model.encode([reference, prediction], convert_to_tensor=False)
        similarity = self.cosine_similarity([embeddings[0]], [embeddings[1]])
        return float(similarity[0][0])


class OpenAIJudge:
    def __init__(self, model_name: str):
        self.client = self._build_client()
        self.model_name = model_name

    def _build_client(self):
        api_key = (
            os.getenv("JUDGE_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or config.OPENAI_API_KEY
        )
        base_url = (
            os.getenv("JUDGE_OPENAI_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or config.OPENAI_BASE_URL
        )
        azure_endpoint = (
            os.getenv("JUDGE_AZURE_OPENAI_ENDPOINT")
            or os.getenv("JUDGE_OPENAI_AZURE_ENDPOINT")
            or os.getenv("AZURE_OPENAI_ENDPOINT")
            or os.getenv("OPENAI_AZURE_ENDPOINT")
            or config.OPENAI_AZURE_ENDPOINT
        )
        api_version = (
            os.getenv("JUDGE_OPENAI_API_VERSION")
            or os.getenv("OPENAI_API_VERSION")
            or config.OPENAI_API_VERSION
        )
        use_azure_env = os.getenv("JUDGE_OPENAI_USE_AZURE")
        if use_azure_env is None:
            use_azure_env = os.getenv("OPENAI_USE_AZURE")
        use_azure = config.OPENAI_USE_AZURE
        if use_azure_env is not None:
            use_azure = use_azure_env.strip().lower() in {"1", "true", "yes", "on"}

        if not api_key:
            raise ValueError(
                "No OpenAI API key found. Set OPENAI_API_KEY in the environment or in "
                "scripts/evaluate/config.py or environment variables."
            )

        if use_azure:
            azure_client = nested_import("openai", "AzureOpenAI")
            if azure_client is None:
                raise ImportError(
                    "The openai package with AzureOpenAI support is required for judge mode. "
                    "Install it with `pip install openai`."
                )
            if not azure_endpoint:
                raise ValueError(
                    "Azure judge mode is enabled but no endpoint was provided. Set "
                    "OPENAI_AZURE_ENDPOINT in config.py or AZURE_OPENAI_ENDPOINT in the environment."
                )
            if not api_version:
                raise ValueError(
                    "Azure judge mode is enabled but no API version was provided. Set "
                    "OPENAI_API_VERSION in config.py or the environment."
                )
            return azure_client(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
            )

        openai_client = nested_import("openai", "OpenAI")
        if openai_client is None:
            raise ImportError(
                "openai is required for judge mode. Install it with `pip install openai`."
            )
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return openai_client(**kwargs)

    @staticmethod
    def _parse_json_payload(payload: str) -> Dict[str, Any]:
        text = payload.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise

    def judge(self, question: str, reference_answer: str, model_answer: str) -> Dict[str, Any]:
        prompt = (
            "You are grading an astronomy benchmark answer.\n"
            "Compare the model answer against the reference answer.\n"
            "Return strict JSON with keys: verdict, score, rationale.\n"
            "Rules:\n"
            '- verdict must be one of "correct", "partial", "incorrect"\n'
            "- score must be 1.0, 0.5, or 0.0\n"
            "- rationale must be brief and factual\n\n"
            f"Question:\n{question}\n\n"
            f"Reference answer:\n{reference_answer}\n\n"
            f"Model answer:\n{model_answer}"
        )

        response = None
        last_exc: Optional[Exception] = None
        for attempt in range(5):
            try:
                response = self.client.responses.create(
                    model=self.model_name,
                    input=prompt,
                    text={"format": {"type": "json_object"}},
                )
                break
            except Exception as exc:
                last_exc = exc
                exc_name = exc.__class__.__name__.lower()
                exc_text = str(exc).lower()
                is_transient = any(
                    marker in exc_name or marker in exc_text
                    for marker in (
                        "timeout",
                        "timed out",
                        "connection",
                        "rate limit",
                        "server",
                        "temporarily",
                    )
                )
                if not is_transient or attempt == 4:
                    raise
                wait = min(5 * (2**attempt), 60)
                print(
                    f"Judge request failed transiently ({exc.__class__.__name__}); "
                    f"retrying in {wait}s ({attempt + 1}/5)."
                )
                time.sleep(wait)
        if response is None:
            raise RuntimeError("Judge request failed without a response.") from last_exc
        payload = response.output_text
        try:
            return self._parse_json_payload(payload)
        except json.JSONDecodeError:
            return {
                "verdict": "incorrect",
                "score": 0.0,
                "rationale": f"Judge returned non-JSON output: {payload[:200]}",
            }


def metric_average(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                continue
        values.append(value)
    if not values:
        return None
    return float(statistics.mean(values))


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "num_examples": len(rows),
        "bleu": metric_average(rows, "bleu"),
        "rouge_l": metric_average(rows, "rouge_l"),
        "bertscore_f1": metric_average(rows, "bertscore_f1"),
        "cosine_similarity": metric_average(rows, "cosine_similarity"),
        "judge_score": metric_average(rows, "judge_score"),
    }

    by_year: Dict[int, List[Dict[str, Any]]] = {}
    by_length: Dict[str, List[Dict[str, Any]]] = {}
    by_modality: Dict[str, List[Dict[str, Any]]] = {"text_only": [], "multimodal": []}
    for row in rows:
        by_year.setdefault(row["year"], []).append(row)
        by_length.setdefault(row["question_length"], []).append(row)
        modality = "multimodal" if row["has_image"] else "text_only"
        by_modality[modality].append(row)

    summary["by_year"] = {
        str(year): {
            "count": len(group),
            "judge_score": metric_average(group, "judge_score"),
            "bertscore_f1": metric_average(group, "bertscore_f1"),
            "cosine_similarity": metric_average(group, "cosine_similarity"),
        }
        for year, group in sorted(by_year.items())
    }
    summary["by_question_length"] = {
        label: {
            "count": len(group),
            "judge_score": metric_average(group, "judge_score"),
            "bertscore_f1": metric_average(group, "bertscore_f1"),
            "cosine_similarity": metric_average(group, "cosine_similarity"),
        }
        for label, group in sorted(by_length.items())
    }
    summary["by_modality"] = {
        label: {
            "count": len(group),
            "judge_score": metric_average(group, "judge_score"),
            "bertscore_f1": metric_average(group, "bertscore_f1"),
            "cosine_similarity": metric_average(group, "cosine_similarity"),
        }
        for label, group in by_modality.items()
        if group
    }
    return summary


def build_worker_command(
    args: argparse.Namespace, run_stem: str, num_shards: int, shard_index: int
) -> List[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        args.model,
        "--dataset-root",
        str(args.dataset_root),
        "--years",
        *[str(year) for year in args.years],
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--output-dir",
        str(args.output_dir),
        "--num-shards",
        str(num_shards),
        "--shard-index",
        str(shard_index),
        "--run-stem",
        run_stem,
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.text_only:
        cmd.append("--text-only")
    if args.multimodal_only:
        cmd.append("--multimodal-only")
    if args.judge:
        cmd.append("--judge")
        cmd.extend(["--judge-model", args.judge_model])
    if args.resume:
        cmd.append("--resume")
    return cmd


def parse_optional_float(value: str) -> Optional[float]:
    if value in {"", "None", None}:
        return None
    return float(value)


def parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def load_rows_from_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "id": row["id"],
                    "year": int(row["year"]),
                    "question_length": row["question_length"],
                    "has_image": parse_bool(row["has_image"]),
                    "question": row["question"],
                    "reference_answer": row["reference_answer"],
                    "prediction": row["prediction"],
                    "bleu": parse_optional_float(row["bleu"]),
                    "rouge_l": parse_optional_float(row["rouge_l"]),
                    "bertscore_f1": parse_optional_float(row["bertscore_f1"]),
                    "cosine_similarity": parse_optional_float(row["cosine_similarity"]),
                    "judge_verdict": row["judge_verdict"] or None,
                    "judge_score": parse_optional_float(row["judge_score"]),
                    "judge_rationale": row["judge_rationale"] or None,
                }
            )
    return rows


def append_csv_row(path: Path, row: Dict[str, Any]) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in CSV_FIELDNAMES})
        handle.flush()


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CSV_FIELDNAMES})


def print_summary(summary: Dict[str, Any]) -> None:
    print("\nSummary")
    print(f"  examples: {summary['num_examples']}")
    for key in ["bleu", "rouge_l", "bertscore_f1", "cosine_similarity", "judge_score"]:
        value = summary.get(key)
        if value is not None:
            print(f"  {key}: {value:.4f}")
    print("\nBy year")
    for year, stats in summary["by_year"].items():
        parts = [f"  {year}: count={stats['count']}"]
        judge = stats["judge_score"]
        bscore = stats["bertscore_f1"]
        cos = stats["cosine_similarity"]
        if judge is not None:
            parts.append(f"judge={judge:.4f}")
        if bscore is not None:
            parts.append(f"bertscore={bscore:.4f}")
        if cos is not None:
            parts.append(f"cosine={cos:.4f}")
        print(" ".join(parts))


def run_parallel_workers(args: argparse.Namespace, run_stem: str, log_path: Path) -> int:
    devices = parse_parallel_devices(args.parallel_devices)
    if not devices:
        raise ValueError("--parallel-devices was provided but no device ids were parsed.")

    print(f"Launching {len(devices)} workers across devices: {', '.join(devices)}")
    processes = []

    for shard_index, device in enumerate(devices):
        cmd = build_worker_command(args, run_stem, len(devices), shard_index)
        env = os.environ.copy()
        env["ROCR_VISIBLE_DEVICES"] = device
        env["HIP_VISIBLE_DEVICES"] = device
        env["CUDA_VISIBLE_DEVICES"] = device
        worker_paths = build_output_paths(args.output_dir, run_stem, len(devices), shard_index)
        print(f"Starting worker {shard_index} on device {device}")
        print("  Command:", " ".join(cmd))
        print(f"  Worker log: {worker_paths['log']}")
        proc = subprocess.Popen(cmd, env=env)
        processes.append((shard_index, device, proc))

    failures = []
    for shard_index, device, proc in processes:
        return_code = proc.wait()
        if return_code != 0:
            failures.append((shard_index, device, return_code))
            print(
                f"Worker {shard_index} on device {device} failed with exit code {return_code}.",
                file=sys.stderr,
            )
        else:
            print(f"Worker {shard_index} on device {device} completed successfully.")

    if failures:
        print("\nParallel run failed. Check the worker logs:", file=sys.stderr)
        for shard_index, device, return_code in failures:
            worker_paths = build_output_paths(args.output_dir, run_stem, len(devices), shard_index)
            print(
                f"  worker {shard_index} device {device}: exit={return_code} log={worker_paths['log']}",
                file=sys.stderr,
            )
        return 1

    rows: List[Dict[str, Any]] = []
    for shard_index in range(len(devices)):
        worker_paths = build_output_paths(args.output_dir, run_stem, len(devices), shard_index)
        rows.extend(load_rows_from_csv(worker_paths["csv"]))

    rows.sort(key=lambda row: row["id"])
    summary = summarize(rows)
    summary["generation_model"] = args.model
    summary["judge_model"] = args.judge_model if args.judge else None
    summary["log_file"] = str(log_path)
    summary["parallel_devices"] = devices
    summary["num_workers"] = len(devices)
    summary["worker_logs"] = [
        str(build_output_paths(args.output_dir, run_stem, len(devices), i)["log"])
        for i in range(len(devices))
    ]

    aggregate_paths = build_output_paths(args.output_dir, run_stem, 1, 0)
    write_csv(aggregate_paths["csv"], rows)
    aggregate_paths["summary"].write_text(json.dumps(summary, indent=2))

    print_summary(summary)
    print(f"\nWrote merged per-example results to: {aggregate_paths['csv']}")
    print(f"Wrote merged summary to: {aggregate_paths['summary']}")
    print(f"Wrote parent log to: {aggregate_paths['log']}")
    return 0


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = args.dataset_root.resolve()
    run_stem = make_run_stem(args.run_stem, args.model)
    paths = build_output_paths(args.output_dir, run_stem, args.num_shards, args.shard_index)
    log_path = paths["log"]

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_handle = log_path.open("w")
    sys.stdout = TeeStream(original_stdout, log_handle)
    sys.stderr = TeeStream(original_stderr, log_handle)

    try:
        print(f"Logging to: {log_path}")
        if args.parallel_devices:
            if args.num_shards != 1 or args.shard_index != 0:
                raise ValueError(
                    "--parallel-devices cannot be combined with explicit shard worker flags."
                )
            return run_parallel_workers(args, run_stem, log_path)

        examples = filter_examples(load_examples(args.years, dataset_root), args)
        examples = shard_examples(examples, args.num_shards, args.shard_index)
        if not examples:
            print("No examples matched the requested filters.", file=sys.stderr)
            return 1

        print(f"Loaded {len(examples)} examples.")
        print(f"Dataset root: {dataset_root}")
        print(f"Generation model: {args.model}")
        print(f"Judge enabled: {args.judge}")

        generator = GemmaGenerator(
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        cosine_scorer = None
        try:
            cosine_scorer = CosineScorer(config.EMBEDDING_MODEL_NAME)
        except ImportError as exc:
            print(f"Skipping cosine similarity: {exc}")

        bertscore_scorer = None
        try:
            bertscore_scorer = BERTScoreScorer()
        except ImportError as exc:
            print(f"Skipping BERTScore: {exc}")

        judge = None
        if args.judge:
            judge = OpenAIJudge(args.judge_model)

        csv_path = paths["csv"]
        json_path = paths["summary"]

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
            prediction = generator.generate(example)

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

            if bertscore_scorer is not None:
                row["bertscore_f1"] = bertscore_scorer.score(
                    example.reference_answer, prediction
                )

            if cosine_scorer is not None:
                row["cosine_similarity"] = cosine_scorer.score(
                    example.reference_answer, prediction
                )

            if judge is not None:
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
        summary["judge_model"] = args.judge_model if args.judge else None
        summary["log_file"] = str(log_path)
        summary["num_shards"] = args.num_shards
        summary["shard_index"] = args.shard_index

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
