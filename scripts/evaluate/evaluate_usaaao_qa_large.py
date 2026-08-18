#!/usr/bin/env python3
"""Benchmark large text-only models on USAAAO_QA with model-parallel loading.

This script is intended for very large causal language models such as 70B-class
checkpoints that need to be split across multiple visible GPUs. It keeps the
same metrics and judge flow as the standard benchmark script, but uses
AutoModelForCausalLM with `device_map="auto"` instead of a pipeline backend.

Example:
    CUDA_VISIBLE_DEVICES=2,3 python3 scripts/evaluate/evaluate_usaaao_qa_large.py \
        --model AstroMLab/AstroSage-70B-20251009 \
        --judge \
        --judge-model gpt-5 \
        --years 2017 2018 2019 \
        --text-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    summarize,
    write_csv,
    print_summary,
    nested_import,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Large Hugging Face causal LM to benchmark.",
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
        help="Skip any rows with linked images. Recommended for large text-only models.",
    )
    parser.add_argument(
        "--multimodal-only",
        action="store_true",
        help=argparse.SUPPRESS,
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
        "--dtype",
        choices=["auto", "bfloat16", "float16"],
        default="bfloat16",
        help="Torch dtype for model loading.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Attention backend to request when supported by the model.",
    )
    parser.add_argument(
        "--run-stem",
        default=None,
        help="Optional custom output stem.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing CSV for the same run stem instead of starting over.",
    )
    return parser.parse_args()


class LargeModelGenerator:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int,
        temperature: float,
        dtype: str,
        attn_implementation: str,
    ):
        transformers = nested_import("transformers", "AutoTokenizer")
        if transformers is None:
            raise ImportError(
                "transformers is required to run large-model generation. "
                "Install it with `pip install transformers torch`."
            )

        AutoTokenizer = nested_import("transformers", "AutoTokenizer")
        AutoModelForCausalLM = nested_import("transformers", "AutoModelForCausalLM")
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError("Failed to import AutoTokenizer/AutoModelForCausalLM.")

        torch = nested_import("torch", "bfloat16")
        if torch is None:
            raise ImportError("torch is required to run large-model generation.")
        torch_module = __import__("torch")
        if not torch_module.cuda.is_available():
            raise RuntimeError(
                "GPU execution was requested, but PyTorch could not initialize CUDA/ROCm. "
                "Check your driver and torch build compatibility."
            )

        dtype_map = {
            "auto": "auto",
            "bfloat16": torch,
            "float16": nested_import("torch", "float16"),
        }

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        model_kwargs: Dict[str, Any] = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "torch_dtype": dtype_map[dtype],
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.primary_device = self._resolve_primary_device()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _resolve_primary_device(self):
        hf_device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(hf_device_map, dict):
            for device in hf_device_map.values():
                if isinstance(device, str) and device not in {"cpu", "disk"}:
                    return device
                if isinstance(device, int):
                    return f"cuda:{device}"
        return getattr(self.model, "device", None)

    def generate(self, question_prompt: str) -> str:
        has_chat_template = bool(getattr(self.tokenizer, "chat_template", None))
        if has_chat_template:
            model_inputs = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": question_prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            model_inputs = self.tokenizer(
                f"{question_prompt}\n\nAnswer:\n",
                return_tensors="pt",
            )["input_ids"]

        if hasattr(model_inputs, "to"):
            if self.primary_device is not None:
                model_inputs = model_inputs.to(self.primary_device)

        attention_mask = None
        if hasattr(model_inputs, "ne") and self.tokenizer.pad_token_id is not None:
            attention_mask = model_inputs.ne(self.tokenizer.pad_token_id).long()

        generate_kwargs: Dict[str, Any] = {
            "input_ids": model_inputs,
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask
        if self.temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = self.temperature
        else:
            generate_kwargs["do_sample"] = False

        outputs = self.model.generate(**generate_kwargs)
        generated_ids = outputs[0][model_inputs.shape[-1] :]
        prediction = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()
        if not prediction:
            raise RuntimeError(
                "The generation model returned an empty answer. This commonly happens "
                "when a base (non-instruction-tuned) model receives a chat-style prompt. "
                "No metrics were saved for this example."
            )
        return prediction


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = args.dataset_root.resolve()
    run_stem = make_run_stem(args.run_stem, args.model)
    paths = build_output_paths(args.output_dir, run_stem, 1, 0)
    log_path = paths["log"]

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_handle = log_path.open("w")
    sys.stdout = TeeStream(original_stdout, log_handle)
    sys.stderr = TeeStream(original_stderr, log_handle)

    try:
        print(f"Logging to: {log_path}")
        examples = filter_examples(load_examples(args.years, dataset_root), args)
        if not args.text_only and any(example.image_path is not None for example in examples):
            raise ValueError(
                "Large-model script is text-only. Re-run with `--text-only` or use "
                "`evaluate_usaaao_qa.py` for multimodal-capable models."
            )
        if not examples:
            print("No examples matched the requested filters.", file=sys.stderr)
            return 1

        print(f"Loaded {len(examples)} examples.")
        print(f"Dataset root: {dataset_root}")
        print(f"Generation model: {args.model}")
        print(f"Judge enabled: {args.judge}")

        generator = LargeModelGenerator(
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
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
            prediction = generator.generate(build_generation_prompt(example))

            row: Dict[str, Any] = {
                "id": example.record_id,
                "year": example.year,
                "question_length": example.question_length,
                "has_image": False,
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
