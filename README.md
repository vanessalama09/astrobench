# AstroBench

AstroBench contains code for evaluating language models on open-ended astronomy and astrophysics question answering. The current benchmark pipeline supports free-response questions derived from publicly available USA Astronomy and Astrophysics Organization (USAAAO) olympiad-style materials, with evaluation across text-only and image-linked examples.

This repository accompanies the workshop paper:

**Rethinking Domain Specialization for Open-Ended Scientific Reasoning in Astronomy Language Models**

## Overview

The project evaluates open-weight and API-served language models on open-ended scientific reasoning in astronomy. The benchmark is designed to study when domain-specific fine-tuning remains valuable in the presence of stronger general-purpose models.

The paper focuses on a 2017--2026 free-response subset containing:

- 300 total free-response examples
- 204 text-only examples
- 96 image-linked examples
- 301 first-round multiple-choice questions reserved for a future track

The evaluation pipeline supports:

- local open-weight model inference
- multi-GPU inference for large models
- OpenAI-compatible API endpoint inference
- reference-based metrics
- LLM-as-judge scoring
- second-judge sensitivity checks
- merged result tables
- paper figure generation

## Repository Structure

```text
astrobench/
├── scripts/
│   ├── evaluate/      # model evaluation and rejudging scripts
│   ├── analysis/      # result merging and aggregation
│   ├── figures/       # figure-generation scripts
│   └── runs/          # example run scripts
├── paper/             # LaTeX source and paper figure references
├── configs/           # example local configuration files
├── data/              # placeholder for dataset release information
├── requirements.txt
└── README.md
```

## Installation

Create and activate a Python environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For local GPU inference, install a CUDA-compatible PyTorch build appropriate for your system.

## Configuration

Do not place API keys directly in source files. Use environment variables.

For OpenAI-compatible API endpoints:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://your-endpoint/v1"
```

For the second-judge experiments using the AmSC/i2 OpenAI-compatible endpoint:

```bash
export AMSC_I2_API_KEY="..."
export JUDGE_OPENAI_API_KEY="$AMSC_I2_API_KEY"
export JUDGE_OPENAI_BASE_URL="<judge-url>"
export SECOND_JUDGE_MODEL="claude-sonnet-4-6"
```

A blank example configuration file is provided in:

```text
configs/config.example.py
```

## Running Evaluations

### Local model evaluation

For standard local models:

```bash
python scripts/evaluate/evaluate_usaaao_qa.py \
  --model MODEL_NAME_OR_PATH \
  --dataset-root data/usaaao_qa_local \
  --years 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 \
  --text-only \
  --judge
```

### Large multi-GPU model evaluation

For large 70B-class models:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/evaluate/evaluate_usaaao_qa_large.py \
  --model MODEL_NAME_OR_PATH \
  --dataset-root data/usaaao_qa_local \
  --years 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 \
  --text-only \
  --judge
```

### API endpoint evaluation

For OpenAI-compatible endpoints:

```bash
python scripts/evaluate/evaluate_usaaao_qa_endpoint.py \
  --provider openai-chat \
  --endpoint-url "$OPENAI_BASE_URL/chat/completions" \
  --api-key-env OPENAI_API_KEY \
  --model MODEL_NAME \
  --dataset-root data/usaaao_qa_local \
  --years 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 \
  --text-only \
  --judge
```

## Rejudging Existing Predictions

To re-score saved model predictions with a second judge:

```bash
python scripts/evaluate/rejudge_usaaao_csv.py \
  --input-csv path/to/model_predictions.csv \
  --output-dir benchmark_results/judge_sensitivity/model_name \
  --judge-model "$SECOND_JUDGE_MODEL" \
  --run-stem model_name_judged_by_claude \
  --text-only \
  --resume
```

The shell scripts in `scripts/runs/` provide examples for the representative second-judge runs used in the paper.

## Merging Results

After model runs finish, merge per-model outputs into analysis-ready tables:

```bash
python scripts/analysis/merge_usaaao_results.py
```

This creates aggregate CSV/JSON files for downstream analysis and figure generation.

## Generating Figures

Paper figures can be regenerated from merged result tables using:

```bash
python scripts/figures/plot_usaaao_data_overview.py
python scripts/figures/plot_usaaao_metric_disagreement.py
python scripts/figures/plot_usaaao_bootstrap_leaderboard.py
python scripts/figures/plot_usaaao_yearly_robustness.py
python scripts/figures/plot_usaaao_question_length_robustness.py
python scripts/figures/plot_usaaao_model_modality_comparison.py
```

Most figure scripts write PDF, PNG, SVG, and summary JSON outputs.

## Data Availability

The full AI-ready USAAAO questionnaire, including free-response records, image assets, and the first-round multiple-choice track, is planned for release on Hugging Face after documentation and licensing review.

This repository currently provides the code and analysis scripts needed to reproduce tables and figures from released or supplied result CSV files. Full regeneration of benchmark outputs requires access to the relevant model checkpoints, API credentials, and GPU resources.

## Artifact Notes

The artifact associated with the paper includes:

- evaluation scripts
- result-merging scripts
- figure-generation scripts
- run scripts for local and API-served models
- paper source files

Large model checkpoints, API credentials, private endpoint metadata, and raw benchmark outputs are not included.

## Citation

If you use this code, please cite the accompanying paper:

```bibtex
@inproceedings{lama2026astrobench,
  title = {Rethinking Domain Specialization for Open-Ended Scientific Reasoning in Astronomy Language Models},
  author = {Lama, Vanessa and Das, Sanjay and Herron, Emily and Ting, Yuan-Sen and de Haan, Tijmen and Yin, Junqi and Wang, Feiyi and Ghosal, Tirthankar},
  year = {2026}
}
```

## License

License information will be added before public release.
