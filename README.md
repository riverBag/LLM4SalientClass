<div align="center">
  <h1 align="center">Assessing Language Models for Salient Class Identification</h1>
</div>


<div align="center">
    <a href="https://github.com/riverBag/LLM4SalientClass">
      <img src="https://img.shields.io/badge/Code-GitHub-2d333b?style=flat-square&logo=github" alt="github">
    </a>
    <a href="https://arxiv.org/abs/2606.21629">
      <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv">
    </a>
    <hr>
</div>

Code review requires reviewers to understand the core intent of code changes,
which becomes difficult when a commit modifies multiple classes. In such
commits, one or more primarily modified classes, referred to as salient classes,
may induce modifications in other classes. This study evaluates whether language
models (LMs) can identify salient classes directly from commit diffs without
feature engineering, dependency graph construction, or supervised training.

This replication package contains the ApacheJavaCM dataset, the prompt
templates, the scripts used to generate and run LM tasks, the reproduced
Most-Modified-Class baseline implementation, and the raw LM prediction files
used in the paper.

## ⚙️ Usage as a Replication Package

First, create the Python environment:

```sh
conda env create -f environment.yml
conda activate llm4coreentities
```

## 📁 Package Structure

- `ApacheJavaCM.jsonl`: the released ApacheJavaCM dataset. Each JSON object is
  retained through the `class_labels` field; later derived fields from the
  working copy were removed.
- `environment.yml`: Conda environment used for the Python scripts.
- `scripts/task_generator.py`: converts ApacheJavaCM records into chat-style LM
  tasks for salient class identification.
- `scripts/batch_commit_generator.py`: runs generated tasks through
  OpenAI-compatible LM APIs.
- `scripts/three_prompts/`: zero-shot, few-shot, and chain-of-thought prompt
  templates.
- `baselines/most_modified_lines/`: source code for the Most-Modified-Class
  baseline and its metric evaluator.
- `experiment_results/`: raw LM prediction files used in the experiments.

## 🔁 Reproduction

We provide the dataset, prompts, scripts, and raw LM prediction files needed to
reproduce the main experimental workflow.

### 📦 Dataset

ApacheJavaCM is derived from ApacheCM and contains 7,911 Java commits and
25,914 labeled classes. Each record contains commit metadata, the changed files,
the unified code diff, and `class_labels`, where each class is labeled as either
`positive` (salient) or `negative` (non-salient).

The dataset file is:

```sh
ApacheJavaCM.jsonl
```

### 🧩 Generating LM Tasks

The three prompts used in the paper are available in `scripts/three_prompts/`.
To generate task files for all prompt settings:

```sh
python scripts/task_generator.py \
  --input ApacheJavaCM.jsonl \
  --output generated_tasks \
  --mode all \
  --prompt-dir scripts/three_prompts
```

This produces one JSONL task file per prompt setting. Each task contains a
`task_id`, the prompt mode, and chat-style `messages`.

### 🚀 Running LM Inference

After configuring the API credentials described above, run a task file or a
directory of task files:

```sh
python scripts/batch_commit_generator.py \
  --input generated_tasks \
  --output generated_predictions/deepseek \
  --model deepseek-v3.2-128k \
  --workers 20
```

The script supports resuming interrupted runs by skipping task IDs that already
have completed predictions in the output file. Use `--overwrite` to rerun all
tasks from scratch.

### 📏 Most-Modified-Class Baseline

The Most-Modified-Class baseline selects the class with the largest number of
modified lines in a commit as the salient class. To reproduce its predictions
and metrics:

```sh
python baselines/most_modified_lines/baseline_modified_lines.py \
  --input ApacheJavaCM.jsonl \
  --output baselines/most_modified_lines/baseline_modified_lines_predictions.jsonl \
  --ties list

python baselines/most_modified_lines/evaluate_metrics.py \
  --input ApacheJavaCM.jsonl \
  --pred baselines/most_modified_lines/baseline_modified_lines_predictions.jsonl \
  --report baselines/most_modified_lines/baseline_modified_lines_metrics.json \
  --label "Most-Modified-Class"
```

## 📊 Experiment Results

The raw LM outputs are stored under:

```sh
experiment_results/
```

Each model directory contains the original prediction files for the three prompt
settings:

- `deepseek/predictions_zero_shot.jsonl`
- `deepseek/predictions_few_shot.jsonl`
- `deepseek/predictions_cot.jsonl`
- `gpt-5.4/predictions_zero_shot.jsonl`
- `gpt-5.4/predictions_few_shot.jsonl`
- `gpt-5.4/predictions_cot.jsonl`
- `qwen3.5_9b/predictions_zero_shot.jsonl`
- `qwen3.5_9b/predictions_few_shot_new.jsonl`
- `qwen3.5_9b/predictions_cot.jsonl`

These are the original model responses. Unified prediction files, diagnostic
files, and metric summaries are intentionally not included in this result
directory.

## ❓ RQ1: What Is the Performance of LMs for Salient Class Identification?

RQ1 compares LM-based salient class identification with the baselines used in
the study. This package keeps the ApacheJavaCM dataset, LM outputs, and
Most-Modified-Class reproduction code. The relevant released files are:

- Dataset: `ApacheJavaCM.jsonl`
- Most-Modified-Class code: `baselines/most_modified_lines/`
- LM results: `experiment_results/`

## 🧪 RQ2: What Is the Impact of LM Selection and Prompting Strategy?

RQ2 evaluates three LMs under three prompt settings:

- GPT-5.4
- DeepSeek-V3.2
- Qwen3.5-9B

The prompt templates are located in `scripts/three_prompts/`, and the original
prediction files for each model-prompt combination are located in
`experiment_results/`.

## 📈 RQ3: How Does Commit Complexity Affect LM Performance?

RQ3 analyzes LM behavior across commit characteristics, including the total
number of classes, the number of salient classes, and diff token length. The
analysis is based on the same ApacheJavaCM dataset and the raw model prediction
files released in `experiment_results/`.

## 🔍 RQ4: What Factors Contribute to Erroneous LM Predictions?

RQ4 performs qualitative analysis of LM failure cases. The released raw
prediction files preserve each model response, including chain-of-thought
outputs where applicable, so that failure cases can be inspected against the
corresponding commit diffs and class labels in `ApacheJavaCM.jsonl`.

## 📝 Citation

```bibtex
@article{Xiong2026Salient,
  author = {Xiong, Bo and Cai, Chaoran and Wang, Chong and Liang, Peng},
  title = {{Assessing Language Models for Salient Class Identification}},
  journal={arXiv preprint arXiv:2606.21629},
  year={2026}
}
```
