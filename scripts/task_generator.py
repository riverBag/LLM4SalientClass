"""
Generate LLM tasks for identifying salient/core classes from Java commit diffs.

Each input JSONL record is expected to contain a ``diff`` field. The selected
prompt template is filled with that diff and written as a chat-style task that
can be consumed by ``batch_commit_generator.py``.
"""

import argparse
import json
import os
from typing import Dict, Iterable, List, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROMPT_DIR = os.path.join(SCRIPT_DIR, "three_prompts")

PROMPT_FILES = {
    "zero_shot": "zero_shot_salient_class_prompt.txt",
    "few_shot": "few_shot_salient_class_prompt.txt",
    "cot": "cot_salient_class_prompt.txt",
}

MODE_ALIASES = {
    "zero": "zero_shot",
    "zero_shot": "zero_shot",
    "zeroshot": "zero_shot",
    "few": "few_shot",
    "few_shot": "few_shot",
    "fewshot": "few_shot",
    "cot": "cot",
    "chain_of_thought": "cot",
    "chain-of-thought": "cot",
}


def normalize_mode(mode: str) -> str:
    """Normalizes user-facing mode names to internal prompt keys."""
    mode_key = mode.strip().lower().replace("-", "_")
    if mode_key == "all":
        return "all"
    if mode_key not in MODE_ALIASES:
        valid_modes = ", ".join(["zero_shot", "few_shot", "cot", "all"])
        raise ValueError(f"Unknown mode: {mode}. Valid modes: {valid_modes}")
    return MODE_ALIASES[mode_key]


def iter_modes(mode: str) -> Iterable[str]:
    """Returns one or all supported prompt modes."""
    normalized = normalize_mode(mode)
    if normalized == "all":
        return PROMPT_FILES.keys()
    return [normalized]


def load_prompt_template(mode: str, prompt_dir: str) -> Tuple[str, str]:
    """
    Loads a prompt template and splits it into system and user prompt sections.

    Prompt files are expected to contain:
    - ``System Prompt:``
    - ``User Prompt:``
    """
    prompt_filename = PROMPT_FILES[mode]
    prompt_path = os.path.join(prompt_dir, prompt_filename)
    with open(prompt_path, "r", encoding="utf-8") as f_prompt:
        template = f_prompt.read()

    system_marker = "System Prompt:"
    user_marker = "User Prompt:"
    if system_marker not in template or user_marker not in template:
        raise ValueError(
            f"Prompt template {prompt_path} must contain both "
            f"{system_marker!r} and {user_marker!r} sections."
        )

    system_part, _, user_part = template.partition(user_marker)
    system_prompt = system_part.replace(system_marker, "", 1).strip()
    user_prompt = user_part.strip()
    return system_prompt, user_prompt


def build_messages(system_prompt: str, user_prompt: str, query_diff: str) -> List[Dict[str, str]]:
    """Builds chat messages by replacing only the diff placeholder."""
    filled_user_prompt = user_prompt.replace("{QUERY_DIFF}", query_diff)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": filled_user_prompt},
    ]


def get_task_id(data: dict) -> str:
    """Uses the dataset's commit_sha as the task id."""
    commit_sha = data.get("commit_sha")
    if not commit_sha:
        raise ValueError("missing commit_sha field")
    return str(commit_sha)[:7]


def build_task(data: dict, line_number: int, mode: str, system_prompt: str, user_prompt: str) -> dict:
    """Converts a dataset record into a single LLM task."""
    query_diff = data.get("diff")
    if not query_diff:
        raise ValueError("missing diff field")

    task = {
        "task_id": get_task_id(data),
        "mode": mode,
        "messages": build_messages(system_prompt, user_prompt, query_diff),
    }
    return task


def process_single_file(dataset_path: str, tasks_path: str, mode: str, prompt_dir: str = DEFAULT_PROMPT_DIR):
    """Processes one dataset JSONL file and writes one task JSONL file."""
    normalized_mode = normalize_mode(mode)
    if normalized_mode == "all":
        raise ValueError("process_single_file expects a single mode, not 'all'.")

    system_prompt, user_prompt = load_prompt_template(normalized_mode, prompt_dir)
    tasks = []
    skipped = 0

    print(f"Processing dataset from {dataset_path} with mode={normalized_mode}...")
    try:
        with open(dataset_path, "r", encoding="utf-8") as f_dataset:
            for line_number, line in enumerate(f_dataset, start=1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    task = build_task(data, line_number, normalized_mode, system_prompt, user_prompt)
                except (json.JSONDecodeError, ValueError) as e:
                    skipped += 1
                    print(f"Warning: Skipping line {line_number}: {e}")
                    continue
                tasks.append(task)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}. Cannot proceed.")
        return

    output_dir = os.path.dirname(tasks_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(tasks_path, "w", encoding="utf-8") as f_tasks:
            for task in tasks:
                f_tasks.write(json.dumps(task, ensure_ascii=False) + "\n")
        print(f"Successfully generated {len(tasks)} tasks at: {tasks_path}")
        if skipped:
            print(f"Skipped {skipped} malformed or incomplete records.")
    except IOError as e:
        print(f"Error writing tasks to {tasks_path}: {e}")


def make_output_filename(input_filename: str, mode: str) -> str:
    """Builds a mode-specific tasks filename for batch outputs."""
    if input_filename.startswith("tasks_"):
        base_name = input_filename[len("tasks_") :]
    else:
        base_name = input_filename
    return f"tasks_{mode}_{base_name}"


def create_tasks(
    input_path: str,
    output_path: str,
    mode: str = "zero_shot",
    prompt_dir: str = DEFAULT_PROMPT_DIR,
):
    """
    Generates task JSONL files for salient/core class identification.

    Args:
        input_path: JSONL dataset file or directory containing JSONL files.
        output_path: Output JSONL file for a single mode, or output directory
            when input_path is a directory or mode='all'.
        mode: One of zero_shot, few_shot, cot, or all.
        prompt_dir: Directory containing the three prompt template files.
    """
    modes = list(iter_modes(mode))

    if os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)
        print(f"Batch processing files from {input_path} to {output_path}...")
        for filename in sorted(os.listdir(input_path)):
            if not filename.endswith(".jsonl"):
                continue
            file_input_path = os.path.join(input_path, filename)
            for prompt_mode in modes:
                file_output_path = os.path.join(
                    output_path,
                    make_output_filename(filename, prompt_mode),
                )
                process_single_file(file_input_path, file_output_path, prompt_mode, prompt_dir)
        return

    if len(modes) > 1:
        os.makedirs(output_path, exist_ok=True)
        input_filename = os.path.basename(input_path)
        for prompt_mode in modes:
            file_output_path = os.path.join(output_path, make_output_filename(input_filename, prompt_mode))
            process_single_file(input_path, file_output_path, prompt_mode, prompt_dir)
        return

    process_single_file(input_path, output_path, modes[0], prompt_dir)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate LLM tasks for salient/core class identification."
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        required=True,
        help="JSONL dataset file or directory containing JSONL files.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        required=True,
        help="Output JSONL file, or output directory when mode='all' or input is a directory.",
    )
    parser.add_argument(
        "--mode",
        default="zero_shot",
        help="Prompt mode: zero_shot, few_shot, cot, or all.",
    )
    parser.add_argument(
        "--prompt-dir",
        default=DEFAULT_PROMPT_DIR,
        help="Directory containing the prompt template files.",
    )
    args = parser.parse_args()

    create_tasks(
        input_path=args.input_path,
        output_path=args.output_path,
        mode=args.mode,
        prompt_dir=args.prompt_dir,
    )


if __name__ == "__main__":
    main()
