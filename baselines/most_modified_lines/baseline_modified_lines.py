#!/usr/bin/env python3
"""Baseline: pick the class with the most modified lines as the "core class".

Input  : a JSONL file where each record contains a unified-diff string in `diff`.
Output : a JSONL file where each record has the original `repo`, `commit_sha`,
         `diff` plus three new fields produced by this baseline:
             - core_class            : str | null
             - modified_class_count  : int
             - modified_lines_per_class : { class_name -> int }

Class attribution rule (see diff_structure.md for full discussion):
    1. When entering a new file block (line starting with "diff --git "), set the
       file_default_class from the new path (`+++ b/<path>`); class name is the
       Java file's stem if it starts with an upper-case letter.
    2. While inside the file, update current_class whenever a `class/interface/
       enum/record <Name>` declaration is observed (in any kind of line, including
       hunk headers, so that inner-class modifications are credited correctly).
    3. Every `+` or `-` line (excluding `+++ ` and `--- ` headers) contributes 1
       to current_class's modified-line counter; if current_class is null
       (e.g. pom.xml), the line is skipped.

Tie-breaking for core_class: the class encountered first in the diff wins.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PACKAGE_ROOT / "ApacheJavaCM.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "baseline_modified_lines_predictions.jsonl"


CLASS_DECL_RE = re.compile(
    r"\b(?:class|interface|enum|record)\s+([A-Z][a-zA-Z0-9_]*)\b"
)
JAVA_FILE_RE = re.compile(r"[ab]/(.+)")


def simple_class_from_path(path_text: str) -> str | None:
    """Return the Java class name implied by a file path, or None."""
    filename = Path(path_text).name
    if "." not in filename:
        return None
    stem = filename.rsplit(".", 1)[0]
    if stem and stem[0].isupper():
        return stem
    return None


def parse_diff(
    diff_text: str,
) -> tuple["OrderedDict[str, int]", list[str]]:
    """Return (modified_lines_per_class, tied_classes_with_max_lines).

    The second element is always a list. It contains every class that ties for
    the maximum modified-line count, in the order they were first observed in
    the diff. Empty list ↔ no Java class was modified at all.

    The caller chooses how to render this in the output JSONL — see
    `format_core_class()` and the `--ties` CLI flag.
    """
    per_class_lines: "OrderedDict[str, int]" = OrderedDict()

    current_class: str | None = None
    file_default_class: str | None = None

    def register(cls: str | None) -> None:
        if cls and cls not in per_class_lines:
            per_class_lines[cls] = 0

    for raw_line in diff_text.splitlines():
        # File-block header
        if raw_line.startswith("diff --git "):
            current_class = None
            file_default_class = None
            parts = raw_line.split()
            if len(parts) >= 4:
                m = JAVA_FILE_RE.match(parts[3])
                if m:
                    cls = simple_class_from_path(m.group(1))
                    file_default_class = cls
                    current_class = cls
                    register(cls)
            continue

        # +++ b/<path>: re-confirm the file-default class (handles renames).
        # For +++ /dev/null (file deletion) we keep the class learned from the
        # `diff --git` header so deleted lines still get attributed.
        if raw_line.startswith("+++ "):
            target = raw_line[4:].strip()
            if target != "/dev/null":
                m = JAVA_FILE_RE.match(target)
                if m:
                    cls = simple_class_from_path(m.group(1))
                    if cls:
                        file_default_class = cls
                        current_class = cls
                        register(cls)
            continue

        # Other file-block prelude lines we ignore for counting
        if raw_line.startswith("--- ") or raw_line.startswith("index "):
            continue

        # Hunk header: only used for class detection in trailing context
        if raw_line.startswith("@@"):
            m = CLASS_DECL_RE.search(raw_line)
            if m:
                current_class = m.group(1)
                register(current_class)
            continue

        if not raw_line:
            continue

        first = raw_line[0]
        if first == "\\":  # e.g. "\ No newline at end of file"
            continue

        content = raw_line[1:] if first in "+- " else raw_line

        # Update current class on any class declaration observed
        m = CLASS_DECL_RE.search(content)
        if m:
            current_class = m.group(1)
            register(current_class)

        if first in ("+", "-"):
            target_class = current_class if current_class else file_default_class
            if target_class:
                per_class_lines[target_class] = (
                    per_class_lines.get(target_class, 0) + 1
                )

    if not per_class_lines:
        return per_class_lines, []

    # Drop classes that ended up with zero modified lines (registered only via
    # context). They are not "modified" in the line-counting sense.
    nonzero = OrderedDict((k, v) for k, v in per_class_lines.items() if v > 0)
    if not nonzero:
        return nonzero, []

    max_count = max(nonzero.values())
    tied_classes = [cls for cls, count in nonzero.items() if count == max_count]
    return nonzero, tied_classes


def iter_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fin:
        for raw_line in fin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield {"__invalid__": True, "__raw__": raw_line[:200]}


def format_core_class(tied: list[str], ties_mode: str):
    """Convert the list of tied classes into the value written to JSONL.

    Modes:
        "first": always pick the first tied class -> str | None  (legacy)
        "list" : if >=2 tied classes, output the list; otherwise the single
                 tied class as a string. None when no class found.       (Plan A)
    """
    if not tied:
        return None
    if ties_mode == "first":
        return tied[0]
    if ties_mode == "list":
        return tied if len(tied) >= 2 else tied[0]
    raise ValueError(f"unknown ties_mode: {ties_mode}")


def process(
    input_path: Path,
    output_path: Path,
    ties_mode: str = "first",
) -> dict:
    total = 0
    written = 0
    invalid = 0
    no_class = 0
    multi_class = 0
    tied_commits = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for item in iter_records(input_path):
            total += 1
            if item.get("__invalid__"):
                invalid += 1
                continue

            diff_text = item.get("diff", "") or ""
            if not isinstance(diff_text, str):
                diff_text = ""

            per_class, tied = parse_diff(diff_text)

            if not tied:
                no_class += 1
            elif len(tied) >= 2:
                tied_commits += 1
            if len(per_class) >= 2:
                multi_class += 1

            output = {
                "repo": item.get("repo", ""),
                "commit_sha": item.get("commit_sha", ""),
                "diff": diff_text,
                "core_class": format_core_class(tied, ties_mode),
                "modified_class_count": len(per_class),
                "modified_lines_per_class": dict(per_class),
            }
            fout.write(json.dumps(output, ensure_ascii=False) + "\n")
            written += 1

    return {
        "total": total,
        "written": written,
        "invalid": invalid,
        "no_class": no_class,
        "multi_class": multi_class,
        "tied_commits": tied_commits,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Baseline: select the class with the most modified lines in each "
            "code diff as the 'core class'."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input JSONL path",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--ties",
        choices=("first", "list"),
        default="first",
        help=(
            "How to render `core_class` when multiple classes tie for the "
            "maximum modified-line count. 'first' (legacy): always pick the "
            "first encountered class as a string. 'list' (Plan A): output a "
            "JSON list of all tied classes; single winner stays a string."
        ),
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 2

    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = process(input_path, output_path, ties_mode=args.ties)
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"ties mode             : {args.ties}")
    print(f"Total non-empty lines : {stats['total']}")
    print(f"Written predictions   : {stats['written']}")
    print(f"Invalid JSON skipped  : {stats['invalid']}")
    print(f"Records w/ no class   : {stats['no_class']}")
    print(f"Records w/ >=2 classes: {stats['multi_class']}")
    print(f"Records w/ tied max   : {stats['tied_commits']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
