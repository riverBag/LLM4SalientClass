#!/usr/bin/env python3
"""Evaluate the modified-lines baseline against the dataset's `class_labels`
using the metrics defined in Section 4.2 of:

    Ren et al. "Graph-Based Salient Class Classification in Commits", QRS 2024.

Metrics implemented (formulas from the paper, equations 3-8):
    PosPre      = TP / (TP + FP)
    NegPre      = TN / (TN + FN)
    PosRecall   = TP / (TP + FN)
    NegRecall   = TN / (TN + FP)
    Accuracy    = (TP + TN) / (TP + FP + TN + FN)
    CmtAccuracy = CorrectCommits / TotalCommits
    F1          = 2 * PosPre * PosRecall / (PosPre + PosRecall)
    MCC         = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

Conventions:
    - Salient (positive) class    = class_labels[c] == "positive"
    - Non-salient (negative) class = class_labels[c] == "negative"
    - The baseline `core_class` field can be:
          * a single string (legacy/'first' mode) -> predicted positive iff
            class_name == core_class
          * a list of strings (Plan A 'list' mode for ties) -> predicted
            positive iff class_name in core_class
          * null -> every class predicted negative
    - The class universe per commit is `class_labels`. Out-of-universe
      predictions (baseline picked an inner class not present in class_labels)
      are still recorded as a counter, but do not change TP/FP/TN/FN tallies
      because we evaluate over the ground-truth class set.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PACKAGE_ROOT / "ApacheJavaCM.jsonl"
DEFAULT_PRED = BASELINE_DIR / "baseline_modified_lines_predictions.jsonl"
DEFAULT_REPORT = BASELINE_DIR / "baseline_modified_lines_metrics.json"


def load_predictions(pred_path: Path) -> dict[tuple[str, str], dict]:
    preds: dict[tuple[str, str], dict] = {}
    with pred_path.open("r", encoding="utf-8") as fin:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            preds[(rec.get("repo", ""), rec.get("commit_sha", ""))] = rec
    return preds


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def predicted_positive_set(core_class) -> set[str]:
    """Return the set of classes the baseline predicts as positive."""
    if core_class is None:
        return set()
    if isinstance(core_class, list):
        return set(core_class)
    return {core_class}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute GBSCI Section 4.2 metrics for the baseline."
    )
    parser.add_argument(
        "--input", default=str(DEFAULT_INPUT), help="Ground-truth JSONL"
    )
    parser.add_argument(
        "--pred", default=str(DEFAULT_PRED), help="Baseline prediction JSONL"
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
        help="Output JSON report path",
    )
    parser.add_argument(
        "--label",
        default="baseline (modified-lines, ties=first)",
        help="Label printed at the top of the report",
    )
    args = parser.parse_args()

    INPUT = Path(args.input)
    PRED = Path(args.pred)
    REPORT = Path(args.report)
    if not INPUT.exists():
        print(f"ERROR: input not found: {INPUT}", file=sys.stderr)
        sys.exit(2)
    if not PRED.exists():
        print(f"ERROR: prediction file not found: {PRED}", file=sys.stderr)
        sys.exit(2)

    preds = load_predictions(PRED)

    tp = fp = tn = fn = 0
    correct_commits = 0
    total_commits = 0
    commits_skipped_no_label = 0
    commits_missing_pred = 0
    out_of_universe_pred = 0           # any predicted positive not in class_labels
    pred_none_commits = 0              # core_class is None
    multi_positive_commits = 0         # commits with >=2 positives in GT
    tied_pred_commits = 0              # core_class is a list (Plan A)

    with INPUT.open("r", encoding="utf-8") as fin:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            class_labels = item.get("class_labels") or {}
            if not isinstance(class_labels, dict) or not class_labels:
                commits_skipped_no_label += 1
                continue

            key = (item.get("repo", ""), item.get("commit_sha", ""))
            pred = preds.get(key)
            if pred is None:
                commits_missing_pred += 1
                continue

            core_class_field = pred.get("core_class")  # str | list[str] | None
            pred_pos_set = predicted_positive_set(core_class_field)

            if core_class_field is None:
                pred_none_commits += 1
            elif isinstance(core_class_field, list):
                tied_pred_commits += 1
            if pred_pos_set and not (pred_pos_set & set(class_labels)):
                out_of_universe_pred += 1

            n_pos = sum(1 for v in class_labels.values() if v == "positive")
            if n_pos >= 2:
                multi_positive_commits += 1

            commit_correct = True
            total_commits += 1

            for class_name, label in class_labels.items():
                is_pos = (label == "positive")
                pred_pos = (class_name in pred_pos_set)

                if is_pos and pred_pos:
                    tp += 1
                elif (not is_pos) and pred_pos:
                    fp += 1
                    commit_correct = False
                elif (not is_pos) and (not pred_pos):
                    tn += 1
                else:  # is_pos and not pred_pos
                    fn += 1
                    commit_correct = False

            if commit_correct:
                correct_commits += 1

    # ---- Metrics ----
    pos_pre = safe_div(tp, tp + fp)
    neg_pre = safe_div(tn, tn + fn)
    pos_rec = safe_div(tp, tp + fn)
    neg_rec = safe_div(tn, tn + fp)
    accuracy = safe_div(tp + tn, tp + fp + tn + fn)
    cmt_accuracy = safe_div(correct_commits, total_commits)
    f1 = safe_div(2 * pos_pre * pos_rec, pos_pre + pos_rec)
    denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / math.sqrt(denom_sq) if denom_sq > 0 else 0.0

    # ---- Print ----
    print("=" * 64)
    print(args.label)
    print(
        "Reference paper: Ren et al., 'Graph-Based Salient Class "
        "Classification in Commits', QRS 2024 (Section 4.2)"
    )
    print("=" * 64)
    print()
    print("Commit-level statistics")
    print(f"  total commits evaluated  : {total_commits}")
    print(f"  commits w/ no class_labels: {commits_skipped_no_label}")
    print(f"  commits w/ no prediction  : {commits_missing_pred}")
    print(f"  baseline core_class=None  : {pred_none_commits}")
    print(f"  baseline core_class list  : {tied_pred_commits}")
    print(f"  baseline picked OOC class : {out_of_universe_pred}")
    print(f"  multi-positive commits    : {multi_positive_commits}")
    print()
    print("Class-level confusion matrix (sum over all commits)")
    print(f"  TP = {tp:>8d}    FN = {fn:>8d}")
    print(f"  FP = {fp:>8d}    TN = {tn:>8d}")
    total = tp + fp + tn + fn
    print(f"  total class-samples      : {total}")
    print(f"  positives (salient)      : {tp + fn}")
    print(f"  negatives (non-salient)  : {tn + fp}")
    print()
    print("GBSCI Section 4.2 metrics")
    print(f"  PosPre      = {pos_pre*100:6.2f}%   ({tp}/{tp+fp})")
    print(f"  NegPre      = {neg_pre*100:6.2f}%   ({tn}/{tn+fn})")
    print(f"  PosRecall   = {pos_rec*100:6.2f}%   ({tp}/{tp+fn})")
    print(f"  NegRecall   = {neg_rec*100:6.2f}%   ({tn}/{tn+fp})")
    print(f"  Accuracy    = {accuracy*100:6.2f}%   ({tp+tn}/{total})")
    print(
        f"  CmtAccuracy = {cmt_accuracy*100:6.2f}%   "
        f"({correct_commits}/{total_commits})"
    )
    print(f"  F1          = {f1:.4f}")
    print(f"  MCC         = {mcc:.4f}")
    print()
    print("(GBSCI paper Table VII reports for ISC: Acc 85.18%, CmtAcc 34.88%, "
          "MCC 0.390, F1 0.476; for their R-GCN model: Acc 88.23%, CmtAcc "
          "57.24%, MCC 0.533, F1 0.605. Numbers above are this baseline on "
          "ccr_ISC, NOT directly comparable to those tables; different "
          "dataset and different model.)")

    # ---- Persist ----
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(
            {
                "label": args.label,
                "pred_path": str(PRED),
                "input_path": str(INPUT),
                "total_commits": total_commits,
                "commits_missing_pred": commits_missing_pred,
                "pred_none_commits": pred_none_commits,
                "tied_pred_commits": tied_pred_commits,
                "out_of_universe_pred": out_of_universe_pred,
                "multi_positive_commits": multi_positive_commits,
                "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
                "metrics": {
                    "PosPre": pos_pre,
                    "NegPre": neg_pre,
                    "PosRecall": pos_rec,
                    "NegRecall": neg_rec,
                    "Accuracy": accuracy,
                    "CmtAccuracy": cmt_accuracy,
                    "F1": f1,
                    "MCC": mcc,
                },
                "correct_commits": correct_commits,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote report: {REPORT}")


if __name__ == "__main__":
    main()
