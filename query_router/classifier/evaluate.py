"""Evaluation metrics — adjacent accuracy is the BRD §5 success metric, not raw accuracy."""
import argparse
import json
from collections import Counter

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import QueryComplexityDataset, load_boundary_eval, load_splits


def exact_accuracy(preds, labels):
    if not labels:
        return 0.0
    return sum(p == l for p, l in zip(preds, labels)) / len(labels)


def adjacent_accuracy(preds, labels, tolerance: int = 1):
    if not labels:
        return 0.0
    return sum(abs(p - l) <= tolerance for p, l in zip(preds, labels)) / len(labels)


def confusion_counts(preds, labels, num_classes: int = 4):
    counts = Counter(zip(labels, preds))
    return [[counts.get((t, p), 0) for p in range(num_classes)] for t in range(num_classes)]


def boundary_pass_rate(preds, boundary_records, tolerance: int = 1):
    """Pass if the prediction lands within `tolerance` of either tier in the boundary pair."""
    if not boundary_records:
        return 0.0
    passed = 0
    for pred, rec in zip(preds, boundary_records):
        lo, hi = rec["expected_tier_range"]
        if (lo - tolerance) <= pred <= (hi + tolerance):
            passed += 1
    return passed / len(boundary_records)


def _predict(model, loader):
    preds, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_labels = batch.pop("labels")
            outputs = model(**batch)
            preds.extend(outputs.logits.argmax(dim=-1).tolist())
            labels.extend(batch_labels.tolist())
    return preds, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/classifier")
    parser.add_argument("--data", default="data/synthetic_dataset.jsonl")
    parser.add_argument("--boundary-data", default="data/gold_boundary_eval.jsonl")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    splits = load_splits(args.data)
    test_loader = DataLoader(QueryComplexityDataset(splits["test"], tokenizer), batch_size=16)
    preds, labels = _predict(model, test_loader)

    results = {
        "test_exact_accuracy": exact_accuracy(preds, labels),
        "test_adjacent_accuracy": adjacent_accuracy(preds, labels),
        "test_confusion_matrix": confusion_counts(preds, labels),
    }

    boundary_records = load_boundary_eval(args.boundary_data)
    boundary_preds = []
    model.eval()
    with torch.no_grad():
        for rec in boundary_records:
            enc = tokenizer(rec["prompt_text"], truncation=True, padding=True, return_tensors="pt")
            logits = model(**enc).logits
            boundary_preds.append(int(logits.argmax(dim=-1).item()))
    results["boundary_pass_rate"] = boundary_pass_rate(boundary_preds, boundary_records)

    print(json.dumps(results, indent=2))
    with open(f"{args.model_dir}/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
