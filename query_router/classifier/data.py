"""Dataset loading for the query-complexity classifier — DESIGN.md §5."""
import json
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

NUM_TIERS = 4


@dataclass
class Example:
    prompt_id: str
    prompt_text: str
    tier_label: int
    split: str


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_splits(dataset_path):
    splits = {"train": [], "val": [], "test": []}
    for r in load_jsonl(dataset_path):
        splits[r["split"]].append(
            Example(r["prompt_id"], r["prompt_text"], r["tier_label"], r["split"])
        )
    return splits


def load_boundary_eval(path):
    return load_jsonl(path)


class QueryComplexityDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length: int = 64):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.prompt_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(ex.tier_label, dtype=torch.long)
        return item
