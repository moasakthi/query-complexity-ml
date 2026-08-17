"""Classifier architecture — DESIGN.md §2.6 decision #4 / §6.1."""
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .tiers import NUM_TIERS, TIER_LABELS

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # MiniLM-L6-scale, ~22M params


def build_model_and_tokenizer(model_name: str = MODEL_NAME, num_labels: int = NUM_TIERS):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=TIER_LABELS,
        label2id={v: k for k, v in TIER_LABELS.items()},
    )
    return model, tokenizer
