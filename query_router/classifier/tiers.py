"""Tier taxonomy and model-roster mapping — single source of truth.

Deliberately has zero heavy imports. model.py and predict.py both pull tier
metadata from here; predict.py in particular must NOT end up importing
transformers just to get these two dicts (see predict.py's docstring).
"""
NUM_TIERS = 4

TIER_LABELS = {0: "trivial", 1: "simple", 2: "moderate", 3: "complex"}

# Informational only — the model a downstream router maps to this tier. This
# classifier does not call these models; it only reports the tier + this label.
TIER_MODELS = {
    0: "Qwen3.5:4b",
    1: "Gemini 2.5 Flash-Lite",
    2: "Gemini 2.5 Flash",
    3: "Gemini 2.5 Flash Pro",
}
