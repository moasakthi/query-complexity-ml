"""Standalone prediction over the exported classifier artifact — DESIGN.md §6/§13.

Loads the ONNX model once (cached at module level) and predicts a complexity
tier for a single prompt. Returns the tier only — routing to an actual model
is handled by an external system, not this function.

Deliberately imports `tokenizers` (the fast tokenizer backend) instead of the
full `transformers` package — on some Windows setups `import transformers`
takes 40-90+ seconds (antivirus/EDR scanning file access during its
importlib.metadata dependency-version checks), which otherwise dwarfs actual
inference (~10-30ms). Training/evaluation still use full `transformers`
since that cost is paid once per run, not once per prediction.
"""
import os
import sys

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .tiers import TIER_LABELS, TIER_MODELS

DEFAULT_MODEL_PATH = "models/classifier_onnx/model.int8.onnx"
MAX_LENGTH = 64

_session = None
_tokenizer = None
_loaded_from = None


def _load(model_path: str):
    global _session, _tokenizer, _loaded_from
    if _session is None or _loaded_from != model_path:
        tokenizer_path = os.path.join(os.path.dirname(model_path), "tokenizer.json")
        _tokenizer = Tokenizer.from_file(tokenizer_path)
        pad_id = _tokenizer.token_to_id("[PAD]") or 0
        _tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]", length=MAX_LENGTH)
        _tokenizer.enable_truncation(max_length=MAX_LENGTH)
        _session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        _loaded_from = model_path
    return _session, _tokenizer


def run_inference(session, tokenizer, text: str):
    """Given a loaded ONNX session + fast tokenizer, return (tier, confidence) for `text`."""
    enc = tokenizer.encode(text)
    inputs = {
        "input_ids": np.array([enc.ids], dtype=np.int64),
        "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
    }
    logits = session.run(["logits"], inputs)[0][0]
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    tier = int(np.argmax(probs))
    return tier, float(probs[tier])


def predict(text: str, model_path: str = DEFAULT_MODEL_PATH) -> dict:
    session, tokenizer = _load(model_path)
    tier, confidence = run_inference(session, tokenizer, text)
    return {
        "tier": tier,
        "label": TIER_LABELS[tier],
        "confidence": confidence,
        "model": TIER_MODELS[tier],
    }


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What is the capital of France?"
    print(predict(query))
