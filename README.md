# Query Complexity ML Classifier

Lightweight text classifier that scores prompt complexity in milliseconds and maps each query to an LLM tier. Built as the routing gatekeeper for an intelligent LLM routing system — send trivial work to a cheap/local model and reserve frontier models for hard reasoning.

See [BRD.md](BRD.md) for business requirements and [DESIGN.md](DESIGN.md) for the full technical design.

## Why this exists

Routing every prompt to a premium LLM wastes cost and latency on simple lookups, formatting, and boilerplate. A small classifier in front of the LLM stack can cut inference spend substantially while keeping quality close to always-using-frontier (adjacent misses only).

**In scope:** single-turn text prompts, complexity classification, tier → model mapping labels.  
**Out of scope:** multi-turn dialogue routing, non-text modalities, training destination LLMs.

## Complexity tiers

| Tier | Label | Characteristics | Destination model |
| --- | --- | --- | --- |
| 0 | Trivial | Lookups, basic Q&A, formatting, single-step arithmetic | Qwen3.5:4b (local / cheap) |
| 1 | Simple | Short reasoning, basic domain knowledge, boilerplate code | Gemini 2.5 Flash-Lite |
| 2 | Moderate | Multi-factor reasoning, system/API design trade-offs | Gemini 2.5 Flash |
| 3 | Complex | Niche expertise, proofs, concurrent/formal systems | Gemini 2.5 Flash Pro |

Success metric (BRD): **adjacent accuracy > 90%** — if the model is wrong, it should be off by at most one tier.

Labeling rules live in [`data/labeling_rubric.md`](data/labeling_rubric.md).

## Repository layout

```
query_router/
  classifier/
    tiers.py          # Tier labels + destination model names
    model.py          # MiniLM classifier builder
    data.py           # JSONL dataset loading
    losses.py         # Cost-sensitive cross-entropy
    train.py          # Fine-tuning entrypoint
    evaluate.py       # Test + gold-boundary evaluation
    export_onnx.py    # ONNX export + INT8 quantization
    predict.py        # Fast ONNX Runtime inference (no full transformers import)
data/
  synthetic_dataset.jsonl   # Labeled training set (320 examples)
  gold_boundary_eval.jsonl  # Ambiguous boundary stress set
  labeling_rubric.md
scripts/
  generate_synthetic_data.py  # Scale-up scaffold (wire your LLM client)
tests/
  test_losses.py
BRD.md
DESIGN.md
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # optional
```

### Train

```bash
python -m query_router.classifier.train \
  --data data/synthetic_dataset.jsonl \
  --output models/classifier
```

Checkpoint selection uses **validation loss** with early stopping (`--patience`, default 4).

### Evaluate

```bash
python -m query_router.classifier.evaluate \
  --model-dir models/classifier \
  --data data/synthetic_dataset.jsonl \
  --boundary-data data/gold_boundary_eval.jsonl
```

Reports exact accuracy, adjacent accuracy, confusion matrix, and gold-boundary pass rate. Writes `models/classifier/eval_results.json`.

### Export (ONNX + INT8)

```bash
python -m query_router.classifier.export_onnx \
  --model-dir models/classifier \
  --output models/classifier_onnx/model.onnx
```

Produces `model.onnx` and `model.int8.onnx` (plus tokenizer files) under `models/classifier_onnx/`.

### Predict

Prediction uses **ONNX Runtime** and the lightweight `tokenizers` package (avoids a slow full `transformers` import on some Windows setups):

```bash
python -m query_router.classifier.predict "What is the capital of France?"
```

Example response:

```python
{
  "tier": 0,
  "label": "trivial",
  "confidence": 0.91,
  "model": "Qwen3.5:4b"
}
```

Programmatic use:

```python
from query_router.classifier.predict import predict

result = predict("Design a multi-tenant billing schema")
# result["tier"], result["label"], result["confidence"], result["model"]
```

## Model & training details

| Item | Choice |
| --- | --- |
| Backbone | `sentence-transformers/all-MiniLM-L6-v2` (~22M params) |
| Objective | Cost-sensitive cross-entropy (penalizes far tier misses more) |
| Serving | ONNX + dynamic INT8 quantization, CPU via ONNX Runtime |
| Latency budget | Classification + routing decision &lt; 50ms (design target) |

This package **classifies and labels the destination tier**; it does not call Gemini/Qwen. An external router should consume `tier` / `model` and perform the actual LLM call + fallback cascade (see DESIGN.md §2.7 / §7).

## Dataset

Current seed set (`data/synthetic_dataset.jsonl`):

- **320** labeled prompts, balanced **80 per tier**
- Splits: **224 train / 48 val / 48 test**
- Domains include coding, realtime, scientific, mathematical, debugging, blogs, grammar, and more

Schema:

```text
prompt_id, prompt_text, tier_label (0–3), source, annotator_id,
generation_model, domain_tag, split (train|val|test)
```

To scale further, implement `call_llm` in `scripts/generate_synthetic_data.py`, generate candidates, review against the rubric, then merge into the training JSONL.

## Tests

```bash
pytest
```

## Configuration

Optional environment values (see `.env.example`): Hugging Face token, base model name, device, confidence threshold. Training/prediction CLIs also accept explicit flags (`--data`, `--output`, `--model-dir`, etc.).

Trained weights and ONNX artifacts under `models/` are gitignored — train and export locally (or in CI) before predicting.

## Docs

| Doc | Contents |
| --- | --- |
| [BRD.md](BRD.md) | Business objectives, NFRs, scope |
| [DESIGN.md](DESIGN.md) | Architecture, prerequisites, routing, rollout |
| [data/labeling_rubric.md](data/labeling_rubric.md) | Tier decision heuristic + anchors |

## License

See [LICENSE](LICENSE).
