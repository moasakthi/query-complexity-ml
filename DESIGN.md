## Design Document: Intelligent LLM Routing System

**Project Name:** Query Complexity ML Classifier
**Document Version:** 1.0
**Source Requirements:** BRD.md v1.0

---

### 1. Purpose & Scope

This document translates the BRD's business requirements into a concrete technical design: system architecture, component responsibilities, data and model design, API contracts, and — as requested — the prerequisites that must be satisfied before implementation begins.

It covers the same in-scope boundary as the BRD (single-turn, text-only routing) and does not re-litigate business justification, which is covered there.

---

### 2. Prerequisites

Nothing below should be started until these are in place — several are blocking dependencies for the rest of the design.

#### 2.1 Organizational / Access Prerequisites

| # | Prerequisite | Why it blocks the project |
| --- | --- | --- |
| 1 | Sign-off on the 4-tier taxonomy and its example prompts (BRD §4.1) | The taxonomy is the labeling rubric; changing it after labeling starts invalidates the dataset. |
| 2 | ~~Confirmed list of destination LLM endpoints per tier~~ — **Resolved, see §2.7**: Qwen3.5-4B / Gemini 2.5 Flash-Lite / Flash / Flash Pro. Still open: named owner/team per model and provisioned API credentials (§2.3). | Routing config and fallback cascade cannot be defined without concrete endpoints. |
| 3 | Confirmation that all destination LLMs share a common request/response schema (BRD §6 assumption) | If false, the router needs a per-model adapter layer — a materially different (larger) design. |
| 4 | Security/compliance approval for where the classifier is allowed to run (which network zone, whether logging prompts is permitted) | Drives the entire deployment architecture (§9) and the governance layer (§4.4). |
| 5 | Defined keyword list / user-role list for rule-based override routing (BRD §4.3) | Needed to build the override engine and to size how much traffic bypasses the ML path. |
| 6 | Budget/quota approval for synthetic data generation via a frontier LLM (if historical data is insufficient) | Synthetic generation at scale has real API cost; needs sign-off before dataset work starts. |

#### 2.2 Data Prerequisites

- Export (or access to) historical prompt logs, if they exist, with any necessary PII scrubbing already defined.
- A labeling guideline document mapping example prompts to each of the 4 tiers, reviewed by whoever owns model quality — this is the arbiter for both human labels and synthetic-generation prompts.
- A held-out "gold" evaluation set (recommend 300–500 hand-labeled examples, doubly-labeled for inter-annotator agreement) that is never used in training — required to trust the adjacent-accuracy number reported later.
- Decision on data residency/retention (how long labeled prompts are stored, where).

#### 2.3 Infrastructure & Environment Prerequisites

- Training compute: at least one GPU (a single mid-range GPU is sufficient for fine-tuning a small transformer; this is not a large-model training job).
- Inference hosting: a co-located, low-latency serving environment within the same network segment as the routing gateway — the <50ms budget (BRD §5) effectively rules out an inference call that leaves the secure perimeter or crosses an extra network hop.
- CPU vs. GPU inference decision: for a model this small, CPU inference with quantization is usually sufficient and avoids provisioning GPU capacity for production serving — needs to be validated against the actual latency budget once a candidate model is chosen (§6.1).
- A feature flag / config-push mechanism so tier→endpoint mappings and override lists can change without redeploying the classifier service.
- **Qwen3.5-4B serving infrastructure** (§2.7, Tier 0 + override destination): a self-hosted generative model at this size needs its own serving stack — a GPU-backed inference server (e.g., vLLM or TGI) for real-time generation latency, distinct from the classifier's CPU-only serving. Sizing this (single GPU is likely enough for 4B at moderate concurrency) is a prerequisite to validate before launch, since it's new infra this design didn't previously require.
- **Gemini API access** for Tiers 1–3 (§2.7): provisioned Google Cloud/AI Studio project, API keys or service-account credentials, and confirmed rate limits/quota for Gemini 2.5 Flash-Lite, Flash, and Flash Pro sufficient for expected production volume — quota exhaustion on any one tier effectively becomes an outage of that tier and should be monitored, not discovered in production.

#### 2.4 Tooling & Software Prerequisites

- Python ML stack: PyTorch, Hugging Face `transformers` + `datasets`.
- Experiment tracking (e.g., MLflow or Weights & Biases) to compare candidate architectures/losses.
- Inference optimization toolchain: ONNX + ONNX Runtime (or equivalent) for quantization and low-latency serving.
- A load-testing tool (e.g., locust/k6) to validate the <50ms p99 latency requirement under realistic concurrency, not just single-request latency.

#### 2.5 Team / Skill Prerequisites

- ML engineer familiar with transformer fine-tuning and (ideally) ordinal/cost-sensitive classification — the adjacent-accuracy metric is non-standard and shapes the training objective (§6.2).
- Backend/platform engineer to own the routing gateway, fallback cascade, and circuit-breaker/availability logic (BRD §5).
- Security reviewer to sign off on the governance layer before any real prompt data flows through the system.

#### 2.6 Architect Decisions (Resolved)

These were flagged in v1.0 of this document as open judgment calls. As architect, I'm deciding them now rather than blocking on sign-off — each includes the rationale so it can be revisited if evidence later contradicts it.

| # | Decision | Choice | Rationale |
| --- | --- | --- | --- |
| 1 | Confidence threshold for cascade fallback | **0.65** | Starting operating point; empirically re-tune post-launch by plotting cascade-rate vs. adjacent-accuracy on the gold set (§6.3). Not a permanent constant — treat as a tunable config value, not a hardcoded one. |
| 2 | Ordinal-aware loss function | **Cost-sensitive weighted cross-entropy**, weight ∝ \|pred_tier − true_tier\| | Ships faster than full ordinal regression (CORAL/CORN), is easy to reason about and debug, and directly penalizes the exact failure mode the BRD cares about (far misses). Escalate to CORAL only if this doesn't clear 90% adjacent accuracy on the gold set. |
| 3 | Tier 0/1 endpoint topology | **Two distinct endpoints** (tier 0 → fast/cheap pool, tier 1 → mid-tier pool) | BRD §4.1's table is the more specific spec; §4.2's grouping of 0+1 is treated as a simplification it supersedes. Finer-grained routing captures more of the cost-reduction objective in §2 of the BRD. |
| 4 | Base classifier architecture | **MiniLM-L6-scale distilled transformer** (~22M params), INT8-quantized via ONNX Runtime, CPU-served, co-located with the gateway | Safest latency margin under the 50ms budget once tokenization overhead is included. DeBERTa-v3-small stays the fallback if this architecture can't hit the accuracy bar. |
| 5 | Security/deployment zone | Classifier + override engine deployed **co-located with the routing gateway inside the same private subnet/VPC** | No raw prompt should need to leave the trusted boundary before a routing decision is made — satisfies BRD §4.3 without waiting on a separate infra decision. |
| 6 | Prompt logging/retention default | **No raw-prompt logging by default.** Only aggregated metrics (tier, confidence, latency, override-hit) are logged. A 1% redacted, access-controlled QA sample is retained 30 days for retraining/audit. | Defaults to the more conservative/compliant posture; can be loosened per-org if legal/security explicitly approves broader retention. |
| 7 | Starter override list (keywords/roles) | Seeded with common PII/PHI terms (ssn, social security, credit card, patient id, medical record, passport number), secrets (api_key, password, private_key), and roles (legal, hr, compliance, executive) | Gives the override engine a functional default on day one instead of blocking on security drafting a list from scratch; security should review and extend it, not originate it from nothing. See `data/labeling_rubric.md` note and router config in §7.2. |

**Genuinely non-decidable here:** the actual destination LLM endpoint URLs/credentials for the fast/mid/strong/frontier pools are organization infrastructure, not a design choice — §7.2 defines the adapter contract and placeholder identifiers so the rest of the system isn't blocked on knowing them in advance.

#### 2.7 Model Roster Binding

The org's available model roster is now known, so the four tier→model placeholders are bound to concrete models instead of generic pool names:

| Tier | Complexity | Bound Model | Hosting | Notes |
| --- | --- | --- | --- | --- |
| 0 | Trivial | **Qwen3.5-4B** | Self-hosted SLM, in-house | Matches BRD §4.1's "fast/cheap **local** model" wording exactly — trivial traffic never needs to leave the secure boundary. |
| 1 | Simple | **Gemini 2.5 Flash-Lite** | Google API | Cheapest hosted tier; sufficient for moderate reasoning without frontier cost. |
| 2 | Moderate | **Gemini 2.5 Flash** | Google API | Also serves as the global classifier-unavailable fail-open default (§4.3) and tier 1's fallback — the best "safe generalist" choice when the router can't decide. |
| 3 | Complex | **Gemini 2.5 Flash Pro** | Google API | Frontier tier for niche/multi-step queries. |

Naming used here is exactly what was provided; treat it as shorthand for whichever official Gemini 2.5 Flash-Lite / Flash / Pro SKU your Google Cloud or AI Studio account exposes.

**Fallback cascade, concretely (supersedes the placeholder chain in §7.2's prior draft):** each tier cascades to the next-higher tier on primary failure, since that's the only direction that doesn't sacrifice more quality than necessary:
- Tier 0 (Qwen3.5-4B) → Tier 1 (Gemini 2.5 Flash-Lite)
- Tier 1 (Flash-Lite) → Tier 2 (Gemini 2.5 Flash)
- Tier 2 (Flash) → Tier 3 (Gemini 2.5 Flash Pro)
- Tier 3 (Flash Pro) → **no higher tier exists in the roster.** Retry the same model once with backoff; if still failing, soft-degrade to Tier 2 (Gemini 2.5 Flash) and mark the response `degraded: true` in telemetry so degraded-quality answers to complex queries are visible, not silent.

**Override/hard-route destination (BRD §4.3):** bound to **Qwen3.5-4B, self-hosted**, since it's the only model in the roster with a guaranteed no-external-egress path — exactly the property the override mechanism exists to guarantee for sensitive prompts. This is a real trade-off, not a free win: a 4B SLM will produce lower-quality answers than the Gemini tiers for genuinely complex sensitive queries (e.g., a hard legal question that also matches the "legal" role override). If that quality gap proves unacceptable in practice, the fix is not to override to a public Gemini endpoint — it's to get a private/VPC-isolated Gemini deployment (e.g., Vertex AI with VPC Service Controls and a data-processing agreement guaranteeing prompts aren't logged/used for training) so sensitive traffic can reach a stronger model without leaving a contractually-guaranteed boundary. Flagged as a follow-up decision for whoever owns the Google Cloud relationship, not something resolvable from this design alone.

---

### 3. System Architecture

```
                 ┌─────────────────────────────────────────────┐
                 │              Routing Gateway                 │
 User Prompt ───▶│                                               │
                 │  ┌───────────────┐      ┌──────────────────┐ │
                 │  │ Rule-Based    │ hit  │  Hard Route to    │ │
                 │  │ Override Check│─────▶│  Qwen3.5-4B       │─┼──▶ Qwen3.5-4B (self-hosted, no egress)
                 │  │ (keywords,    │      │  (self-hosted)    │ │
                 │  │  user role)   │      └──────────────────┘ │
                 │  └───────┬───────┘                            │
                 │       no match                                │
                 │          ▼                                    │
                 │  ┌───────────────┐                            │
                 │  │  ML Complexity │──── < 50ms ───┐            │
                 │  │  Classifier    │               │            │
                 │  └───────┬───────┘               │            │
                 │          ▼                        │            │
                 │  ┌───────────────┐                │            │
                 │  │ Routing        │  confidence /  │            │
                 │  │ Decision Logic │◀── failure ────┘            │
                 │  └───────┬───────┘  triggers fallback           │
                 │          │                                      │
                 └──────────┼──────────────────────────────────────┘
                            ▼
     ┌───────────────┬────────────────────┬───────────────┬─────────────────────┐
     │   Tier 0       │      Tier 1        │    Tier 2      │       Tier 3        │
     │  Qwen3.5-4B    │ Gemini 2.5         │ Gemini 2.5     │ Gemini 2.5          │
     │ (self-hosted)  │ Flash-Lite (API)   │ Flash (API)    │ Flash Pro (API)     │
     └───────────────┴────────────────────┴───────────────┴─────────────────────┘
          ▲ fallback cascades one tier to the right on failure (§2.7); Tier 3
            retries itself, then soft-degrades to Tier 2 with a `degraded: true` flag.
```

Everything left of the dashed boundary (override check, classifier, decision logic) must execute inside the secure environment per BRD §4.3 — no prompt leaves that boundary until a routing decision has been made. Note that only Tier 0 (Qwen3.5-4B) and the override path are actually self-hosted; Tiers 1–3 are external Google API calls made *after* the routing decision, so the boundary claim applies to the decision itself, not to where the eventual answer gets generated (see §4.4).

#### Component list

| Component | Responsibility |
| --- | --- |
| Rule-Based Override Engine | Evaluates prompt/user-role against a config-driven list; short-circuits ML classification when matched. |
| ML Complexity Classifier | Fine-tuned small transformer; returns `(tier, confidence)` in <50ms. |
| Routing Decision Logic | Maps tier → endpoint; applies confidence threshold; owns the fallback cascade. |
| Fallback/Cascade Handler | On primary-model timeout/error/low-confidence, retries against an approved backup model. |
| Security & Governance Layer | Enforces in-boundary execution, controls what (if anything) gets logged, applies redaction. |
| Observability | Latency, tier distribution, confidence distribution, override-hit-rate, cascade-trigger-rate. |

---

### 4. Component Design

#### 4.1 Rule-Based Override Engine

- Config-driven (not code-driven) so security/compliance can update keyword and role lists without a deploy.
- Runs **before** the ML classifier, since a match should skip ML inference entirely (both for correctness and to avoid burning the latency budget).
- Suggested config shape: ordered list of `{match_type: keyword|regex|role, pattern, destination_model, reason}` — first match wins, evaluated top to bottom for deterministic precedence.

#### 4.2 ML Complexity Classifier

- Input: single prompt string (no conversation history — out of scope per BRD §3).
- Output: tier (0–3) + confidence score.
- Must be a self-contained, versioned artifact (model weights + tokenizer + label map) so it can be rolled back independently of the gateway code.
- See §6 for model/training design.

#### 4.3 Routing & Fallback Gateway

- Stateless service; tier→endpoint mapping is external config, not hardcoded, so new model tiers or endpoint swaps don't require a code change.
- Fallback cascade (BRD §4.2) triggers on: (a) primary endpoint timeout, (b) primary endpoint error/5xx, (c) classifier confidence below the agreed threshold (§2.6). Concrete cascade chain is bound in §2.7: each tier escalates to the next model up (Qwen3.5-4B → Flash-Lite → Flash → Flash Pro), with Flash Pro retrying itself before soft-degrading to Flash.
- Availability requirement (BRD §5): if the classifier service itself is unreachable or errors, the gateway must fail open to a pre-defined default model rather than blocking the request — this must be a hard timeout + circuit breaker around the classifier call, not a retry loop, to protect the latency budget. Bound to **Gemini 2.5 Flash** (§2.7) as the "safe generalist" default — cheap enough not to defeat the cost objective if sustained, capable enough not to badly fail hard queries while the classifier is down.

#### 4.4 Security & Governance Layer

- Classifier and override engine run inside the org's secured network boundary; only the routing decision (not necessarily the raw prompt) needs to cross into less-trusted infra, depending on where destination models are hosted.
- Logging policy must be explicit: whether raw prompts are logged for retraining/QA, and if so, retention period and access controls (ties to §2.2 data residency decision).
- **With the roster bound in §2.7, this is no longer theoretical:** Tier 0 and override traffic stay fully in-house (Qwen3.5-4B, self-hosted), but Tiers 1–3 send the prompt to Google's Gemini API — an actual data egress point, not a hypothetical one. Before any real traffic flows: confirm the org's Google Cloud/AI Studio agreement's data-processing terms (whether prompts are logged or used for training on Google's side), and treat that agreement — not this design — as the actual control satisfying BRD §4.3 for tiers 1–3. If the org's data sensitivity bar is higher than the standard API terms allow, that's the trigger for a Vertex AI private-endpoint / VPC-SC setup, per the note in §2.7.

#### 4.5 Observability & Monitoring

Minimum metrics to emit from day one:
- p50/p95/p99 latency for the classification+routing step (validates the <50ms NFR continuously, not just at launch).
- Tier distribution over time (detects prompt-mix drift).
- Override-engine hit rate (detects rule list going stale or over/under-triggering).
- Cascade/fallback trigger rate, broken out by cause (timeout vs. error vs. low-confidence).
- Adjacent-accuracy on a rolling sample of production traffic that gets spot-checked/labeled (proxy for the offline eval metric holding up in production).

---

### 5. Data Design

#### 5.1 Label taxonomy

Mirrors BRD §4.1 exactly — this table is the single source of truth for labeling:

| Tier | Label | Characteristics | Destination |
| --- | --- | --- | --- |
| 0 | Trivial | Simple lookups, basic Q&A, formatting requests | Fast/cheap model |
| 1 | Simple | Moderate reasoning, basic domain knowledge | Mid-tier model |
| 2 | Moderate | Complex reasoning, deep knowledge required | Strong model |
| 3 | Complex | Niche expertise, multi-step logic, complex code generation | Frontier model |

#### 5.2 Data sources & synthetic generation

1. **Historical prompts** (if available and approved for use, §2.2): sample across time/user segments to avoid recency or user-population bias.
2. **Synthetic generation**: use a strong LLM prompted with the tier rubric + few-shot examples per tier to generate diverse candidate prompts, across varied domains (coding, writing, factual Q&A, analysis, etc.) so the classifier doesn't overfit to topic instead of complexity.
3. **Human labeling pass**: every synthetic example still needs a human (or a second, different strong LLM as an adjudicator) spot-check — LLM-generated "tier 2" prompts frequently drift toward tier 1 or 3 without review.

Recommend a target mix of roughly balanced classes (not necessarily matching production's real class distribution) for training, while keeping the held-out gold eval set representative of true production distribution — otherwise the offline accuracy number won't predict production behavior.

#### 5.3 Dataset schema

```
prompt_id, prompt_text, tier_label (0-3), source (historical|synthetic), 
annotator_id, generation_model (if synthetic), domain_tag, split (train|val|test)
```

#### 5.4 Split strategy & labeling QA

- Stratify train/val/test by tier so all splits have representative class balance.
- Compute inter-annotator agreement (e.g., Cohen's kappa) on a double-labeled subset before trusting the full label set — low agreement here is a signal the rubric (§2.1 item 1) is ambiguous and needs revision before scaling up labeling.

---

### 6. Model Design

#### 6.1 Candidate architectures & latency budget

The <50ms NFR (BRD §5) is the dominant constraint on model choice — it rules out anything at the scale of a full BERT-base/DeBERTa-large served over a network hop. Candidates, smallest/fastest to largest:

| Model | Approx. params | Notes |
| --- | --- | --- |
| MiniLM / DistilBERT (distilled) | 22M–66M | Best latency headroom; recommended starting point. |
| DeBERTa-v3-small | ~140M | BRD explicitly mentions this; stronger but tighter latency margin, especially on CPU. |
| DeBERTa-v3-base | ~184M+ | Likely too slow for <50ms on CPU without aggressive optimization; only worth it if accuracy targets aren't met by smaller models. |

**Decision (§2.6 #4):** MiniLM-L6-scale, quantized to INT8, served via ONNX Runtime, co-located with the gateway. Benchmark actual p99 latency (including tokenization, not just the forward pass) before committing — tokenization overhead is often underestimated in latency budgets. If accuracy falls short, step up to DeBERTa-v3-small before considering anything larger.

#### 6.2 Training objective (adjacent-accuracy aware)

BRD §5 defines success as **adjacent accuracy**, not raw accuracy — a tier-0-labeled-as-tier-3 error is much worse than tier-2-as-tier-3. Plain categorical cross-entropy treats all misclassifications equally and doesn't directly optimize for this. Options, in increasing order of complexity:

1. **Cost-sensitive cross-entropy**: weight the loss by `|predicted_tier - true_tier|` so distant errors are penalized more. Simplest change, works with a standard classification head.
2. **Ordinal regression (e.g., CORAL/CORN)**: reframes the 4-class problem as a set of ordered binary thresholds, which naturally respects tier ordering. More correct for this metric, more implementation complexity.
3. **Regression + bucketing**: predict a continuous complexity score and bucket into tiers at inference; simple but loses calibrated per-class confidence, which the routing logic needs (§4.3).

**Decision (§2.6 #2):** start with (1), cost-sensitive weighted cross-entropy, as the primary training objective. Escalate to (2) only if adjacent accuracy on the gold eval set falls short of the 90% target — this keeps the training pipeline simple until there's evidence it needs to be more sophisticated.

#### 6.3 Confidence calibration

- The routing logic needs a trustworthy confidence score to decide when to cascade (BRD §4.2). Raw softmax probabilities from a fine-tuned transformer are often overconfident — consider temperature scaling on the validation set as a lightweight calibration step.
- The confidence threshold (§2.6 #1, default 0.65) should be re-tuned against the gold eval set by plotting cascade-trigger-rate vs. adjacent-accuracy-if-not-cascaded, and picking the operating point that meets the 90% target without cascading so much traffic that cost-reduction goals (BRD §2) are undermined.

#### 6.4 Inference optimization

- Export to ONNX, apply dynamic or static INT8 quantization, benchmark accuracy delta (quantization can shift tier boundaries near decision thresholds — re-check adjacent accuracy post-quantization, not just pre-quantization).
- Batch size of 1 (real-time single-request routing) — optimize for single-request latency, not throughput-oriented batching.

---

### 7. API & Config Contracts

#### 7.1 Classifier service (internal) API

```
POST /classify
Request:  { "prompt": string, "user_role": string (optional) }
Response: { "tier": 0-3, "confidence": float, "model_version": string, "latency_ms": float }
```

#### 7.2 Router config schema (illustrative)

```yaml
# Starter list per §2.6 #7 — security team owns extending/pruning this.
# Routes to the in-house model per §2.7: overrides exist to guarantee no external egress.
overrides:
  - match_type: role
    pattern: "legal|hr|compliance|executive"
    destination: qwen3.5-4b-selfhosted
  - match_type: keyword
    pattern: "ssn|social security|credit card|patient id|medical record|passport number"
    destination: qwen3.5-4b-selfhosted
  - match_type: keyword
    pattern: "api_key|password|private_key"
    destination: qwen3.5-4b-selfhosted

# Model roster bound per §2.7. Fallback cascades one tier up; Tier 3 has no
# higher tier, so it retries itself before soft-degrading to Tier 2.
tiers:
  0: { destination: qwen3.5-4b-selfhosted,  fallback: gemini-2.5-flash-lite }
  1: { destination: gemini-2.5-flash-lite,  fallback: gemini-2.5-flash }
  2: { destination: gemini-2.5-flash,       fallback: gemini-2.5-flash-pro }
  3: { destination: gemini-2.5-flash-pro,   fallback: gemini-2.5-flash-pro, retry_then_degrade_to: gemini-2.5-flash }

confidence_threshold: 0.65   # decided default, §2.6 #1 — re-tune empirically post-launch
classifier_timeout_ms: 50
classifier_unavailable_default: gemini-2.5-flash   # §2.7 — safe generalist when the router itself is blind
```

---

### 8. Non-Functional Requirements Traceability

| BRD Requirement | Design Decision |
| --- | --- |
| <50ms classification+routing latency | Small distilled model, INT8/ONNX, co-located inference, single-request optimization (§6.1, §6.4) |
| >90% adjacent accuracy, never off by >1 tier | Cost-sensitive/ordinal training objective (§6.2), gold eval set for validation (§2.2) |
| High availability, fail-open on classifier downtime | Circuit breaker + hard timeout in the gateway defaulting to Gemini 2.5 Flash (§4.3, §2.7) |
| No data leakage prior to routing | Classifier + override engine run inside secure boundary; override traffic routes to self-hosted Qwen3.5-4B for no-egress guarantee (§4.4, §2.7) |
| Rule-based bypass for sensitive prompts/roles | Override engine, config-driven, evaluated before ML classifier, hard-routed to the in-house model (§4.1, §2.7) |
| Fallback cascade on failure/timeout/low-confidence | Fallback handler cascading Qwen3.5-4B → Flash-Lite → Flash → Flash Pro, with Flash Pro retrying itself before soft-degrading (§4.3, §4.5, §2.7) |

---

### 9. Deployment & Rollout Plan

**Environments:** dev (offline eval only) → staging (shadow mode against real traffic, no live routing impact) → canary (small % of live traffic) → GA.

**Phased rollout:**
1. **Shadow mode**: classifier runs on production traffic in parallel, logs its decision, but the existing single-LLM path still serves the response. Used to validate real-world adjacent accuracy and latency before it can affect users.
2. **Canary**: route a small, low-risk traffic slice (e.g., internal users or a specific low-stakes surface) through the full pipeline.
3. **Staged rollout**: expand traffic percentage while watching the observability metrics in §4.5, with an explicit rollback trigger (e.g., adjacent accuracy drop or latency SLO breach).
4. **GA**: full traffic, override engine and fallback cascade fully live.

---

### 10. Testing & Validation Strategy

- **Offline**: adjacent accuracy and confusion matrix on the held-out gold set (§2.2); latency benchmarking under realistic concurrent load, not just single-request.
- **Shadow-mode validation**: compare classifier's would-be routing decisions against a sample manually reviewed for correctness, to catch gaps between synthetic training data and real traffic patterns.
- **Fallback/chaos testing**: deliberately fail the classifier service and each destination model endpoint in staging to confirm the fail-open and cascade paths behave as designed (BRD §5 availability requirement is only as good as this test).
- **Regression testing on override rules**: unit tests per keyword/role rule to prevent silent rule-list drift.

---

### 11. Risks & Mitigations

| Risk | Mitigation |
| --- | --- |
| Synthetic training data doesn't reflect real prompt distribution | Shadow-mode validation against real traffic before any live routing impact (§9). |
| Quantization degrades accuracy near tier boundaries | Re-validate adjacent accuracy post-quantization, not just pre-quantization (§6.4). |
| Confidence threshold miscalibrated → excessive cascading erodes cost savings | Tune threshold against gold set with explicit cascade-rate vs. accuracy trade-off plot (§6.3). |
| Destination LLM API schemas diverge over time | Adapter layer isolated in the gateway config, not hardcoded per-model logic, to contain blast radius (§7.2). |
| Rule-based override list goes stale (misses new sensitive keywords) | Regular review cadence owned by security, plus regression tests (§10). |
| Override traffic (sensitive prompts) gets materially worse answers than tiers 1–3 because it's hard-routed to a 4B self-hosted model | Acceptable as the default given the current roster's only no-egress option (§2.7); if quality proves inadequate for legal/HR use cases, escalate to procuring a private/VPC-isolated Gemini deployment rather than routing sensitive traffic to the public API. |
| Gemini API quota/outage on one tier looks like a routing bug but is actually a vendor-side limit | Monitor per-tier quota usage and API error rates separately from classifier health (§2.3, §4.5) so on-call can distinguish "our classifier is wrong" from "Google throttled us." |

---

### 12. Decisions Log

All items previously tracked here as open questions were resolved by architect decision — see §2.6 for the full table and rationale. The model roster itself is now bound to concrete models (§2.7): Qwen3.5-4B (Tier 0 + override), Gemini 2.5 Flash-Lite (Tier 1), Gemini 2.5 Flash (Tier 2 + classifier-down default), Gemini 2.5 Flash Pro (Tier 3). The only remaining item is not a design decision but environment/contractual work: provisioning the actual API credentials and confirming Google's data-processing terms are acceptable for Tiers 1–3 (§2.3, §4.4).

---

### 13. Data Artifacts

To unblock training rather than leave it as a dependency on future data collection, a synthetic seed dataset was generated directly against the rubric in §5.1:

| Artifact | Location | Contents |
| --- | --- | --- |
| Labeling rubric | `data/labeling_rubric.md` | Expanded tier definitions with decision heuristics, used both for human labeling and as the generation prompt template. |
| Seed training dataset | `data/synthetic_dataset.jsonl` | 160 labeled prompts (40/tier), stratified 70/15/15 into train/val/test, spanning ~20 domains (coding, legal, finance, science, systems design, etc.) so the classifier learns complexity rather than topic. |
| Gold boundary eval set | `data/gold_boundary_eval.jsonl` | 24 deliberately ambiguous prompts (8 each at the 0/1, 1/2, and 2/3 boundaries) — the specific stress test for the BRD's adjacent-accuracy requirement, since generic synthetic data rarely produces true boundary cases on its own. |
| Scale-up pipeline | `scripts/generate_synthetic_data.py` | Provider-agnostic scaffold for generating further batches at production scale via an LLM API, following the same rubric and schema. |

**Important caveat:** 160 examples is a pilot/bootstrap set — enough to validate the end-to-end pipeline (data → training → eval) and run initial experiments, but not enough for a production-quality fine-tune. Production training should scale this to at least several hundred examples per tier (low thousands preferred) using the generation script, with the same human-review step called out in §5.2 before trusting any newly generated batch.
