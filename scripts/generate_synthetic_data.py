"""
Scale-up generator for the query-complexity training set.

Produces additional labeled prompts per tier, following the same rubric and
schema as data/synthetic_dataset.jsonl (see data/labeling_rubric.md). This is
the production scale-up path referenced in DESIGN.md Sections 5.2 and 13 --
the 160-example seed set was hand-authored to bootstrap the pipeline; real
training needs this run at volume against an actual LLM API.

Provider-agnostic by design: plug in whichever LLM client your org has
already approved (BRD Section 6 assumes a strong LLM is available for this).
Nothing here calls a live API -- `call_llm` is the one function to implement.

Usage:
    python generate_synthetic_data.py --tier 2 --count 200 --output data/batch_tier2.jsonl
"""

import argparse
import json
import random

TIERS = {
    0: {
        "label": "Trivial",
        "characteristics": "Simple lookups, basic Q&A, formatting requests, single-step arithmetic, single-fact recall.",
        "anchors": [
            "What is the capital of Japan?",
            "Convert 5 kilometers to miles.",
            "List the planets in our solar system.",
        ],
    },
    1: {
        "label": "Simple",
        "characteristics": "Moderate reasoning, basic domain knowledge, short explanations, boilerplate code.",
        "anchors": [
            "Explain the difference between weather and climate.",
            "Write a Python function that reverses a string.",
            "What are three benefits of regular exercise?",
        ],
    },
    2: {
        "label": "Moderate",
        "characteristics": "Complex reasoning, deep knowledge, multi-factor trade-off analysis, non-trivial system/code design.",
        "anchors": [
            "Design a database schema for a multi-tenant SaaS billing system.",
            "Compare GraphQL versus REST for a high-traffic public API.",
            "Explain the CAP theorem and how it should influence database design.",
        ],
    },
    3: {
        "label": "Complex",
        "characteristics": "Niche expertise, multi-step logic/proofs, complex code generation, cross-domain synthesis.",
        "anchors": [
            "Design a distributed rate limiter with exactly-once semantics across data centers, with a correctness argument.",
            "Derive the Black-Scholes-Merton PDE from first principles.",
            "Write a formally verified smart contract resistant to flash-loan price manipulation.",
        ],
    },
}

DOMAINS = [
    "coding", "legal", "finance", "science", "systems_design", "security",
    "business", "health", "creative", "history", "geography", "ml_ai",
    "distributed_systems", "devops", "data_engineering", "economics",
    "operations", "casual", "language", "math",
]

MISLABELING_TRAPS = (
    "Length is not complexity -- do not write a long prompt just to make it "
    "look harder than its tier, or a short prompt for a genuinely hard tier. "
    "A familiar domain does not imply a low tier. Code presence does not "
    "imply a high tier -- boilerplate is Tier 1, only structurally complex "
    "or correctness-critical code is Tier 2/3. A multi-part question "
    "inherits the tier of its hardest part, not an average."
)


def build_generation_prompt(tier: int, n: int, domain: str) -> str:
    spec = TIERS[tier]
    anchors = "\n".join(f"- {a}" for a in spec["anchors"])
    return (
        f"Generate {n} distinct user prompts in the '{domain}' domain that belong to "
        f"Tier {tier} ({spec['label']}) of a query-complexity taxonomy.\n\n"
        f"Tier {tier} characteristics: {spec['characteristics']}\n\n"
        f"Anchor examples for this tier (for calibration, do not repeat them):\n{anchors}\n\n"
        f"Avoid these mislabeling traps: {MISLABELING_TRAPS}\n\n"
        f"Return one prompt per line, no numbering, no extra commentary."
    )


def call_llm(prompt: str) -> list[str]:
    """
    Implement this against your org's approved LLM API client.
    Must return a list of generated prompt strings, one per requested line.
    Left unimplemented here since no live API is wired into this environment --
    see DESIGN.md Section 2.1 item 6 for the budget/quota approval this needs first.
    """
    raise NotImplementedError(
        "Wire this up to your approved LLM client before running at scale."
    )


def generate_batch(tier: int, count: int, batch_size: int = 20) -> list[dict]:
    records = []
    remaining = count
    idx = 1
    while remaining > 0:
        n = min(batch_size, remaining)
        domain = random.choice(DOMAINS)
        prompt = build_generation_prompt(tier, n, domain)
        generated_lines = call_llm(prompt)
        for line in generated_lines:
            text = line.strip("- ").strip()
            if not text:
                continue
            records.append({
                "prompt_id": f"GEN-T{tier}-{idx:04d}",
                "prompt_text": text,
                "tier_label": tier,
                "source": "synthetic",
                "annotator_id": "pending_review",
                "generation_model": "TBD",
                "domain_tag": domain,
                "split": "unassigned",
            })
            idx += 1
        remaining -= n
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", type=int, choices=[0, 1, 2, 3], required=True)
    parser.add_argument("--count", type=int, required=True, help="Number of prompts to generate")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    records = generate_batch(args.tier, args.count)

    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(records)} candidate Tier {args.tier} prompts to {args.output}")
    print(
        "annotator_id is 'pending_review' -- per DESIGN.md Section 5.2/5.4, every "
        "generated batch needs a human (or adjudicator-model) pass against "
        "data/labeling_rubric.md and a split assignment before merging into "
        "the training set."
    )


if __name__ == "__main__":
    main()
