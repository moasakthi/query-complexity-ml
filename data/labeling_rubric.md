## Labeling Rubric: Query Complexity Tiers

Source of truth for both human labeling and synthetic-data generation prompts. Mirrors BRD §4.1 / DESIGN.md §5.1 — do not fork the tier definitions elsewhere.

### Decision heuristic (apply in order)

Ask these questions about the prompt, in order, and stop at the first "yes":

1. **Does answering correctly require niche/specialist expertise, a multi-step proof or derivation, or generation of non-trivial, structurally complex code/systems design?** → **Tier 3**
2. **Does it require deep domain knowledge or multi-factor reasoning (e.g., comparing trade-offs, designing a system, analyzing a scenario with several interacting variables)?** → **Tier 2**
3. **Does it require some domain knowledge or a short reasoning chain, but a generalist could answer it correctly without specialist background?** → **Tier 1**
4. **Otherwise** (lookup, definition, basic arithmetic, formatting, single-fact recall) → **Tier 0**

### Tier definitions and anchors

| Tier | Label | Characteristics | Anchor examples | Destination |
| --- | --- | --- | --- | --- |
| 0 | Trivial | Simple lookups, basic Q&A, formatting requests, single-step arithmetic, single-fact recall | "What is the capital of Japan?" · "Convert 5 km to miles." · "List the planets." | Fast/cheap model |
| 1 | Simple | Moderate reasoning, basic domain knowledge, short explanations, boilerplate code | "Explain the difference between weather and climate." · "Write a Python function to reverse a string." | Mid-tier model |
| 2 | Moderate | Complex reasoning, deep knowledge, multi-factor trade-off analysis, non-trivial system/code design | "Design a database schema for multi-tenant SaaS billing." · "Compare GraphQL vs REST for a high-traffic public API." | Strong model |
| 3 | Complex | Niche expertise, multi-step logic/proofs, complex code generation, cross-domain synthesis | "Design a distributed rate limiter with exactly-once semantics across data centers, with a correctness argument." | Frontier model |

### Common mislabeling traps

- **Length ≠ complexity.** A long prompt asking for a simple bullet-point summary of a well-known topic is still Tier 0/1. A short prompt asking for a subtle proof is still Tier 3.
- **Familiar domain ≠ low tier.** "Explain how vaccines work" (Tier 1) vs. "Design a differentially private federated learning system for hospitals" (Tier 3) — both are "healthcare," complexity differs.
- **Code presence ≠ high tier automatically.** A boilerplate function (reverse a string, check even/odd) is Tier 1. A concurrent data structure with a correctness argument is Tier 3.
- **Multi-part questions inherit the tier of their hardest part**, not an average.

### Use in synthetic generation

When prompting an LLM to generate candidate examples for a tier, include: (a) the tier's row from the table above, (b) 3–5 anchor examples for that tier, (c) an explicit instruction to vary domain (coding, legal, science, business, casual, etc.) so the model learns complexity signal independent of topic, and (d) an instruction to avoid the mislabeling traps above.

Every generated batch — synthetic or human-written — should have a sample re-checked against this rubric by a second reviewer before being merged into the training set (see DESIGN.md §5.2/§5.4).
