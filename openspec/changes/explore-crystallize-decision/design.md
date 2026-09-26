## Context

OpenJevPro currently provides fast probabilistic decisions (Mode A via open LLMs / Laya; Mode B via TypeSafe Jev Guard). On well-defined tasks, calibrated System 1 decisions achieve 100% selective accuracy and sub-35ms latency. However, when complex, ambiguous, or out-of-distribution queries arrive, the probability distribution is flattened across competing candidates, triggering safe abstention (`choice="UNKNOWN"`, `abstained=True`).

Without an epistemic feedback loop, these ambiguous queries remain permanently unanswered by System 1 unless a human engineer manually reviews logs and refines the criteria prompt. Conversely, routing all traffic through reasoning models (such as DeepSeek-R1, Qwen-Thinking, or o1/o3-mini) is cost-prohibitive ($0.02+/call) and too slow (1.5s–5.0s) for synchronous agent loops.

By implementing a deliberative exploration and crystallization flywheel, System 2 reasoning is treated as an offline/near-line compiler for System 1 fast decisions.

## Goals / Non-Goals

**Goals:**
- Provide a `DeliberativeDecisionFlywheel` orchestrator that intercepts low-confidence or abstained decisions and invokes an exploratory reasoning LLM.
- Implement a structured `CrystallizationOperator` that distills reasoning traces into:
  1. Refined, discriminative criteria for candidates.
  2. Semantic exemplar memory storing decision precedents.
- Enable automatic fast-path promotion: subsequent queries in the same semantic neighborhood SHALL be resolved by System 1 at <35ms without re-invoking the reasoning model.
- Provide full auditable provenance: each decision contains an optional `crystallization_receipt` detailing why and how the decision boundary was formed.
- Support candidate discovery: flag recurring unmapped clusters and suggest schema additions (new Enum options).

**Non-Goals:**
- Replacing offline fine-tuning pipelines: the flywheel operates at the prompt-criteria, exemplar-retrieval, and calibration layer; weight fine-tuning can asynchronously consume these records.
- Forcing slow reasoning on clear, high-confidence queries: queries with confidence $\ge \tau$ bypass System 2 completely.

## Decisions

### Decision 1: Conditional Escalation Triggering
- **Rationale**: Escalation to System 2 is gated by the existing `TypeSafeJevGuardHarness` contracts:
  $$\text{Trigger System 2} \iff \text{abstained} = \text{True} \lor \text{confidence} < \tau_{\text{effective}}$$
- High-confidence queries (typically 90–95% of traffic) never incur System 2 latency or token expense.

### Decision 2: Structured Exploration Schema
- **Rationale**: When System 2 is invoked, the prompt forces a structured JSON response containing:
  1. `winning_candidate`: The chosen candidate category.
  2. `justification`: Causal argument explaining why other candidates were eliminated.
  3. `discriminative_rule`: A concise, generalized rule distinguishing this candidate from the near-miss competitor.
- This ensures the exploration result is directly machine-distillable.

### Decision 3: Two-Tier Crystallization Substrate
- **Rationale**:
  - **Tier A (In-Context Criteria Enrichment)**: The generated `discriminative_rule` is merged into the runtime `criteria` dictionary under the winning candidate.
  - **Tier B (Exemplar Memory Store)**: The state and rule are stored in an in-memory / local disk key-value or vector cache (`CrystallizationStore`).
  - When future queries evaluate candidates, System 1 utilizes the enriched criteria and relevant exemplars, lifting logprob separation and posterior confidence above the abstention cutoff $\tau$.

## Risks / Trade-offs

- **[Risk] Hallucinated or drifting criteria updates**: An LLM might produce overly specific or contradictory criteria during exploration.
  - **Mitigation**: Constrain criteria updates with a validation gate: refined criteria must increase target candidate score without dropping existing test fixture performance. Provide developer review hooks (`auto_crystallize=False` mode for human-in-the-loop review).
- **[Risk] Memory bloat in criteria prompt**: Unchecked criteria growth could bloat prompt length and increase TTFT.
  - **Mitigation**: Cap criteria length per candidate (e.g. max 500 characters) and use summarization/consolidation when updates exceed limits.
