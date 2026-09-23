## Context

In response to GitHub Issue #1 by @MrJev, this design addresses:
1. **Harness Asymmetry**: `TypeSafeJevEngine` did not support abstention escape hatches, evaluating `ans.get("choice") == "UNKNOWN"` against a criteria dict lacking `"UNKNOWN"`.
2. **Metric Automation**: ECE metrics were calculated externally or omitted from serialized test artifacts, leading to irreproducible claims in documentation.
3. **API Safety**: `ChoiceDecision` currently returns the tentative argmax choice even when `abstained=True`, creating a trap for downstream developers who only inspect `.value`.
4. **Documentation Accuracy**: Reversing transposed latency numbers (P50 4979ms vs 772ms) and in-domain accuracy ratios (30/30 vs 29/30) in `README.md` and `site/index.html`.

## Goals / Non-Goals

**Goals:**
- Provide symmetric abstention configuration in `TypeSafeJevEngine` so both OpenJevPro and TypeSafe Jev are offered `"UNKNOWN"` when `allow_abstain=True`.
- Implement in-process 10-bin Expected Calibration Error (ECE) computation in `OpenJevProHarness` and persist it to `examples/harness_benchmark_results.json`.
- Normalize `ChoiceDecision.value` to `"UNKNOWN"` when `abstained=True`, documenting the behavioral contract clearly.
- Align all figures in `README.md` and `site/index.html` with raw numbers in `examples/harness_benchmark_results.json`.
- Add explicit attribution and links to `ekzhang/openjev-sglang` for JevBench v1 evaluations.

**Non-Goals:**
- Retraining or fine-tuning models to artificially reduce cloud latency.
- Modifying upstream TypeSafe Jev cloud endpoints or behavior.
- Altering unrelated licensing or packaging structures.

## Decisions

### Decision 1: Symmetric Abstention Injection in `TypeSafeJevEngine`
- **Approach**: When `allow_abstain=True` is passed to `evaluate_choice`, check if `"UNKNOWN"` exists in criteria. If absent, inject `criteria["UNKNOWN"] = "None of the other categories apply, or query is out of scope."`.
- **Alternatives Considered**:
  - *Keep criteria unchanged and remove OOS evaluation*: Dismissed because open-world rejection is a core evaluation objective.
  - *Hardcode prompt string injection*: Dismissed; injecting into the criteria dictionary matches the official TypeSafe Jev candidate specification format.

### Decision 2: Automated 10-Bin ECE Metric in Harness Summary
- **Approach**: Add an `_compute_ece(confidences, accuracies, n_bins=10)` helper in `OpenJevProHarness` that partitions confidences into equal-width bins $[0, 0.1), [0.1, 0.2), \dots, [0.9, 1.0]$ and weighs $|acc(B_m) - conf(B_m)|$. Export `ece` under each engine's summary metrics.
- **Alternatives Considered**:
  - *Adaptive equal-frequency binning*: Standard literature (Guo et al., 2017) uses 10 equal-width bins for NLP classification calibration.

### Decision 3: Safe `value` Assignment on Abstention
- **Approach**: In `OpenJevProClient._decide_choice_ollama` and `_decide_choice_openai`, when `abstained=True`, set `value = "UNKNOWN"`. Preserve the underlying categorical probabilities in `probabilities` dictionary for callers who need introspection.
- **Alternatives Considered**:
  - *Set value to `None`*: Breaking change for Pydantic string validation (`value: str`). Setting `value = "UNKNOWN"` preserves the string contract and matches the abstention token.

### Decision 4: Transparent Latency and Hardware Disclosure
- **Approach**: Update the latency cells to show `~4,979 ms` (Cloud ~248ms / Network + Client parsing overhead), explicitly explaining that this run tested unoptimized cloud chat completions rather than a collocated vLLM engine with prefix caching.

## Risks / Trade-offs

- **[Risk] Downstream caller expects argmax in `.value` even when abstained**  
  → *Mitigation*: Callers can still inspect `max(decision.probabilities, key=decision.probabilities.get)` if they intentionally require forced guesses; `.value = "UNKNOWN"` prevents unintended catastrophic actions.
- **[Risk] Slower reported latency reduces initial marketing appeal**  
  → *Mitigation*: Honesty and scientific reproducibility build durable credibility; the table will highlight the architectural difference between collocated vLLM single-token logprob decoding (sub-50ms) vs end-to-end unoptimized cloud inference (~5s).
