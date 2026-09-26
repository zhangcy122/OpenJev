## Why

Empirical measurements in Issue #3 reveal that for single-pass open LLM prompt classification, permuting candidate option order alters the winning label in ~20% of cases (mean 1.2 distinct winners across 8 orderings, with borderline samples producing up to 2–3 different winners). Subtracting letter-prior logprobs (null-state calibration) fails to resolve this because changing option order rewrites the prompt prefill context, inducing causal attention drift that dominates token prior skews.

Production decision systems and auditing workflows require deterministic consistency across candidate orderings. Providing an opt-in order-invariant evaluation mode and clear architectural guidance allows callers who prioritize reproducibility over single-pass latency to achieve guaranteed mathematical invariance.

## What Changes

- Add opt-in `order_invariant: bool = False` parameter to `OpenJevProClient.decide_choice()`.
- Implement `_decide_choice_order_invariant()` in `OpenJevProClient`: evaluates each candidate option independently in isolated forward passes without competitor options in the prompt, computing normalized likelihoods whose final softmax is strictly commutative across option order (guaranteeing exactly 1.0 distinct winners across any candidate permutation).
- Provide architectural documentation in README detailing the order sensitivity of single-pass causal decoders and recommending Tier 0 bidirectional System 1 models (such as Laya ModernBERT-large 322M) or `order_invariant=True` for high-consistency applications.
- Add comprehensive test suites verifying mathematical order-invariance across all permutations when `order_invariant=True`.

## Capabilities

### New Capabilities
- `order-invariant-choice`: Opt-in independent per-option decision mode in `OpenJevProClient` that guarantees 1.0 permutation invariance across candidate enum orderings by eliminating prompt option cross-interference.

### Modified Capabilities
<!-- Leave empty if no requirement changes -->

## Impact

- Affected code: `openjevpro/client.py`, `README.md`, `README_zh.md`.
- Backward compatibility: 100% backward compatible. Default behavior remains single-pass fast inference (`order_invariant=False`).
- Performance: When `order_invariant=True`, inference issues $N$ independent forward calls (or batched parallel requests) instead of 1 call, trading latency for strict permutation invariance.
