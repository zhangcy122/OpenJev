## Context

In standard single-pass classification mode, OpenJevPro formats candidate options as an enumerated list (e.g. `A. card_arrival \n B. change_pin...`) embedded directly in the prompt. For causal autoregressive decoder models (e.g. Qwen, Gemma), this creates two interlinked biases documented in Issue #3:
1. **Context Prefill Rewriting (Dominant)**: Changing option order alters the sequence of tokens in the prompt, causing attention weights to drift.
2. **Token Prior Asymmetry**: Letter tokens (`A`, `B`, `C`, etc.) exhibit unequal baseline logprobs.

Our CaW Epistemic Sandbox deduction confirmed that null-state calibration (subtracting letter priors) cannot fix this because prefill rewriting dominates token priors. The only mathematical path to guaranteed order invariance within causal decoders is removing the option list from the prompt entirely and scoring each candidate in its own isolated forward evaluation.

## Goals / Non-Goals

**Goals:**
- Provide an opt-in `order_invariant: bool = False` flag in `OpenJevProClient.decide_choice()`.
- Implement `_decide_choice_order_invariant()` to evaluate each candidate option independently in isolated forward passes without competitor options in the prompt context.
- Aggregate independent scores via commutative temperature calibration and softmax normalization, guaranteeing exactly 1.0 distinct winners across all permutations of candidate order.
- Maintain sub-150ms execution via concurrent `ThreadPoolExecutor` dispatches for the $N$ option evaluations.
- Add clear architectural guidelines and README documentation explaining the single-pass vs. order-invariant trade-offs.

**Non-Goals:**
- Forcing `order_invariant=True` as default: default remains `order_invariant=False` to preserve sub-50ms single-forward-pass latency.
- Altering Laya's native microservice protocol: Laya operates as a bidirectional ModernBERT encoder and already exhibits high architectural invariance in single-pass mode.

## Decisions

### Decision 1: Independent Binary Prompting with Commutative Softmax
- **Rationale**: If option $c$ is evaluated in a prompt containing only state $x$ and candidate $c$, the forward pass output $s(c)$ depends strictly on $(x, c)$ and is independent of all other candidates $c' \in C$.
- The aggregated vector $\mathbf{s} = [s(c_1), s(c_2), \dots, s(c_N)]$ is then calibrated via `TemperatureCalibrator.calibrate()`.
- Because softmax is permutation-equivariant, $\text{argmax}(\text{softmax}(\mathbf{s}))$ is strictly order-invariant.
- **Alternatives Considered**:
  - *Letter-prior subtraction*: Proved ineffective in Issue #3 and CaW sandbox deduction.
  - *Bidirectional mirror ensembling (forward + reverse)*: Reduces variance with 2 calls, but does not provide mathematical guarantee (1.0 distinct winners) on complex borderline cases.

### Decision 2: Parallel Request Execution
- **Rationale**: An enum with $N=5$ to $8$ options issued sequentially would incur $8 \times$ latency (~300–500ms). Using `ThreadPoolExecutor(max_workers=min(len(options), 8))` issues calls concurrently to local vLLM / SGLang / Ollama servers, keeping wall-clock latency within ~80–150ms.

### Decision 3: Transparent Receipt Metadata
- **Rationale**: When `order_invariant=True`, the resulting `ChoiceDecision` populates `raw_logits` with the independent per-option scores, recording `order_invariant=True` in its metadata for auditability.

## Risks / Trade-offs

- **[Risk] Increased server load**: $N$ concurrent forward passes multiply inference request volume by $N$.
  - **Mitigation**: Default to `order_invariant=False`. Document that standard single-pass mode is recommended for high-QPS routing, while `order_invariant=True` is targeted at compliance, auditing, and mission-critical reproducibility.
- **[Risk] High option counts ($N > 15$)**: Large candidate sets could cause thread exhaustion or rate-limiting.
  - **Mitigation**: Bound maximum worker threads to 8 and warn if candidate count exceeds 20.
