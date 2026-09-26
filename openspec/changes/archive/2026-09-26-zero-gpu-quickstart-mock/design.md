## Context

OpenJevPro provides high-throughput, deterministic decision primitives (`Choice<T>`, `Noul`, `Score`) with mathematical order-invariance and post-hoc probability calibration. However, running `OpenJevProClient` currently requires an active vLLM, SGLang, or Laya HTTP endpoint that supports logprobs and vocabulary masking.

Without an active local GPU or configured cloud endpoint, prospective developers cannot immediately execute sample code, leading to high drop-off during initial evaluation. We need an offline mock runtime that behaves identically to the production client from an API standpoint, while executing locally in pure Python with zero external dependencies.

## Goals / Non-Goals

**Goals:**
- Provide a drop-in `MockClient` (and `OpenJevProClient(mock=True)`) that implements the complete decision protocol without network I/O.
- Implement an in-memory heuristic logit generator that simulates isolated candidate scoring and commutative softmax to showcase 100% order-invariance offline.
- Enable `python -m openjevpro.demo` to run out of the box with zero arguments on any environment, displaying formatted terminal receipts.
- Provide a deterministic test double for agent developers writing unit tests for their workflow routing logic.

**Non-Goals:**
- Replacing real neural inference for production accuracy. The mock runtime is an evaluation and testing double, not a replacement for vLLM/Laya models.
- Embedding a multi-gigabyte neural weight checkpoint. The mock must remain ultra-lightweight (<50KB code footprint).

## Decisions

### Decision 1: Pure-Python Deterministic Semantic Scorer
- **Choice**: Implement keyword-matching heuristics combined with deterministic SHA-256 hash seeding to produce realistic conditional logits and calibrated confidence scores.
- **Rationale**: Keeps the package lightweight with zero dependencies beyond the standard library (no ONNX runtime or PyTorch required).
- **Alternative considered**: Bundling an ONNX quantized embedding model. Rejected because adding a 100MB+ model package creates installation latency and CPU compatibility issues across architectures.

### Decision 2: API Parity via Interface Alignment
- **Choice**: `MockClient` implements the identical interface signatures as `OpenJevProClient`: `decide_choice`, `evaluate_noul`, and `evaluate_score`, returning standard `DecisionReceipt[T]`, `NoulReceipt`, and `ScoreReceipt` objects.
- **Rationale**: Code written against `mock=True` can switch to production `base_url="http://localhost:8000/v1"` by changing a single parameter.

### Decision 3: Commutative Softmax & Permutation Invariance Emulation
- **Choice**: Candidate scoring computes each candidate's raw score independently (`isolated_score(state, candidate)`), then normalizes via `exp(score_i) / sum(exp(score_j))` with temperature scaling.
- **Rationale**: Demonstrates OpenJevPro's signature mathematical order-invariance (`order_invariant=True`), ensuring tests that verify permutation invariance pass offline.

## Risks / Trade-offs

- **[Risk] Users confusing Mock with Production Engine** → **Mitigation**: Every receipt produced by `MockClient` sets `model="mock-simulator-v1"` and prints a terminal notice indicating mock evaluation mode.
- **[Risk] Mock behavior drifting from actual server responses** → **Mitigation**: Test suite runs parity checks ensuring schemas, fields, and exceptions match between `MockClient` and `OpenJevProClient`.
