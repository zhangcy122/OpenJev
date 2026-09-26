## ADDED Requirements

### Requirement: Opt-in Order-Invariant Decision Mode
The `OpenJevProClient.decide_choice()` method SHALL accept an optional parameter `order_invariant: bool = False`. When `order_invariant=True`, the client SHALL evaluate each candidate option in an isolated forward pass without competitor options appearing in the prompt, eliminating cross-option prefill context interference.

#### Scenario: Invariant evaluation across permuted candidate lists
- **WHEN** `decide_choice` is invoked with `order_invariant=True` on a query across different candidate permutations (e.g. forward, reversed, shuffled)
- **THEN** the returned winning choice and relative candidate score distribution SHALL remain identical across all candidate list permutations (1.0 distinct winners across all orderings)

#### Scenario: Default single-pass execution preserved
- **WHEN** `decide_choice` is invoked without specifying `order_invariant` or with `order_invariant=False`
- **THEN** the client SHALL execute standard single-forward-pass inference, preserving sub-50ms execution speed

### Requirement: Commutative Likelihood Normalization and Abstention
When executing in order-invariant mode, the client SHALL gather independent candidate likelihood scores into a unified distribution, apply temperature calibration via `TemperatureCalibrator`, and apply abstention thresholding matching the standard `ChoiceDecision` contract.

#### Scenario: Low confidence triggers symmetric UNKNOWN abstention
- **WHEN** an ambiguous or out-of-scope query yields calibrated confidence below the effective abstention threshold under `order_invariant=True`
- **THEN** the returned `ChoiceDecision.value` SHALL be normalized to `"UNKNOWN"` with `abstained=True` and `tentative_value` set to the top candidate

#### Scenario: Concurrent forward pass dispatch
- **WHEN** `order_invariant=True` is executed for $N$ candidate options
- **THEN** independent option evaluation calls SHALL be dispatched concurrently using a thread pool to minimize wall-clock latency

### Requirement: Documentation of Order Sensitivity Trade-offs
The documentation in `README.md` and `README_zh.md` SHALL clearly disclose that single-pass causal decoder classification is sensitive to candidate option ordering on borderline queries, provide instructions on enabling `order_invariant=True`, and explain the performance and cost trade-offs.

#### Scenario: User references order sensitivity in documentation
- **WHEN** a user consults `README.md` or `README_zh.md` regarding option ordering
- **THEN** they SHALL find an explicit explanation of prompt prefill context effects, the recommendation to freeze enum member order, and the availability of `order_invariant=True` and Tier 0 Laya for order-robust workloads
