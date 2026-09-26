## 1. Core Client Implementation

- [ ] 1.1 Add `order_invariant: bool = False` parameter to `OpenJevProClient.decide_choice()` signature in `openjevpro/client.py`. <!-- id: task-client-order-invariant-arg -->
- [ ] 1.2 Implement `_score_single_candidate()` in `OpenJevProClient` to evaluate candidate likelihood in an isolated prompt without competitor options. <!-- id: task-client-single-candidate-scorer -->
- [ ] 1.3 Implement `_decide_choice_order_invariant()` in `OpenJevProClient` using `ThreadPoolExecutor` to score candidate options concurrently. <!-- id: task-client-concurrent-dispatch -->
- [ ] 1.4 Integrate candidate log-likelihood aggregation with `TemperatureCalibrator.calibrate()` and abstention thresholding, populating `raw_logits` and `tentative_value`. <!-- id: task-client-calibration-and-abstain -->

## 2. Unit Testing & Permutation Invariance Verification

- [ ] 2.1 Add unit test `test_order_invariant_choice_permutations` in `tests/test_harness_and_client.py` verifying identical winner across original, reversed, and shuffled candidate orderings. <!-- id: task-test-permutation-invariance -->
- [ ] 2.2 Add unit test verifying that ambiguous or out-of-scope queries safely abstain (`choice="UNKNOWN"`, `abstained=True`) under `order_invariant=True`. <!-- id: task-test-invariant-abstention -->
- [ ] 2.3 Run full test suite via `pytest -v tests/` to confirm all existing and new tests pass cleanly. <!-- id: task-test-regression-check -->

## 3. Documentation & Architectural Guidance

- [ ] 3.1 Add a dedicated note in `README.md` and `README_zh.md` explaining single-pass causal decoder order sensitivity and enum member freezing best practices. <!-- id: task-doc-order-sensitivity-note -->
- [ ] 3.2 Add a code snippet demonstrating `client.decide_choice(..., order_invariant=True)` for mission-critical reproducibility. <!-- id: task-doc-invariant-code-snippet -->
- [ ] 3.3 Highlight Tier 0 Laya (ModernBERT 322M) as an ultra-fast bidirectional alternative that inherently resists causal prompt order drift. <!-- id: task-doc-laya-guidance -->
