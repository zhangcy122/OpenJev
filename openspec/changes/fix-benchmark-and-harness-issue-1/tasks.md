## 1. Harness Symmetry & Metrics

- [x] 1.1 Update `TypeSafeJevEngine.evaluate_choice` in `openjevpro/harness.py` to accept `allow_abstain: bool = True` and inject `"UNKNOWN"` into criteria when abstention is enabled.
- [x] 1.2 Implement 10-bin Expected Calibration Error (`_compute_ece`) helper in `OpenJevProHarness` and persist `ece` into `summary.engine_metrics`.
- [x] 1.3 Add test cases verifying criteria injection for TypeSafe Jev and automated ECE calculation.

## 2. Decision Client Abstention Return Contract

- [x] 2.1 Update `openjevpro/client.py` (`_decide_choice_ollama` and `_decide_choice_openai`) to set `value = "UNKNOWN"` whenever `abstained=True`.
- [x] 2.2 Update `openjevpro/schemas.py` docstrings to formally document the `"UNKNOWN"` abstention contract.
- [x] 2.3 Add test cases asserting `value == "UNKNOWN"` when `abstained is True`.

## 3. Benchmark Artifact & Documentation Alignment

- [x] 3.1 Update `examples/harness_benchmark_results.json` to record computed `ece` under summary metrics.
- [x] 3.2 Update `README.md` empirical replication table with true P50 latency (~4979ms), in-domain accuracy (30/30 vs 29/30), and empirical ECE values.
- [x] 3.3 Update `site/index.html` empirical replication table and takeaway cards to match the verified artifact figures.
- [x] 3.4 Add explicit citation and repository link to `ekzhang/openjev-sglang` in both `README.md` and `site/index.html`.

## 4. Verification & Issue Resolution

- [x] 4.1 Run unit and integration tests with `pytest` to guarantee zero regressions.
- [x] 4.2 Validate local preview server response on `site/index.html`.
- [ ] 4.3 Post the structured official reply on GitHub Issue #1.
