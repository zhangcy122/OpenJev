## Why

GitHub Issue #1 reported by community reviewer @MrJev revealed critical reproduction gaps between the committed benchmark artifact `examples/harness_benchmark_results.json` and the public documentation (`README.md` and `site/index.html`). Specifically, latency numbers were inverted due to a decimal transcription error (P50 reported as ~508ms instead of 4979ms), in-domain accuracy denominators were misreported (29/29 instead of 30/30 vs 29/30), ECE metrics lacked artifact backing, the benchmark harness evaluated out-of-scope rejection asymmetrically (withholding the `UNKNOWN` escape hatch from TypeSafe Jev), and `ChoiceDecision` leaked categorical values when models abstained. Fixing these issues now restores scientific integrity, ensures fair benchmark comparisons, and hardens the SDK decision contracts.

## What Changes

- **Benchmark Harness Symmetry**: Update `TypeSafeJevEngine` in `openjevpro/harness.py` to symmetrically inject `criteria["UNKNOWN"]` when `allow_abstain=True`, ensuring fair evaluation across both open-set and closed-set rejection tasks.
- **Automated ECE Metric Computation**: Extend `OpenJevProHarness` to compute and export 10-bin Expected Calibration Error (ECE) directly into the benchmark results artifact (`examples/harness_benchmark_results.json`).
- **Safe Return Semantics on Abstention**: Update `OpenJevProClient.decide_choice` in `openjevpro/client.py` and schema definitions in `openjevpro/schemas.py` so that `decision.value` is safely normalized to `"UNKNOWN"` whenever `abstained=True`, preventing callers from consuming invalid argmax choices.
- **Documentation & Landing Page Accuracy**: Correct the latency rows, in-domain accuracy ratios, and ECE values in `README.md` and `site/index.html` to strictly match the empirical results in `examples/harness_benchmark_results.json`, explaining the cloud MoE test setup transparently.
- **Attribution & Citations**: Explicitly attribute and cite `ekzhang/openjev-sglang` and JevBench v1 in both `README.md` and `site/index.html`.

## Capabilities

### New Capabilities
- `benchmark-harness`: Fair, symmetric multi-engine evaluation harness supporting dynamic abstention option injection and automated ECE calculation.
- `decision-client`: Calibrated decision client with safe abstention return semantics (`value="UNKNOWN"` on `abstained=True`).
- `benchmark-reporting`: Reproducible, artifact-backed documentation and landing page benchmark tables with explicit attribution.

### Modified Capabilities
<!-- No existing global specs to modify -->

## Impact

- **Affected Code**: `openjevpro/harness.py`, `openjevpro/client.py`, `openjevpro/schemas.py`.
- **Affected Documentation & Web**: `README.md`, `site/index.html`.
- **Affected Data Artifacts**: `examples/harness_benchmark_results.json`.
- **APIs**: When `decision.abstained == True`, `decision.value` now returns `"UNKNOWN"` instead of a tentative category name (breaking for callers relying on raw argmax during abstention, but aligns with safety contracts).
