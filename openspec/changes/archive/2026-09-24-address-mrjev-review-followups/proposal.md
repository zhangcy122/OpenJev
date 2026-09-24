## Why

Following the empirical replication and live verification of OpenJevPro by external reviewer `@MrJev` (which confirmed 100% reproduction of benchmark accuracy, selective metrics, latencies, and ECE), several follow-up technical details and governance clarifications were identified. Addressing these items strengthens OpenJevPro's reproducibility, automated testing, algorithmic calibration flexibility, and open licensing transparency.

## What Changes

- **P95 Latency Convention Documentation**: Formally specify the rank index convention $P95 = \text{sorted}(L)[\lfloor 0.95 \cdot N \rfloor]$ in `README.md` and benchmark documentation, noting nearest-rank comparisons to eliminate cross-benchmark ambiguity on small sample sizes ($N=36$).
- **Automated CI Test Workflow**: Add `.github/workflows/test.yml` running the 6 offline unit tests across Python 3.10, 3.11, and 3.12 on all pushes and pull requests to prevent regressions.
- **Independent Architecture Attribution**: Clearly distinguish external reference architecture `openjev-sglang` (by @ekzhang) as an independent baseline in the comparative benchmark tables and documentation.
- **Dynamic Temperature Calibration Fitting**: Extend `TemperatureCalibrator` with a data-driven `fit()` method optimizing negative log-likelihood (NLL) on labeled validation sets, documenting 1.25 as an empirical default prior.
- **Source-Available License Terminology Alignment**: Standardize license terminology across `README.md`, `pyproject.toml`, and site documentation to explicitly state "Source-Available / PolyForm Noncommercial 1.0.0", distinguishing it from OSI-approved open source.

## Capabilities

### New Capabilities
- `benchmark-conventions-and-attribution`: Standardized latency percentile conventions and attribution guidelines for external baseline architectures.
- `ci-test-automation`: Continuous integration workflow specification guaranteeing automated offline unit test execution on GitHub Actions.
- `temperature-calibration-fitting`: Algorithmic dynamic temperature fitting via negative log-likelihood minimization on labeled validation log-odds.
- `license-terminology-alignment`: Precise legal classification and messaging alignment for PolyForm Noncommercial source-available software.

### Modified Capabilities
<!-- None, as this is the foundational formalization of these specifications. -->

## Impact

- **Code**: `openjevpro/calibrator.py` gains `fit()` method; new unit tests in `tests/test_calibrator.py` or `tests/test_harness_and_client.py`.
- **CI / Workflows**: New file `.github/workflows/test.yml`.
- **Documentation**: `README.md`, `site/index.html`, `pyproject.toml` updated with clarified license terminology, P95 formula notes, and `openjev-sglang` attribution.
- **Dependencies**: No breaking changes; maintains zero external C/heavy dependencies (pure numpy/Python 1D optimization for calibration fitting).
