## Context

Following the third-party review by `@MrJev`, OpenJevPro successfully demonstrated 100% reproducible accuracy, selective accuracy, coverage, latency, and ECE against official production baselines. To resolve remaining technical ambiguities and institutional rigor gaps, this design establishes formal conventions for percentile calculations, automated CI regression prevention, data-driven temperature calibration, and source-available license disclosures.

## Goals / Non-Goals

**Goals:**
- Provide complete transparency regarding latency percentile calculations ($P95$ rank convention) and document nearest-rank values for cross-tool comparability.
- Establish an automated GitHub Actions CI pipeline executing all unit tests offline across Python 3.10, 3.11, and 3.12.
- Introduce `TemperatureCalibrator.fit()` for empirical negative log-likelihood (NLL) optimization on validation logits while preserving existing defaults.
- Accurately attribute `ekzhang/openjev-sglang` as an independent external implementation.
- Align license terminology to "Source-Available / Non-Commercial" (PolyForm Noncommercial 1.0.0) without claiming OSI open-source status.

**Non-Goals:**
- Modifying raw benchmark data records in `examples/harness_benchmark_results.json` (already verified to reproduce down to the decimal).
- Altering default temperature value ($1.25$) when no validation dataset is provided (to preserve backward compatibility).
- Switching the underlying license from PolyForm Noncommercial 1.0.0 to Apache-2.0 or MIT.

## Decisions

### Decision 1: Percentile Convention Specification
- **Choice**: Document $P95 = \text{sorted}(L)[\lfloor 0.95 \cdot N \rfloor]$ explicitly in `README.md` and benchmark outputs. Provide side-by-side nearest-rank comparisons for $N=36$ (Index 34 ceil-rank vs Index 33 nearest-rank).
- **Rationale**: On small sample sizes ($N=36$), index definitions cause up to 250ms discrepancy. Explicitly naming the index eliminates confusion while preserving backward artifact consistency.
- **Alternatives Considered**: Changing the formula to `numpy.percentile(method='linear')` would invalidate already verified and committed summary artifacts.

### Decision 2: CI Test Automation via GitHub Actions
- **Choice**: Implement `.github/workflows/test.yml` using `pytest -v tests/` across matrix `['3.10', '3.11', '3.12']`.
- **Rationale**: The 6 unit tests run completely offline in under 0.3s without external API keys or Ollama server dependencies, making them fast and deterministic in CI.
- **Alternatives Considered**: Mocking Ollama server in CI. Deemed unnecessary since existing unit tests test pure Python contracts, math, and harness logic.

### Decision 3: Temperature Calibrator Dynamic Fitting API
- **Choice**: Add `TemperatureCalibrator.fit(logits_list, targets)` using bounded 1D scalar optimization (golden-section search / Brent's method) to minimize NLL $\mathcal{L}(T) = -\frac{1}{M}\sum \log P_T(y_i \mid x_i)$.
- **Rationale**: Keeps dependencies strictly within standard Python / `numpy` without requiring heavy ML libraries (torch/scikit-learn).
- **Alternatives Considered**: Requiring PyTorch / SciPy. Avoided to keep OpenJevPro lightweight and dependency-minimal.

### Decision 4: Attribution & License Terminology Normalization
- **Choice**: Clarify table entries to designate `openjev-sglang` as "External Independent Reference" by `@ekzhang`. Update repo badges, README header, and table descriptions to "Source-Available / PolyForm Noncommercial 1.0.0".
- **Rationale**: Respects Eric Zhang's independent work and aligns with Open Source Initiative (OSI) compliance standards.

## Risks / Trade-offs

- **[Small sample percentile sensitivity]** → Mitigated by publishing both ceil-rank and nearest-rank values in documentation notes.
- **[CI run duration on PRs]** → Mitigated by 100% offline tests completing in < 1 second.
- **[Overfitting temperature on tiny validation sets]** → Mitigated by bounding $T \in [0.1, 10.0]$ and defaulting to $T=1.25$ when validation samples are insufficient ($N < 5$).
