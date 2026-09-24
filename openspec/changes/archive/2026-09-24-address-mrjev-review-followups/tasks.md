## 1. Continuous Integration (CI) Workflow Setup

- [x] 1.1 Create `.github/workflows/test.yml` with Python 3.10, 3.11, and 3.12 matrix running `pytest -v tests/`
- [x] 1.2 Verify GitHub Actions workflow syntax and ensure all offline unit tests execute in < 1s

## 2. Dynamic Temperature Calibration Fitting

- [x] 2.1 Implement `TemperatureCalibrator.fit(logits_list, targets)` in `openjevpro/calibrator.py` via NLL minimization
- [x] 2.2 Add unit tests in `tests/test_calibrator.py` verifying convergence, boundary safety ($T \in [0.1, 10.0]$), and fallback on edge cases
- [x] 2.3 Run full pytest suite and verify all unit tests pass

## 3. Benchmark Conventions & Attribution Documentation

- [x] 3.1 Document $P95 = \text{sorted}(L)[\lfloor 0.95 \cdot N \rfloor]$ rank index formula in `README.md` and publish nearest-rank comparison values
- [x] 3.2 Update first benchmark comparison table in `README.md` to clearly designate `openjev-sglang` (@ekzhang) as an independent external reference

## 4. License Terminology Normalization & Validation

- [x] 4.1 Standardize license terminology across `README.md`, `pyproject.toml`, and site documentation to "Source-Available / PolyForm Noncommercial 1.0.0"
- [x] 4.2 Re-run CaW Epistemic Sandbox (`caw explore-sandbox . --run --timeout 15`) and confirm all verification checks pass

