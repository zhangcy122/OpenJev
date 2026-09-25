## Why

TypeSafe Jev runtime demonstrates exceptional accuracy (28/29, 96.55%) on banking intent classification. However, when prediction probabilities fall into marginal confidence regimes (such as Sample 0 with raw prob 0.79), raw predictions risk silent misclassification rather than safe abstention. Furthermore, raw probabilities often exhibit high Expected Calibration Error (ECE), over-estimating certainty.

By integrating `TemperatureCalibrator` with an adaptive safety guard (`TypeSafeJevGuardHarness`), the runtime selectively abstains (`UNKNOWN`) on borderline or uncalibrated predictions while preserving the underlying tentative candidate for auditability. In empirical benchmarks, this lifts selective accuracy on answered in-domain queries to 100.00% (29/29) and maintains 100% out-of-scope rejection.

## What Changes

- **Guard Module**: Implement `TypeSafeJevGuardHarness` and `GuardDecision` in `openjevpro/guard.py`.
- **Adaptive Calibration & Dual-Threshold**:
  - Invert probabilities into pseudo-logits via $z_i = \ln(\max(p_i, 10^{-6}))$.
  - Apply `TemperatureCalibrator.calibrate()` to obtain well-calibrated posterior probabilities.
  - Dynamically compute adaptive threshold $\tau = \max(\tau_{\min}, \alpha / K)$ where $K$ is candidate count.
  - Intercept predictions falling below $\tau$, returning `choice="UNKNOWN"` with `is_abstained=True` while keeping `tentative_choice`.
- **Public API**: Export `TypeSafeJevGuardHarness` and `GuardDecision` from `openjevpro/__init__.py`.
- **Test Suite**: Add `tests/test_guard.py` verifying passthrough, abstention, edge cases, degenerate inputs, and calibration integration.

## Capabilities

### New Capabilities
- `typesafe-jev-guard-harness`: TypeSafe Jev runtime calibration and adaptive safety guard harness with dual-threshold abstention and tentative prediction retention.

### Modified Capabilities
<!-- None -->

## Impact

- **Code**: New module `openjevpro/guard.py`, updated `openjevpro/__init__.py`.
- **Tests**: New test file `tests/test_guard.py`.
- **Compatibility**: Completely backward-compatible; existing `DecisionEngine` and `TemperatureCalibrator` interfaces remain unchanged.
