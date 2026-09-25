## ADDED Requirements

### Requirement: TypeSafe Jev Runtime Safety Guard Harness
The `openjevpro.guard` module SHALL provide `TypeSafeJevGuardHarness` to evaluate raw probabilities or engine outputs, perform temperature calibration, and apply adaptive dual-threshold abstention.

#### Scenario: High-confidence prediction passes through safely
- **WHEN** calibrated maximum probability is greater than or equal to threshold $\tau = \max(\tau_{\min}, \alpha / K)$
- **THEN** `GuardDecision.choice` equals the top candidate, `is_abstained` is `False`, and `tentative_choice` equals `choice`

#### Scenario: Borderline prediction triggers abstention
- **WHEN** calibrated maximum probability is below threshold $\tau$
- **THEN** `GuardDecision.choice` is `"UNKNOWN"`, `is_abstained` is `True`, and `tentative_choice` preserves the candidate with highest calibrated score

#### Scenario: Pseudo-logit inversion and temperature calibration
- **WHEN** raw probabilities are provided without explicit logits
- **THEN** the harness maps probabilities to pseudo-logits via $z_i = \ln(\max(p_i, 10^{-6}))$ and applies `TemperatureCalibrator.calibrate(logits)`

#### Scenario: Degenerate or empty candidate handling
- **WHEN** candidate dict is empty
- **THEN** the harness returns `GuardDecision` with `choice="UNKNOWN"`, `confidence=0.0`, and `is_abstained=True` without raising unhandled exceptions
