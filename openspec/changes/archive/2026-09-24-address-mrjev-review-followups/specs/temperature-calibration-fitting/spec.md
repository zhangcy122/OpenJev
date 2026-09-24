## ADDED Requirements

### Requirement: Dynamic Temperature Calibration Fitting
The `TemperatureCalibrator` class SHALL provide a `fit()` method that determines the optimal temperature parameter $T > 0$ by minimizing negative log-likelihood (NLL) over validation logits and labels.

#### Scenario: User fits temperature using a validation dataset
- **WHEN** `TemperatureCalibrator.fit(logits_list, targets)` is invoked with validation logits and corresponding ground truth keys
- **THEN** the method minimizes cross-entropy / NLL loss, updates `self.temperature` with the optimal value, and returns the fitted temperature

#### Scenario: Fallback behavior on invalid or insufficient validation data
- **WHEN** `fit()` is called with empty data or fewer than 2 valid samples
- **THEN** a `ValueError` or fallback warning is raised and the calibrator retains its existing temperature setting
