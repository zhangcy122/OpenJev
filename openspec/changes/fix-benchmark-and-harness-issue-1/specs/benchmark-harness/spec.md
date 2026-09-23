## ADDED Requirements

### Requirement: Symmetric Out-of-Scope Criteria Injection
The `TypeSafeJevEngine` SHALL support an `allow_abstain` parameter and MUST automatically inject `"UNKNOWN"` into the evaluation criteria dictionary when `allow_abstain=True` and `"UNKNOWN"` is not already provided, allowing the target engine an equivalent escape hatch for out-of-scope queries.

#### Scenario: Criteria injection on out-of-scope query
- **WHEN** `TypeSafeJevEngine.evaluate_choice` is invoked with `allow_abstain=True`
- **THEN** the request payload sent to the API contains `"UNKNOWN"` as an allowed criteria key with descriptive fallback instructions

### Requirement: Automated Expected Calibration Error Calculation
The `OpenJevProHarness` SHALL compute Expected Calibration Error (ECE) partitioned over 10 equal-width bins for each evaluated engine and MUST serialize the resulting score under `summary.engine_metrics[engine_name].ece` in the results JSON file.

#### Scenario: Automated metric output in benchmark summary
- **WHEN** `OpenJevProHarness.run` completes evaluating all dataset records
- **THEN** each engine's metrics in the serialized checkpoint file contains an `ece` float property representing its 10-bin calibration error
