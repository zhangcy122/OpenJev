## ADDED Requirements

### Requirement: Safe Abstention Value Assignment
When `OpenJevProClient.decide_choice` triggers abstention due to low confidence (`confidence < abstain_threshold`) or explicit selection of the `"UNKNOWN"` token, the returned `ChoiceDecision.value` MUST be set to `"UNKNOWN"`, and `ChoiceDecision.abstained` MUST be set to `True`.

#### Scenario: Low-confidence prediction triggers safe abstention
- **WHEN** client evaluates candidates where maximum calibrated confidence is below `abstain_threshold`
- **THEN** the returned `ChoiceDecision` object has `abstained=True` and `value="UNKNOWN"`

#### Scenario: High-confidence in-domain prediction
- **WHEN** client evaluates candidates where maximum calibrated confidence is equal to or greater than `abstain_threshold`
- **THEN** the returned `ChoiceDecision` object has `abstained=False` and `value` matches the winning candidate key
