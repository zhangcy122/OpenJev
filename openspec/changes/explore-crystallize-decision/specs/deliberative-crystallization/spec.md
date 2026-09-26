## ADDED Requirements

### Requirement: Conditional Deliberative Escalation
The `DeliberativeDecisionFlywheel` SHALL evaluate an input state against candidate options using the fast-path System 1 decision engine (`OpenJevProClient`). If the fast path decision produces high calibrated confidence ($\text{confidence} \ge \tau$ and not abstained), the flywheel SHALL return the fast path result immediately without invoking System 2 reasoning models.

#### Scenario: High-confidence query executes on fast path
- **WHEN** a state query produces calibrated confidence meeting or exceeding the effective abstention threshold $\tau$ under System 1
- **THEN** the flywheel SHALL return the System 1 `ChoiceDecision` within sub-50ms latency without invoking external reasoning models

#### Scenario: Ambiguous or abstained query escalates to System 2 exploration
- **WHEN** a state query produces calibrated confidence below the effective threshold or yields `choice="UNKNOWN"` with `abstained=True`
- **THEN** the flywheel SHALL invoke the configured exploratory reasoning LLM to perform counterfactual deliberation and return an exploration receipt containing the deduced choice, causal justification trace, and refined discriminative criterion

### Requirement: Crystallization Operator and Fast-Path Promotion
When an exploratory deliberation completes, the `CrystallizationOperator` SHALL distill the exploration receipt into the System 1 decision substrate by updating the active criteria dictionary and indexing the precedent in the crystallization store. Subsequent queries within the same semantic subspace SHALL be resolved directly by the fast-path engine above the confidence threshold without re-triggering System 2 escalation.

#### Scenario: Automated criteria refinement and fast-path promotion
- **WHEN** a cold-start edge query is resolved and crystallized via the `CrystallizationOperator`
- **THEN** subsequent identical or near-duplicate queries SHALL be answered directly by the System 1 engine with confidence exceeding the abstention threshold, recording sub-50ms execution latency

#### Scenario: Audit provenance and reasoning trace retention
- **WHEN** a decision is evaluated under the flywheel with crystallization enabled
- **THEN** the returned decision object SHALL include metadata recording whether System 2 was invoked, the reasoning trace (if escalated), and whether crystallization occurred
