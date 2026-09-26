## ADDED Requirements

### Requirement: Drop-in Mock Decision Client
The system SHALL provide a `MockClient` class and an optional `mock: bool = False` factory parameter on `OpenJevProClient` that executes discrete choice, assertion, and score evaluations completely offline without sending network requests.

#### Scenario: Instantiate offline mock client
- **WHEN** a developer initializes `OpenJevProClient(mock=True)` or `MockClient()` without providing a `base_url`
- **THEN** the client SHALL successfully initialize without raising connection or configuration errors

#### Scenario: Offline discrete choice evaluation
- **WHEN** calling `decide_choice` on `MockClient` with state dict and candidate list `["billing", "technical_support", "security_fraud"]`
- **THEN** the client SHALL return a valid `DecisionReceipt[str]` containing `value`, `confidence`, `distribution`, and `latency_ms < 10.0`

### Requirement: Order-Invariant Simulation
The mock runtime SHALL implement isolated candidate scoring and commutative softmax normalization such that candidate ordering in the candidate list does not alter the winning choice or calibrated probabilities.

#### Scenario: Full permutation invariance across candidate orderings
- **WHEN** calling `decide_choice` with `order_invariant=True` on all $N!$ permutations of candidate lists
- **THEN** the winning candidate SHALL be identical across all permutations and the maximum probability variance SHALL be $\Delta P < 10^{-4}$

### Requirement: Zero-Argument Quickstart CLI Demo
The system SHALL provide an executable module `openjevpro.demo` that can be run via `python -m openjevpro.demo` with zero command-line arguments.

#### Scenario: Execute zero-argument demo runner
- **WHEN** running `python -m openjevpro.demo` in a terminal without an active inference server
- **THEN** the process SHALL exit with code 0 and output formatted decision receipts demonstrating discrete choice routing, binary guard asserts, and confidence scores in under 1 second
