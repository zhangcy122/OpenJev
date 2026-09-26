## 1. Core Flywheel & Crystallization Operator Implementation

- [x] 1.1 Implement `DeliberativeDecisionFlywheel` orchestrator in `openjevpro/flywheel.py` wrapping `OpenJevProClient` and reasoning model backend. <!-- id: task-flywheel-orchestrator -->
- [x] 1.2 Implement `CrystallizationOperator` and `CrystallizationStore` to refine candidate criteria and store decision precedents. <!-- id: task-crystallization-operator -->
- [x] 1.3 Implement conditional escalation logic triggered when System 1 confidence falls below threshold $\tau$ or produces `choice="UNKNOWN"`. <!-- id: task-conditional-escalation -->
- [x] 1.4 Expose convenience factory `create_decision_flywheel()` on `OpenJevProClient` and `decide_with_crystallization()` API. <!-- id: task-client-flywheel-convenience -->

## 2. Unit Testing & Cognitive Flywheel Convergence Verification

- [x] 2.1 Add unit test in `tests/test_flywheel.py` verifying high-confidence queries execute on the sub-50ms fast path without invoking System 2. <!-- id: task-test-fast-path-passthrough -->
- [x] 2.2 Add unit test verifying that ambiguous or abstained queries trigger System 2 exploration and generate a valid `crystallization_receipt`. <!-- id: task-test-exploration-escalation -->
- [x] 2.3 Add unit test verifying fast-path promotion: subsequent queries in the crystallized subspace succeed on System 1 at <50ms without re-escalation. <!-- id: task-test-fast-path-promotion -->
- [x] 2.4 Run full test suite via `python3 -m pytest -v tests/` to confirm all existing and new tests pass cleanly. <!-- id: task-test-full-regression -->

## 3. Documentation & Architectural Guidance

- [x] 3.1 Document the "Explore First, Crystallize Later" cognitive flywheel in `README.md` and `README_zh.md` detailing the System 2 to System 1 distillation loop. <!-- id: task-doc-flywheel-concept -->
- [x] 3.2 Add executable code snippet demonstrating `DeliberativeDecisionFlywheel` setup and amortized cost-latency savings. <!-- id: task-doc-flywheel-snippets -->
