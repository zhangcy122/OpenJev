## 1. Core Mock Engine & Synthesis

- [ ] 1.1 Implement `openjevpro/mock.py` with deterministic semantic keyword scoring and isolated candidate score synthesis <!-- id: task-core-mock-scorer -->
- [ ] 1.2 Implement simulated commutative softmax normalization in `MockClient` with temperature scaling and order-invariance <!-- id: task-mock-commutative-softmax -->
- [ ] 1.3 Add `evaluate_noul` and `evaluate_score` implementations to `MockClient` returning standard typed receipts <!-- id: task-mock-primitives-receipts -->

## 2. Client Integration & Exports

- [ ] 2.1 Update `OpenJevProClient` in `openjevpro/client.py` to accept `mock: bool = False` and route calls to `MockClient` when enabled <!-- id: task-client-mock-routing -->
- [ ] 2.2 Export `MockClient` from `openjevpro/__init__.py` <!-- id: task-export-mock-client -->

## 3. CLI Demonstration Runner

- [ ] 3.1 Update `openjevpro/demo.py` to support execution via `python -m openjevpro.demo` with automatic mock fallback when no backend is reachable <!-- id: task-demo-fallback-runner -->
- [ ] 3.2 Format interactive terminal receipts showcasing discrete choice routing, confidence score, and permutation invariance test <!-- id: task-demo-terminal-formatting -->

## 4. Test Verification

- [ ] 4.1 Create `tests/test_mock_client.py` verifying full API parity, isolated candidate scoring, and mathematical permutation invariance across $N!$ candidate orders <!-- id: task-test-mock-invariance -->
- [ ] 4.2 Verify existing and new test suites pass with 100% green status under pytest <!-- id: task-pytest-regression-verification -->
