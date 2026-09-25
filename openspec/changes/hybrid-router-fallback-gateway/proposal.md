## Why

Deploying categorical intent decision engines solely on cloud commercial APIs (such as TypeSafe Jev Cloud or OpenAI/Anthropic APIs) introduces two major enterprise liabilities:
1. **Cost Inefficiency**: 70%-80% of routine traffic consists of high-frequency, standard intents. Sending all queries to cloud APIs inflates operational expenditure unnecessarily.
2. **SLA Fragility & Single Points of Failure**: Network latency spikes, rate limits (HTTP 429), timeouts, and cloud outages (HTTP 503) cause catastrophic interruptions for downstream autonomous agents and production workflows.

By implementing a two-tier Hybrid Router with an automated Fallback Gateway (`HybridJevGateway`):
- **Tier 1 (Fast Local Edge Engine)** intercepts high-frequency requests with low latency (<20ms) and zero incremental cloud cost.
- **Guard Interceptor** verifies calibrated posterior confidence against adaptive threshold $\tau$. When confidence is sufficient, the decision is resolved locally, slashing cloud API bills by over 60%.
- **Tier 2 (Cloud Decision Engine)** handles long-tail, low-confidence, and out-of-domain queries.
- **Circuit Breaker & Fallback Handler** intercepts network severance, timeouts, or API errors, gracefully degrading to local execution with explicit `degraded=True` and `tentative_value` retention, guaranteeing 99.99% business SLA continuity without process crashes.

## What Changes

- **Gateway Module**: Implement `openjevpro/gateway.py` containing:
  - `CircuitState` enum (`CLOSED`, `OPEN`, `HALF_OPEN`).
  - `CircuitBreaker` configuration and state machine.
  - `HybridGatewayDecision` structured schema.
  - `HybridJevGateway` combining local and cloud engines with adaptive routing and fallback.
- **Client Integration**:
  - Export `HybridJevGateway`, `CircuitBreaker`, `CircuitState`, and `HybridGatewayDecision` in `openjevpro/__init__.py`.
  - Add optional hybrid gateway factory / routing helpers in `openjevpro/client.py`.
- **Test Suite**: Add comprehensive test coverage in `tests/test_gateway.py`:
  - Local fast-path dispatch for high-confidence intents.
  - Penetration upgrade to cloud for ambiguous/low-confidence intents.
  - Circuit breaker trip on consecutive network/timeout failures.
  - Automatic fallback degradation with SLA continuity when cloud is unreachable.
  - Circuit half-open recovery upon cloud service restoration.

## Capabilities

### New Capabilities
- `hybrid-router-fallback-gateway`: High-concurrency two-tier cost arbitrage and fault-tolerant fallback gateway with circuit breaker state machine for OpenJev.

### Modified Capabilities
<!-- None -->

## Impact

- **Code**: New module `openjevpro/gateway.py`, updated `openjevpro/__init__.py` and `openjevpro/client.py`.
- **Tests**: New test file `tests/test_gateway.py`.
- **Compatibility**: 100% backward-compatible. Existing `OpenJevProClient` and `TypeSafeJevGuardHarness` remain unaffected.
