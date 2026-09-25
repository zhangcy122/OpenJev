# Hybrid Router & Fallback Gateway Spec

## 1. Overview
The Hybrid Router & Fallback Gateway provides a resilient, cost-arbitraged decision routing framework between a fast local/edge engine (Tier 1) and an expressive cloud engine (Tier 2).

## 2. Requirements

### 2.1 Routing Invariants
- **High-Confidence Fast Path**: When the local engine produces a calibrated decision with confidence $\ge \tau_{\text{local}}$ and is not abstained, the gateway MUST return the local decision immediately without calling the cloud engine.
- **Low-Confidence Escalation**: When the local engine produces a decision below $\tau_{\text{local}}$ or abstains, the gateway MUST route the request to the cloud engine if the circuit breaker allows it.

### 2.2 Fault Tolerance & Circuit Breaker Invariants
- **Circuit Breaker States**:
  - `CLOSED`: Allows routing to cloud engine.
  - `OPEN`: Disallows routing to cloud engine upon reaching `failure_threshold`.
  - `HALF_OPEN`: Allows a trial request after `recovery_timeout` has elapsed.
- **Graceful Fallback**: When the cloud engine raises any exception (e.g. `TimeoutError`, `ConnectionError`, `RuntimeError`) or the circuit is `OPEN`, the gateway MUST NOT propagate the exception. Instead, it MUST fall back to the Tier 1 local decision, setting `degraded=True` and retaining the `tentative_choice`.
- **SLA Guarantee**: The gateway must guarantee a returned `HybridGatewayDecision` under all simulated network severance scenarios.
