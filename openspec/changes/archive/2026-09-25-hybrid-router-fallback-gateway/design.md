## Context

OpenJev provides high-precision decision engines (`TypeSafeJevEngine`, `DirectStructuredEngine`) and adaptive calibration harnesses (`TypeSafeJevGuardHarness`, `TemperatureCalibrator`).
In high-throughput enterprise deployments, running all categorization tasks against cloud endpoints incur non-negligible latency and expense. Furthermore, network or provider outages require client-level resilience.

## Goals / Non-Goals

### Goals
- Cut cloud categorization expenditure by $\ge 60\%$ by resolving high-confidence head traffic on local/edge engines.
- Ensure zero uncaught network exceptions by routing through an automated circuit breaker and fallback layer.
- Preserve full decision telemetry (`tier`, `cost_units`, `degraded`, `confidence`, `tentative_value`).
- Provide clean synchronous and asynchronous support if applicable, matching existing harness standards.

### Non-Goals
- Replacing neural weights or model architectures.
- Managing low-level GPU memory allocations or model weight downloads.

## Architecture

### 1. Data Schema: `HybridGatewayDecision`
```python
@dataclass
class HybridGatewayDecision:
    choice: str
    confidence: float
    tier: str  # "Tier1_Local" | "Tier2_Cloud" | "Tier1_Fallback"
    cost_units: int  # 0 for local/fallback, 1 for cloud
    degraded: bool
    is_abstained: bool
    tentative_choice: Optional[str] = None
    fallback_reason: Optional[str] = None
    calibrated_probs: Dict[str, float] = field(default_factory=dict)
    raw_probs: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 2. State Machine: `CircuitBreaker`
- `CLOSED`: Normal operation. Can route to Tier 2 Cloud.
- `OPEN`: Failures $\ge$ `failure_threshold`. Rejects Tier 2 Cloud immediately, routes directly to Tier 1 Fallback.
- `HALF_OPEN`: `current_time - last_failure_time >= recovery_timeout`. Allows single exploratory call. If success $\to$ `CLOSED`, if failure $\to$ `OPEN`.

### 3. Gateway Workflow: `HybridJevGateway`
1. **Tier 1 Evaluation**: Invoke `local_engine.evaluate_choice()`.
2. **Confidence Check**:
   - Check if $p_1 \ge \tau$ and `is_abstained` is False.
   - If True $\to$ return decision as `tier="Tier1_Local"`, `cost_units=0`.
3. **Tier 2 Cloud Attempt**:
   - Check `circuit_breaker.can_attempt()`.
   - If permitted, try `cloud_engine.evaluate_choice()`.
   - If success $\to$ record success, return decision as `tier="Tier2_Cloud"`, `cost_units=1`.
4. **Fallback Handler**:
   - On exception or open circuit $\to$ record failure, degrade to local decision, set `tier="Tier1_Fallback"`, `degraded=True`, preserve tentative candidate.
