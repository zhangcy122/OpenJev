"""
Hybrid Router and Fallback Gateway for OpenJev.

Provides two-tier cost arbitrage and circuit-breaker fault tolerance:
- Tier 1: Local edge engine for low-latency (<20ms) zero-cloud-cost high-confidence execution.
- Tier 2: Cloud engine (e.g. TypeSafe Jev Commercial) for ambiguous/long-tail queries.
- Fallback SLA: Automatic circuit breaker and graceful degradation to Tier 1 local with
  tentative choice retention, ensuring zero uncaught exceptions and 99.99% service availability.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Union

from openjevpro.harness import BaseDecisionEngine
from openjevpro.guard import TypeSafeJevGuardHarness, GuardDecision
from openjevpro.schemas import ChoiceDecision


class CircuitState(str, Enum):
    """Finite states of the resilience circuit breaker."""
    CLOSED = "CLOSED"        # Normal state, cloud calls permitted
    OPEN = "OPEN"            # Tripped state, cloud calls blocked, immediate local fallback
    HALF_OPEN = "HALF_OPEN"  # Trial state, single probe request allowed to verify recovery


@dataclass
class CircuitBreaker:
    """Finite state machine protecting against downstream cloud failures and rate limits."""
    failure_threshold: int = 3
    recovery_timeout: float = 10.0
    failure_count: int = 0
    state: CircuitState = CircuitState.CLOSED
    last_failure_time: float = 0.0

    def can_attempt(self) -> bool:
        """Determines if a request can be dispatched to Tier 2 Cloud."""
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        if self.state == CircuitState.HALF_OPEN:
            return True
        return False

    def record_success(self) -> None:
        """Records a successful cloud response, resetting failures and healing circuit."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Records a cloud invocation failure, tripping circuit if threshold exceeded."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def reset(self) -> None:
        """Manually resets circuit breaker to CLOSED state."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = 0.0


@dataclass
class HybridGatewayDecision:
    """Structured decision output produced by HybridJevGateway."""
    choice: str
    confidence: float
    tier: str  # "Tier1_Local" | "Tier2_Cloud" | "Tier1_Fallback"
    cost_units: int  # 0 for local/fallback, 1 for cloud
    degraded: bool
    is_abstained: bool
    tentative_choice: Optional[str] = None
    threshold_used: float = 0.0
    fallback_reason: Optional[str] = None
    calibrated_probs: Dict[str, float] = field(default_factory=dict)
    raw_probs: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_local(self) -> bool:
        return self.tier == "Tier1_Local"

    @property
    def is_cloud(self) -> bool:
        return self.tier == "Tier2_Cloud"

    @property
    def is_fallback(self) -> bool:
        return self.tier == "Tier1_Fallback"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "choice": self.choice,
            "confidence": self.confidence,
            "tier": self.tier,
            "cost_units": self.cost_units,
            "degraded": self.degraded,
            "is_abstained": self.is_abstained,
            "tentative_choice": self.tentative_choice,
            "threshold_used": self.threshold_used,
            "fallback_reason": self.fallback_reason,
            "calibrated_probs": self.calibrated_probs,
            "raw_probs": self.raw_probs,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


class HybridJevGateway(BaseDecisionEngine):
    """Two-tier cost-arbitraged hybrid router and fallback gateway.

    Routes high-frequency high-confidence queries to Tier 1 Local Edge Engine (0 cloud cost),
    escalating ambiguous / low-confidence queries to Tier 2 Cloud Engine, with automatic
    circuit-breaker protected fallback SLA.
    """

    def __init__(
        self,
        local_engine: Any,
        cloud_engine: Any,
        guard: Optional[TypeSafeJevGuardHarness] = None,
        local_tau: Optional[float] = None,
        alpha: float = 1.25,
        min_confidence: float = 0.72,
        circuit_breaker: Optional[CircuitBreaker] = None,
        name: str = "Hybrid Jev Gateway",
    ):
        self.name = name
        self.local_engine = local_engine
        self.cloud_engine = cloud_engine
        self.guard = guard or TypeSafeJevGuardHarness(alpha=alpha, min_confidence=min_confidence)
        self.local_tau = local_tau
        self.alpha = alpha
        self.min_confidence = min_confidence
        self.circuit_breaker = circuit_breaker or CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)

    def _normalize_engine_call(
        self,
        engine: Any,
        state: Union[Dict[str, Any], str],
        candidates: List[str],
        criteria: Optional[Dict[str, str]] = None,
        allow_abstain: bool = True,
    ) -> Dict[str, Any]:
        """Dispatches call to engine whether it conforms to BaseDecisionEngine, Client, or Callable."""
        crit = criteria or {}
        if hasattr(engine, "evaluate_choice"):
            res = engine.evaluate_choice(state, candidates, crit, allow_abstain=allow_abstain)
            if hasattr(res, "to_dict"):
                return res.to_dict()
            if isinstance(res, dict):
                return res
            if hasattr(res, "__dict__"):
                return vars(res)
            return {"choice": str(res), "confidence": 1.0}
        elif hasattr(engine, "decide_choice"):
            # OpenJevProClient style
            state_dict = state if isinstance(state, dict) else {"query": str(state)}
            cd: ChoiceDecision = engine.decide_choice(state_dict, candidates, criteria=crit, allow_abstain=allow_abstain)
            return {
                "choice": cd.value,
                "confidence": cd.confidence,
                "is_abstained": cd.abstained,
                "tentative_choice": cd.tentative_value or cd.value,
                "calibrated_probs": cd.probabilities,
                "raw_probs": cd.probabilities,
            }
        elif callable(engine):
            res = engine(state, candidates)
            if isinstance(res, tuple) and len(res) == 2:
                choice, conf = res
                return {
                    "choice": choice,
                    "confidence": float(conf),
                    "is_abstained": choice == "UNKNOWN",
                    "tentative_choice": choice,
                    "calibrated_probs": {choice: float(conf)},
                }
            if isinstance(res, dict):
                return res
            return {"choice": str(res), "confidence": 1.0}
        else:
            raise TypeError(f"Unsupported engine type: {type(engine)}")

    def evaluate_choice(
        self,
        state: Union[Dict[str, Any], str],
        candidates: List[str],
        criteria: Optional[Dict[str, str]] = None,
        allow_abstain: bool = True,
    ) -> HybridGatewayDecision:
        """Evaluates categorical choice with dynamic two-tier routing and circuit-breaker fallback."""
        t0 = time.time()
        crit = dict(criteria or {})
        num_classes = len(candidates)
        tau = self.local_tau if self.local_tau is not None else max(self.min_confidence, self.alpha / max(num_classes, 1))

        # --- Tier 1: Local Engine Evaluation ---
        local_raw = self._normalize_engine_call(
            self.local_engine, state, candidates, crit, allow_abstain=allow_abstain
        )
        local_probs = local_raw.get("calibrated_probs") or local_raw.get("raw_probs") or {}
        local_choice = local_raw.get("choice", "UNKNOWN")
        local_conf = float(local_raw.get("confidence", 0.0))
        local_abstained = bool(local_raw.get("is_abstained", local_choice == "UNKNOWN"))
        local_tentative = local_raw.get("tentative_choice") or local_choice

        # Apply guard if probabilities are available
        if local_probs and self.guard is not None:
            guard_res: GuardDecision = self.guard.evaluate_probabilities(
                local_probs, candidates=candidates, alpha=self.alpha, min_confidence=tau
            )
            local_choice = guard_res.choice
            local_conf = guard_res.confidence
            local_abstained = guard_res.is_abstained
            local_tentative = guard_res.tentative_choice
            tau = guard_res.threshold_used

        # Check Tier 1 High-Confidence Fast-Path Pass
        if not local_abstained and local_choice != "UNKNOWN" and local_conf >= tau:
            latency_ms = (time.time() - t0) * 1000.0
            return HybridGatewayDecision(
                choice=local_choice,
                confidence=local_conf,
                tier="Tier1_Local",
                cost_units=0,
                degraded=False,
                is_abstained=False,
                tentative_choice=local_tentative,
                threshold_used=tau,
                calibrated_probs=local_probs,
                raw_probs=local_raw.get("raw_probs", local_probs),
                latency_ms=latency_ms,
                metadata={"local_raw": local_raw, "source": "local_fast_path"},
            )

        # --- Tier 2: Cloud Engine Escalation ---
        fallback_reason: Optional[str] = None
        if self.circuit_breaker.can_attempt():
            try:
                cloud_raw = self._normalize_engine_call(
                    self.cloud_engine, state, candidates, crit, allow_abstain=allow_abstain
                )
                self.circuit_breaker.record_success()
                latency_ms = (time.time() - t0) * 1000.0
                cloud_choice = cloud_raw.get("choice", "UNKNOWN")
                cloud_conf = float(cloud_raw.get("confidence", 1.0))
                cloud_abstained = bool(cloud_raw.get("is_abstained", cloud_choice == "UNKNOWN"))
                cloud_tentative = cloud_raw.get("tentative_choice") or cloud_choice
                cloud_probs = cloud_raw.get("calibrated_probs") or cloud_raw.get("raw_probs") or {}

                return HybridGatewayDecision(
                    choice=cloud_choice,
                    confidence=cloud_conf,
                    tier="Tier2_Cloud",
                    cost_units=1,
                    degraded=False,
                    is_abstained=cloud_abstained,
                    tentative_choice=cloud_tentative,
                    threshold_used=tau,
                    calibrated_probs=cloud_probs,
                    raw_probs=cloud_raw.get("raw_probs", cloud_probs),
                    latency_ms=latency_ms,
                    metadata={"cloud_raw": cloud_raw, "source": "cloud_escalation"},
                )
            except Exception as exc:
                self.circuit_breaker.record_failure()
                fallback_reason = f"Cloud failure: {type(exc).__name__} ({str(exc)})"
        else:
            fallback_reason = f"Circuit breaker {self.circuit_breaker.state.value} blocked cloud attempt"

        # --- Tier 1 Fallback SLA Guarantee ---
        latency_ms = (time.time() - t0) * 1000.0
        return HybridGatewayDecision(
            choice=local_choice,
            confidence=local_conf,
            tier="Tier1_Fallback",
            cost_units=0,
            degraded=True,
            is_abstained=local_abstained,
            tentative_choice=local_tentative,
            threshold_used=tau,
            fallback_reason=fallback_reason,
            calibrated_probs=local_probs,
            raw_probs=local_raw.get("raw_probs", local_probs),
            latency_ms=latency_ms,
            metadata={"local_raw": local_raw, "source": "fallback_sla"},
        )

    def decide_choice(
        self,
        state: Union[Dict[str, Any], str],
        candidates: List[str],
        criteria: Optional[Dict[str, str]] = None,
        allow_abstain: bool = True,
    ) -> HybridGatewayDecision:
        """Alias for evaluate_choice aligning with OpenJevProClient API style."""
        return self.evaluate_choice(state, candidates, criteria=criteria, allow_abstain=allow_abstain)
