"""
Unit tests for Hybrid Router and Fallback Gateway (openjevpro.gateway).
"""

import time
import pytest
from typing import Dict, Any, List

from openjevpro.gateway import (
    CircuitState,
    CircuitBreaker,
    HybridGatewayDecision,
    HybridJevGateway,
)
from openjevpro.schemas import ChoiceDecision
from openjevpro.client import OpenJevProClient


class MockEngine:
    """Mock engine allowing configurable response, latency, and simulated failures."""
    def __init__(self, name: str, fixed_response: Dict[str, Any] = None, fail: bool = False, exception_type=RuntimeError):
        self.name = name
        self.fixed_response = fixed_response or {}
        self.fail = fail
        self.exception_type = exception_type
        self.call_count = 0

    def evaluate_choice(
        self,
        state: Any,
        candidates: List[str],
        criteria: Dict[str, str] = None,
        allow_abstain: bool = True,
    ) -> Dict[str, Any]:
        self.call_count += 1
        if self.fail:
            raise self.exception_type(f"{self.name} simulated failure!")
        return dict(self.fixed_response)


class TestCircuitBreaker:
    """Tests for the resilience circuit breaker finite state machine."""

    def test_initial_state_and_success(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        assert cb.state == CircuitState.CLOSED
        assert cb.can_attempt() is True
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_failure_tripping_to_open(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 1
        assert cb.can_attempt() is True

        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 2
        assert cb.can_attempt() is False

    def test_half_open_recovery_and_healing(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.can_attempt() is False

        # Sleep past recovery timeout
        time.sleep(0.06)
        assert cb.can_attempt() is True
        assert cb.state == CircuitState.HALF_OPEN

        # Success heals to CLOSED
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_half_open_failure_re_trips(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.06)
        assert cb.can_attempt() is True
        assert cb.state == CircuitState.HALF_OPEN

        # Failure while HALF_OPEN re-trips to OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN


class TestHybridJevGateway:
    """Tests for the HybridJevGateway two-tier routing and fallback SLA."""

    def test_high_confidence_local_fast_path(self):
        local = MockEngine("Local", fixed_response={
            "choice": "check_balance",
            "confidence": 0.95,
            "is_abstained": False,
            "calibrated_probs": {"check_balance": 0.95, "transfer": 0.03, "UNKNOWN": 0.02},
        })
        cloud = MockEngine("Cloud", fixed_response={
            "choice": "check_balance",
            "confidence": 0.99,
        })

        gateway = HybridJevGateway(local_engine=local, cloud_engine=cloud, local_tau=0.75)
        decision = gateway.evaluate_choice("query", ["check_balance", "transfer", "UNKNOWN"])

        assert decision.tier == "Tier1_Local"
        assert decision.cost_units == 0
        assert decision.degraded is False
        assert decision.choice == "check_balance"
        assert decision.confidence >= 0.75
        assert local.call_count == 1
        assert cloud.call_count == 0  # Cloud was never invoked!

    def test_low_confidence_escalation_to_cloud(self):
        local = MockEngine("Local", fixed_response={
            "choice": "check_balance",
            "confidence": 0.40,  # Below threshold
            "is_abstained": False,
            "calibrated_probs": {"check_balance": 0.40, "transfer": 0.35, "UNKNOWN": 0.25},
        })
        cloud = MockEngine("Cloud", fixed_response={
            "choice": "transfer",
            "confidence": 0.92,
            "is_abstained": False,
            "calibrated_probs": {"transfer": 0.92, "check_balance": 0.08},
        })

        gateway = HybridJevGateway(local_engine=local, cloud_engine=cloud, local_tau=0.75)
        decision = gateway.evaluate_choice("query", ["check_balance", "transfer", "UNKNOWN"])

        assert decision.tier == "Tier2_Cloud"
        assert decision.cost_units == 1
        assert decision.degraded is False
        assert decision.choice == "transfer"
        assert decision.confidence == 0.92
        assert local.call_count == 1
        assert cloud.call_count == 1

    def test_local_abstained_escalates_to_cloud(self):
        local = MockEngine("Local", fixed_response={
            "choice": "UNKNOWN",
            "confidence": 0.85,
            "is_abstained": True,
            "tentative_choice": "check_balance",
        })
        cloud = MockEngine("Cloud", fixed_response={
            "choice": "credit_inquiry",
            "confidence": 0.88,
        })

        gateway = HybridJevGateway(local_engine=local, cloud_engine=cloud, local_tau=0.75)
        decision = gateway.evaluate_choice("query", ["check_balance", "credit_inquiry", "UNKNOWN"])

        assert decision.tier == "Tier2_Cloud"
        assert decision.cost_units == 1
        assert decision.choice == "credit_inquiry"
        assert cloud.call_count == 1

    def test_cloud_failure_triggers_graceful_fallback(self):
        local = MockEngine("Local", fixed_response={
            "choice": "card_activation",
            "confidence": 0.55,  # Marginal
            "tentative_choice": "card_activation",
            "calibrated_probs": {"card_activation": 0.55, "card_lock": 0.45},
        })
        cloud = MockEngine("Cloud", fail=True, exception_type=ConnectionError)

        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        gateway = HybridJevGateway(local_engine=local, cloud_engine=cloud, local_tau=0.75, circuit_breaker=cb)

        decision = gateway.evaluate_choice("query", ["card_activation", "card_lock", "UNKNOWN"])

        # Graceful fallback: No exception raised!
        assert decision.tier == "Tier1_Fallback"
        assert decision.cost_units == 0
        assert decision.degraded is True
        assert decision.is_fallback is True
        assert decision.tentative_choice == "card_activation"
        assert "ConnectionError" in (decision.fallback_reason or "")
        assert cb.failure_count == 1

    def test_circuit_open_prevents_cloud_attempts_and_falls_back_instantly(self):
        local = MockEngine("Local", fixed_response={
            "choice": "card_activation",
            "confidence": 0.50,
        })
        cloud = MockEngine("Cloud", fail=True, exception_type=TimeoutError)

        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=10.0)
        gateway = HybridJevGateway(local_engine=local, cloud_engine=cloud, local_tau=0.75, circuit_breaker=cb)

        # Trigger 2 failures to trip circuit
        gateway.evaluate_choice("q1", ["a", "b"])
        gateway.evaluate_choice("q2", ["a", "b"])
        assert cb.state == CircuitState.OPEN
        assert cloud.call_count == 2

        # 3rd call: Circuit is open, cloud should NOT be called at all
        d3 = gateway.evaluate_choice("q3", ["a", "b"])
        assert d3.tier == "Tier1_Fallback"
        assert d3.degraded is True
        assert cloud.call_count == 2  # Not incremented!
        assert "blocked cloud attempt" in (d3.fallback_reason or "")

    def test_callable_and_client_engine_compatibility(self):
        def local_fn(state, candidates):
            return ("check_balance", 0.92)

        def cloud_fn(state, candidates):
            return ("transfer", 0.99)

        gateway = HybridJevGateway(local_engine=local_fn, cloud_engine=cloud_fn, local_tau=0.80)
        d = gateway.decide_choice("q", ["check_balance", "transfer"])
        assert d.tier == "Tier1_Local"
        assert d.choice == "check_balance"

    def test_client_create_hybrid_gateway_factory(self):
        client = OpenJevProClient(base_url="http://localhost:8000/v1")
        cloud = MockEngine("Cloud", fixed_response={"choice": "transfer", "confidence": 0.9})
        gw = client.create_hybrid_gateway(cloud_engine=cloud, local_tau=0.70)
        assert isinstance(gw, HybridJevGateway)
        assert gw.local_engine is client
        assert gw.cloud_engine is cloud
