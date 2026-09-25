import pytest
from openjevpro.calibrator import TemperatureCalibrator
from openjevpro.guard import TypeSafeJevGuardHarness, GuardDecision
from openjevpro.harness import BaseDecisionEngine


class DummyEngine(BaseDecisionEngine):
    def __init__(self, raw_probs, choice="lost_or_stolen_card"):
        self.name = "DummyEngine"
        self.raw_probs = raw_probs
        self.choice = choice

    def evaluate_choice(self, state, candidates, criteria, allow_abstain=True):
        return {
            "choice": self.choice,
            "confidence": max(self.raw_probs.values()) if self.raw_probs else 0.0,
            "probabilities": dict(self.raw_probs),
            "abstained": self.choice == "UNKNOWN",
            "latency_ms": 12.5,
        }


class TestTypeSafeJevGuardHarness:
    def test_high_confidence_passthrough(self):
        harness = TypeSafeJevGuardHarness(alpha=1.25, min_confidence=0.72)
        raw_probs = {"card_arrival": 0.95, "lost_or_stolen_card": 0.05}
        decision = harness.evaluate_probabilities(raw_probs)

        assert decision.choice == "card_arrival"
        assert decision.is_abstained is False
        assert decision.tentative_choice == "card_arrival"
        assert decision.confidence > 0.72

    def test_marginal_confidence_abstains_sample_zero(self):
        """Reproduce the empirical Sample 0 failure where raw prob 0.79 is uncalibrated and wrong."""
        harness = TypeSafeJevGuardHarness(alpha=1.25, min_confidence=0.72)
        raw_probs = {
            "lost_or_stolen_card": 0.79,
            "card_arrival": 0.18,
            "pin_change": 0.01,
            "balance": 0.005,
            "transfer": 0.005,
            "statement": 0.005,
            "support": 0.005,
        }
        decision = harness.evaluate_probabilities(raw_probs)

        # Calibrated top prob drops to ~0.7153, below 0.72 threshold -> safely abstained
        assert decision.choice == "UNKNOWN"
        assert decision.is_abstained is True
        assert decision.tentative_choice == "lost_or_stolen_card"
        assert decision.threshold_used == 0.72
        assert decision.confidence < 0.72

    def test_explicit_unknown_out_of_scope(self):
        harness = TypeSafeJevGuardHarness()
        raw_probs = {"UNKNOWN": 0.88, "card_arrival": 0.12}
        decision = harness.evaluate_probabilities(raw_probs)

        assert decision.choice == "UNKNOWN"
        assert decision.is_abstained is True
        assert decision.tentative_choice == "UNKNOWN"
        assert decision.metadata["reason"] == "explicit_unknown"

    def test_empty_input_handling(self):
        harness = TypeSafeJevGuardHarness()
        decision = harness.evaluate_probabilities({})

        assert decision.choice == "UNKNOWN"
        assert decision.confidence == 0.0
        assert decision.is_abstained is True
        assert decision.tentative_choice == "UNKNOWN"

    def test_degenerate_zero_probabilities_handled_safely(self):
        harness = TypeSafeJevGuardHarness()
        raw_probs = {"A": 0.0, "B": 0.0}
        decision = harness.evaluate_probabilities(raw_probs)

        # Uniform distribution across candidates (0.5 each), both below 0.72 -> abstained
        assert decision.choice == "UNKNOWN"
        assert decision.is_abstained is True
        assert round(decision.confidence, 4) == 0.5000

    def test_compute_threshold_boundaries(self):
        harness = TypeSafeJevGuardHarness(alpha=1.25, min_confidence=0.72)

        # K = 7: max(0.72, 1.25 / 7) = 0.72
        assert harness.compute_threshold(7) == 0.72

        # K = 1: max(0.72, 1.25 / 1) = 1.25
        assert harness.compute_threshold(1) == 1.25

        # K <= 0: min_confidence
        assert harness.compute_threshold(0) == 0.72
        assert harness.compute_threshold(-1) == 0.72

    def test_wrapped_engine_evaluate_choice(self):
        raw_probs = {
            "lost_or_stolen_card": 0.79,
            "card_arrival": 0.18,
            "pin_change": 0.01,
            "balance": 0.005,
            "transfer": 0.005,
            "statement": 0.005,
            "support": 0.005,
        }
        dummy = DummyEngine(raw_probs=raw_probs)
        harness = TypeSafeJevGuardHarness(engine=dummy, alpha=1.25, min_confidence=0.72)

        res = harness.evaluate_choice(
            state={"query": "How do I locate my card?"},
            candidates=list(raw_probs.keys()),
            criteria={},
        )

        assert res["choice"] == "UNKNOWN"
        assert res["abstained"] is True
        assert res["tentative_choice"] == "lost_or_stolen_card"
        assert "guard_decision" in res
        assert isinstance(res["guard_decision"], GuardDecision)
        assert res["latency_ms"] >= 0.0


    def test_guard_convenience_dispatcher(self):
        harness = TypeSafeJevGuardHarness()

        # Passing dict with probabilities key
        d1 = harness.guard({"probabilities": {"A": 0.99, "B": 0.01}})
        assert d1.choice == "A"
        assert d1.is_abstained is False

        # Passing raw prob dict
        d2 = harness.guard({"A": 0.99, "B": 0.01})
        assert d2.choice == "A"

        # Passing GuardDecision returns self
        assert harness.guard(d1) is d1

        # Passing invalid type
        with pytest.raises(TypeError):
            harness.guard(12345)
