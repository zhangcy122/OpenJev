import unittest
from unittest.mock import MagicMock, patch
from openjevpro.schemas import ChoiceDecision
from openjevpro.client import OpenJevProClient
from openjevpro.flywheel import (
    DeliberativeDecisionFlywheel,
    CrystallizationStore,
    CrystallizationOperator,
)

class TestDeliberativeDecisionFlywheel(unittest.TestCase):
    def test_fast_path_passthrough(self):
        """High-confidence System 1 decisions must return immediately without invoking System 2."""
        mock_client = MagicMock(spec=OpenJevProClient)
        # Fast path returns high confidence decision
        mock_client.decide_choice.return_value = ChoiceDecision(
            value="card_arrival",
            probabilities={"card_arrival": 0.95, "change_pin": 0.05, "UNKNOWN": 0.0},
            confidence=0.95,
            abstained=False,
        )
        mock_client._get_effective_threshold.return_value = 0.45

        mock_reasoner = MagicMock()

        flywheel = DeliberativeDecisionFlywheel(
            fast_engine=mock_client,
            reasoning_engine=mock_reasoner,
        )

        decision = flywheel.evaluate_choice(
            state={"query": "where is my new card?"},
            candidates=["card_arrival", "change_pin"],
            criteria="Classify bank query"
        )

        self.assertEqual(decision.value, "card_arrival")
        self.assertFalse(decision.abstained)
        self.assertFalse(decision.escalated)
        self.assertIsNotNone(decision.crystallization_receipt)
        self.assertFalse(decision.crystallization_receipt["escalated"])
        # System 2 must never have been called
        mock_reasoner.assert_not_called()

    def test_exploration_escalation_and_crystallization(self):
        """Ambiguous query must trigger System 2 deliberation and crystallize criteria."""
        mock_client = MagicMock(spec=OpenJevProClient)
        # Fast path abstains initially
        mock_client.decide_choice.return_value = ChoiceDecision(
            value="UNKNOWN",
            probabilities={"card_arrival": 0.35, "lost_or_stolen_card": 0.35, "UNKNOWN": 0.30},
            confidence=0.35,
            abstained=True,
            tentative_value="lost_or_stolen_card"
        )
        mock_client._get_effective_threshold.return_value = 0.45

        def mock_reasoning_fn(state, candidates, criteria, tentative):
            return {
                "winning_candidate": "lost_or_stolen_card",
                "justification": "Interception threat elevates delivery inquiry to theft risk.",
                "discriminative_rule": "Treat suspected interception as stolen card."
            }

        store = CrystallizationStore()
        flywheel = DeliberativeDecisionFlywheel(
            fast_engine=mock_client,
            reasoning_engine=mock_reasoning_fn,
            crystallization_store=store,
            auto_crystallize=True,
        )

        decision = flywheel.evaluate_choice(
            state={"query": "card not received, suspect intercepted by someone"},
            candidates=["card_arrival", "lost_or_stolen_card"],
            criteria={"card_arrival": "Delivery status", "lost_or_stolen_card": "Stolen cards"}
        )

        self.assertEqual(decision.value, "lost_or_stolen_card")
        self.assertFalse(decision.abstained)
        self.assertTrue(decision.escalated)
        self.assertIsNotNone(decision.crystallization_receipt)
        self.assertTrue(decision.crystallization_receipt["crystallized"])
        self.assertIn("Interception threat", decision.crystallization_receipt["reasoning_trace"])

        # Check that criteria in store was augmented
        aug = store.get_augmented_criteria({"lost_or_stolen_card": "Stolen cards"}, "lost_or_stolen_card")
        self.assertIn("Distilled boundaries", aug)
        self.assertIn("Treat suspected interception as stolen card", aug)

    def test_fast_path_promotion(self):
        """After crystallization, subsequent similar queries must hit System 1 fast path without re-escalation."""
        # Simulated dynamic client that improves once criteria contains distilled boundary
        class DynamicFastClient:
            def __init__(self):
                self.calls = 0

            def _get_effective_threshold(self, n):
                return 0.45

            def decide_choice(self, state, candidates, criteria="", allow_abstain=True, order_invariant=False):
                self.calls += 1
                crit_text = str(criteria)
                # If criteria was crystallized with the new boundary rule, confidence jumps to 0.92
                if "intercepted" in crit_text or "Distilled" in crit_text:
                    return ChoiceDecision(
                        value="lost_or_stolen_card",
                        probabilities={"lost_or_stolen_card": 0.92, "card_arrival": 0.08, "UNKNOWN": 0.0},
                        confidence=0.92,
                        abstained=False,
                    )
                # Cold start: abstained
                return ChoiceDecision(
                    value="UNKNOWN",
                    probabilities={"lost_or_stolen_card": 0.35, "card_arrival": 0.35, "UNKNOWN": 0.30},
                    confidence=0.35,
                    abstained=True,
                    tentative_value="lost_or_stolen_card"
                )

        dynamic_client = DynamicFastClient()
        reasoning_mock = MagicMock()
        reasoning_mock.return_value = {
            "winning_candidate": "lost_or_stolen_card",
            "justification": "Interception equals theft",
            "discriminative_rule": "Flag intercepted as stolen"
        }

        flywheel = DeliberativeDecisionFlywheel(
            fast_engine=dynamic_client,
            reasoning_engine=reasoning_mock,
            auto_crystallize=True
        )

        # Call 1: Cold start -> Escalates to System 2
        d1 = flywheel.evaluate_choice(
            state={"query": "card intercepted"},
            candidates=["card_arrival", "lost_or_stolen_card"],
            criteria={"card_arrival": "Delivery", "lost_or_stolen_card": "Lost/stolen"}
        )
        self.assertEqual(d1.value, "lost_or_stolen_card")
        self.assertTrue(d1.escalated)
        self.assertEqual(reasoning_mock.call_count, 1)

        # Call 2: Second identical query -> Fast-path promotion! (Hits System 1 directly, 0 new System 2 calls)
        d2 = flywheel.evaluate_choice(
            state={"query": "card intercepted"},
            candidates=["card_arrival", "lost_or_stolen_card"],
            criteria={"card_arrival": "Delivery", "lost_or_stolen_card": "Lost/stolen"}
        )
        self.assertEqual(d2.value, "lost_or_stolen_card")
        self.assertFalse(d2.escalated)
        self.assertFalse(d2.abstained)
        self.assertEqual(reasoning_mock.call_count, 1)  # System 2 count remains 1!

    def test_client_convenience_flywheel(self):
        """Test OpenJevProClient.create_decision_flywheel and decide_with_crystallization."""
        client = OpenJevProClient(base_url="http://mock:8000/v1")
        flywheel = client.create_decision_flywheel()
        self.assertIsInstance(flywheel, DeliberativeDecisionFlywheel)
        self.assertIs(flywheel.fast_engine, client)

if __name__ == "__main__":
    unittest.main()
