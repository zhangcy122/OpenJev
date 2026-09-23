import unittest
from unittest.mock import patch, MagicMock
from openjevpro.schemas import ChoiceDecision
from openjevpro.client import OpenJevProClient
from openjevpro.harness import TypeSafeJevEngine, OpenJevProHarness

class TestHarnessAndClient(unittest.TestCase):
    def test_choice_decision_abstention_contract(self):
        """When abstained is True, value must be normalized to 'UNKNOWN' while preserving tentative_value."""
        client = OpenJevProClient(
            base_url="http://mock:11434",
            abstain_threshold=0.50
        )
        # Mocking Ollama chat response with low confidence (max calibrated < 0.50)
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": '{"choice": "card_arrival", "scores": {"card_arrival": 0.2, "change_pin": 0.1}}'
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response):
            decision = client.decide_choice(
                state={"query": "random question"},
                candidates=["card_arrival", "change_pin"],
                criteria="Select one category"
            )

            self.assertTrue(decision.abstained)
            self.assertEqual(decision.value, "UNKNOWN")
            self.assertEqual(decision.tentative_value, "card_arrival")

    def test_typesafe_jev_engine_symmetric_abstain(self):
        """TypeSafeJevEngine must inject UNKNOWN to criteria when allow_abstain=True."""
        engine = TypeSafeJevEngine(api_key="mock_key")
        original_criteria = {
            "card_arrival": "Delivery status of cards",
            "change_pin": "Change security PIN"
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "answers": {
                "decision": {
                    "choice": "UNKNOWN",
                    "confidence": 0.99,
                    "probabilities": {"UNKNOWN": 0.99, "card_arrival": 0.01}
                }
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            res = engine.evaluate_choice(
                state={"query": "what is Python"},
                candidates=["card_arrival", "change_pin"],
                criteria=original_criteria,
                allow_abstain=True
            )
            # Verify UNKNOWN was injected into the criteria payload sent over wire
            sent_payload = mock_post.call_args[1]["json"]
            sent_criteria = sent_payload["questions"]["decision"]["criteria"]
            self.assertIn("UNKNOWN", sent_criteria)
            # Verify original input dictionary was not mutated
            self.assertNotIn("UNKNOWN", original_criteria)
            # Verify result fields
            self.assertTrue(res["abstained"])
            self.assertEqual(res["choice"], "UNKNOWN")

    def test_direct_structured_engine_symmetric_abstain(self):
        """DirectStructuredEngine must inject UNKNOWN into criteria and prompt when allow_abstain=True."""
        from openjevpro.harness import DirectStructuredEngine
        engine = DirectStructuredEngine(base_url="http://mock:11434", model="test-model")
        original_criteria = {
            "card_arrival": "Delivery status of cards",
            "change_pin": "Change security PIN"
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": '{"choice": "UNKNOWN", "confidence": 0.95}'
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            res = engine.evaluate_choice(
                state={"query": "what is Python"},
                candidates=["card_arrival", "change_pin"],
                criteria=original_criteria,
                allow_abstain=True
            )
            # Verify UNKNOWN was injected into prompt sent to model
            sent_prompt = mock_post.call_args[1]["json"]["messages"][0]["content"]
            self.assertIn("- UNKNOWN:", sent_prompt)
            # Verify original criteria was not mutated
            self.assertNotIn("UNKNOWN", original_criteria)
            # Verify result fields
            self.assertTrue(res["abstained"])
            self.assertEqual(res["choice"], "UNKNOWN")

    def test_compute_ece(self):
        """Verify automated ECE calculation on synthetic records."""
        records = [
            {
                "ground_truth": "card_arrival",
                "engine_results": {
                    "TestEngine": {"choice": "card_arrival", "confidence": 0.9, "abstained": False}
                }
            },
            {
                "ground_truth": "UNKNOWN",
                "engine_results": {
                    "TestEngine": {"choice": "UNKNOWN", "confidence": 0.8, "abstained": True}
                }
            }
        ]
        ece = OpenJevProHarness.compute_ece(records, "TestEngine", n_bins=10)
        self.assertIsInstance(ece, float)
        self.assertGreaterEqual(ece, 0.0)

if __name__ == "__main__":
    unittest.main()
