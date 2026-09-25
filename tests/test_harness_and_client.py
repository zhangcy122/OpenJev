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

    def test_auto_abstain_threshold(self):
        """When abstain_threshold='auto', effective threshold scales inversely with candidate count."""
        client = OpenJevProClient(base_url="http://mock:11434", abstain_threshold="auto")
        # 10 options -> 1.25 / 10 = 0.125
        self.assertAlmostEqual(client._get_effective_threshold(10), 0.125)
        # 5 options -> 1.25 / 5 = 0.25
        self.assertAlmostEqual(client._get_effective_threshold(5), 0.25)

    def test_harness_selective_and_tentative_metrics(self):
        """Verify selective_accuracy and tentative_accuracy computation."""
        records = [
            {
                "ground_truth": "card_arrival",
                "engine_results": {
                    "TestEngine": {
                        "choice": "UNKNOWN",
                        "tentative_value": "card_arrival",
                        "abstained": True,
                        "confidence": 0.25,
                        "latency_ms": 100.0
                    }
                }
            },
            {
                "ground_truth": "change_pin",
                "engine_results": {
                    "TestEngine": {
                        "choice": "change_pin",
                        "tentative_value": None,
                        "abstained": False,
                        "confidence": 0.95,
                        "latency_ms": 100.0
                    }
                }
            }
        ]
        harness = OpenJevProHarness(engines=[])
        summary = harness._compute_summary(records)
        metrics = summary["engine_metrics"]["TestEngine"]
        self.assertEqual(metrics["accuracy"], 50.0)  # 1/2
        self.assertEqual(metrics["selective_accuracy"], 100.0)  # 1/1 on answered
        self.assertEqual(metrics["tentative_accuracy"], 100.0)  # 2/2 on tentative
        self.assertEqual(metrics["coverage_rate"], 50.0)  # 1/2 answered

    def test_extract_top_logprobs_legacy_and_modern(self):
        """Test _extract_top_logprobs parses both legacy dict and modern content list structures."""
        # 1. Legacy format
        legacy_obj = {"top_logprobs": [{"A": -0.1, "B": -3.5, " C": -4.0}]}
        res_legacy = OpenJevProClient._extract_top_logprobs(legacy_obj)
        self.assertEqual(res_legacy.get("A"), -0.1)
        self.assertEqual(res_legacy.get("B"), -3.5)

        # 2. Modern chat completions format
        modern_obj = {
            "content": [{
                "token": "A",
                "logprob": -0.05,
                "top_logprobs": [
                    {"token": "A", "logprob": -0.05},
                    {"token": "B", "logprob": -4.20},
                    {"token": "C", "logprob": -6.80}
                ]
            }]
        }
        res_modern = OpenJevProClient._extract_top_logprobs(modern_obj)
        self.assertEqual(res_modern.get("A"), -0.05)
        self.assertEqual(res_modern.get("B"), -4.20)
        self.assertEqual(res_modern.get("C"), -6.80)

        # 3. Modern format single fallback token without nested top_logprobs
        modern_single = {
            "content": [{
                "token": "A",
                "logprob": -0.02
            }]
        }
        res_single = OpenJevProClient._extract_top_logprobs(modern_single)
        self.assertEqual(res_single.get("A"), -0.02)

        # 4. Empty or malformed
        self.assertEqual(OpenJevProClient._extract_top_logprobs(None), {})
        self.assertEqual(OpenJevProClient._extract_top_logprobs({}), {})
        self.assertEqual(OpenJevProClient._extract_top_logprobs({"content": []}), {})

    def test_decide_choice_openai_chat_transport(self):
        """Verify OpenJevProClient uses chat/completions transport with system prompt and logprobs."""
        client = OpenJevProClient(
            base_url="http://mock-vllm:8000/v1",
            use_chat=True,
            chat_template_kwargs={"enable_thinking": False}
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "A"},
                "logprobs": {
                    "content": [{
                        "token": "A",
                        "logprob": -0.02,
                        "top_logprobs": [
                            {"token": "A", "logprob": -0.02},
                            {"token": "B", "logprob": -5.00}
                        ]
                    }]
                }
            }]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp) as mock_post:
            decision = client.decide_choice(
                state={"text": "hello"},
                candidates=["opt_a", "opt_b"],
                criteria="Select option",
                allow_abstain=False
            )

            self.assertEqual(decision.value, "opt_a")
            self.assertFalse(decision.abstained)
            self.assertGreater(decision.confidence, 0.90)

            # Verify endpoint called and payload
            call_args = mock_post.call_args
            self.assertEqual(call_args[0][0], "http://mock-vllm:8000/v1/chat/completions")
            payload = call_args[1]["json"]
            self.assertTrue(payload["logprobs"])
            self.assertEqual(payload["top_logprobs"], 20)
            self.assertIn("extra_body", payload)
            self.assertEqual(payload["extra_body"]["chat_template_kwargs"]["enable_thinking"], False)
            self.assertEqual(len(payload["messages"]), 2)

    def test_decide_choice_openai_chat_fallback_to_completions(self):
        """When chat/completions returns 404, client should fall back gracefully to completions."""
        client = OpenJevProClient(
            base_url="http://legacy-vllm:8000/v1",
            use_chat=True
        )

        resp_404 = MagicMock()
        resp_404.status_code = 404

        resp_completions = MagicMock()
        resp_completions.status_code = 200
        resp_completions.json.return_value = {
            "choices": [{
                "text": "A",
                "logprobs": {
                    "top_logprobs": [{"A": -0.01, "B": -6.0}]
                }
            }]
        }
        resp_completions.raise_for_status = MagicMock()

        with patch("requests.post", side_effect=[resp_404, resp_completions]) as mock_post:
            decision = client.decide_choice(
                state={"text": "fallback test"},
                candidates=["opt_a", "opt_b"],
                allow_abstain=False
            )

            self.assertEqual(decision.value, "opt_a")
            self.assertEqual(mock_post.call_count, 2)
            # First call was to chat/completions
            self.assertEqual(mock_post.call_args_list[0][0][0], "http://legacy-vllm:8000/v1/chat/completions")
            # Second call fell back to completions
            self.assertEqual(mock_post.call_args_list[1][0][0], "http://legacy-vllm:8000/v1/completions")

if __name__ == "__main__":
    unittest.main()

