import json
import math
import logging
from typing import Dict, Any, Type, Union, List, Optional
from enum import Enum
import requests

from openjevpro.schemas import ChoiceDecision, NoulDecision
from openjevpro.calibrator import TemperatureCalibrator

logger = logging.getLogger("openjevpro.client")

class OpenJevProClient:
    """Client for querying open-source LLM APIs with Jev-style typed probabilistic decisions."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "Qwen/Qwen3-4B-Instruct",
        temperature_scaling: float = 1.25,
        abstain_threshold: Union[float, str] = 0.45,
        backend: str = "auto",
        use_chat: bool = True,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.calibrator = TemperatureCalibrator(temperature=temperature_scaling)
        self.abstain_threshold = abstain_threshold
        self.use_chat = use_chat
        self.chat_template_kwargs = chat_template_kwargs or {}

        if backend == "auto":
            if "11434" in self.base_url:
                self.backend = "ollama"
            else:
                self.backend = "openai"
        else:
            self.backend = backend.lower()

    def _get_effective_threshold(self, n_options: int) -> float:
        """Computes the effective confidence threshold for abstention.
        If abstain_threshold is 'auto', scales inversely with candidate count (1.25 / n_options).
        """
        if isinstance(self.abstain_threshold, str) and self.abstain_threshold.lower() == "auto":
            return 1.25 / max(n_options, 1)
        return float(self.abstain_threshold)

    def decide_choice(
        self,
        state: Dict[str, Any],
        candidates: Union[Type[Enum], List[str]],
        criteria: Union[str, Dict[str, str]] = "",
        allow_abstain: bool = True
    ) -> ChoiceDecision:
        """Evaluates a categorical choice decision across the given candidates with calibrated probabilities."""
        if isinstance(candidates, type) and issubclass(candidates, Enum):
            options = [e.value for e in candidates]
        else:
            options = list(candidates)

        if allow_abstain and "UNKNOWN" not in options:
            options.append("UNKNOWN")

        if self.backend == "ollama":
            return self._decide_choice_ollama(state, options, criteria, allow_abstain)
        else:
            return self._decide_choice_openai(state, options, criteria, allow_abstain)

    def _decide_choice_ollama(
        self,
        state: Dict[str, Any],
        options: List[str],
        criteria: Union[str, Dict[str, str]],
        allow_abstain: bool
    ) -> ChoiceDecision:
        """Evaluates categorical choice via Ollama structured scoring + temperature calibration."""
        if isinstance(criteria, dict):
            crit_text = "\n".join([f"- {k}: {v}" for k, v in criteria.items()])
        else:
            crit_text = str(criteria)

        prompt = (
            f"You are a probabilistic decision engine.\n\n"
            f"State / Context:\n{json.dumps(state, ensure_ascii=False, indent=2)}\n\n"
            f"Evaluation Criteria:\n{crit_text}\n\n"
            f"Candidate Options:\n" + "\n".join([f"- {opt}" for opt in options]) + "\n\n"
            f"Rate the relative likelihood score (0.0 to 10.0) of each candidate being the single correct choice.\n"
            f"Output ONLY a valid JSON object matching this schema:\n"
            f'{{"scores": {json.dumps({opt: 0.0 for opt in options})}}}'
        )

        ollama_endpoint = f"{self.base_url}/api/chat" if not self.base_url.endswith("/api") else f"{self.base_url}/chat"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "format": "json",
            "think": False,
            "stream": False,
        }

        resp = requests.post(ollama_endpoint, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        content = data["message"]["content"].strip()
        if "```" in content:
            parts = content.split("```")
            for p in parts:
                p_clean = p.strip()
                if p_clean.startswith("json"):
                    p_clean = p_clean[4:].strip()
                if p_clean.startswith("{") and p_clean.endswith("}"):
                    content = p_clean
                    break
        parsed = json.loads(content)
        raw_scores = parsed.get("scores", {})

        extracted_logits: Dict[str, float] = {}
        for opt in options:
            extracted_logits[opt] = float(raw_scores.get(opt, 0.0))

        calibrated_probs = self.calibrator.calibrate(extracted_logits)
        best_choice = max(calibrated_probs, key=calibrated_probs.get)
        confidence = calibrated_probs[best_choice]

        effective_thresh = self._get_effective_threshold(len(options))
        abstained = False
        if (allow_abstain and best_choice == "UNKNOWN") or confidence < effective_thresh:
            abstained = True

        final_value = "UNKNOWN" if abstained else best_choice
        tentative_val = best_choice if (abstained and best_choice != "UNKNOWN") else None

        return ChoiceDecision(
            value=final_value,
            probabilities=calibrated_probs,
            confidence=confidence,
            abstained=abstained,
            tentative_value=tentative_val,
            raw_logits=extracted_logits,
        )

    @staticmethod
    def _extract_top_logprobs(logprobs_obj: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Extracts candidate token logprobs from either legacy completions or modern chat completions format."""
        if not logprobs_obj or not isinstance(logprobs_obj, dict):
            return {}

        # Format 1: Legacy completions format: {"top_logprobs": [{"A": -0.1, ...}]}
        legacy = logprobs_obj.get("top_logprobs")
        if isinstance(legacy, list) and legacy and isinstance(legacy[0], dict):
            if all(isinstance(v, (int, float)) for v in legacy[0].values()):
                return {str(k): float(v) for k, v in legacy[0].items()}

        # Format 2: Modern chat completions format: {"content": [{"token": "A", "logprob": -0.1, "top_logprobs": [...]}]}
        content = logprobs_obj.get("content")
        if isinstance(content, list) and content:
            out: Dict[str, float] = {}
            first_token_data = content[0]
            if isinstance(first_token_data, dict):
                top_list = first_token_data.get("top_logprobs") or []
                for item in top_list:
                    if isinstance(item, dict):
                        tok = item.get("token")
                        if tok is not None and "logprob" in item:
                            out.setdefault(str(tok), float(item["logprob"]))
                if out:
                    return out
                if "token" in first_token_data and "logprob" in first_token_data:
                    return {str(first_token_data["token"]): float(first_token_data["logprob"])}

        return {}

    def _decide_choice_openai(
        self,
        state: Dict[str, Any],
        options: List[str],
        criteria: Union[str, Dict[str, str]],
        allow_abstain: bool
    ) -> ChoiceDecision:
        """Evaluates categorical choice via vLLM/OpenAI completions or chat/completions logprobs."""
        letters = [chr(65 + i) for i in range(len(options))]
        option_map = {letter: opt for letter, opt in zip(letters, options)}

        if isinstance(criteria, dict):
            crit_text = "\n".join([f"- {k}: {v}" for k, v in criteria.items()])
        else:
            crit_text = str(criteria)

        options_prompt = "\n".join([f"{letter}. {opt}" for letter, opt in option_map.items()])
        prompt = (
            f"Given the following state:\n{json.dumps(state, ensure_ascii=False, indent=2)}\n\n"
            f"Evaluation criteria:\n{crit_text}\n\n"
            f"Select the single best option from the list below:\n{options_prompt}\n\n"
            f"Reply with ONLY the option letter (e.g. A, B, C):"
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data: Optional[Dict[str, Any]] = None
        endpoint_used = ""

        if self.use_chat:
            system_msg = "You are a precise, single-token categorical decision classifier. Reply with ONLY the option letter (e.g. A, B, C)."
            chat_payload: Dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": 20,
            }
            if self.chat_template_kwargs:
                chat_payload["extra_body"] = {"chat_template_kwargs": self.chat_template_kwargs}

            endpoint_used = f"{self.base_url}/chat/completions"
            resp = requests.post(endpoint_used, headers=headers, json=chat_payload, timeout=30)
            if resp.status_code in (404, 405):
                # Fallback to legacy completions if chat endpoint is not supported by server
                endpoint_used = f"{self.base_url}/completions"
                comp_payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "logprobs": 20
                }
                resp = requests.post(endpoint_used, headers=headers, json=comp_payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        else:
            endpoint_used = f"{self.base_url}/completions"
            comp_payload = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": 20
            }
            resp = requests.post(endpoint_used, headers=headers, json=comp_payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()

        choice_data = data["choices"][0]
        logprobs_data = choice_data.get("logprobs")
        choice_logprobs = self._extract_top_logprobs(logprobs_data)

        if not choice_logprobs:
            logger.warning(
                "Endpoint '%s' returned empty or unparseable logprobs. "
                "Ensure logprobs are supported and enabled on the server.",
                endpoint_used
            )

        extracted_logits: Dict[str, float] = {}
        matched_count = 0
        for letter, opt in option_map.items():
            l_prob = choice_logprobs.get(letter)
            if l_prob is None:
                l_prob = choice_logprobs.get(f" {letter}")
            if l_prob is None:
                extracted_logits[opt] = -100.0
            else:
                extracted_logits[opt] = float(l_prob)
                matched_count += 1

        if choice_logprobs and matched_count == 0:
            logger.warning(
                "None of the candidate letters %s appeared in top logprobs: %s",
                list(option_map.keys()),
                list(choice_logprobs.keys())
            )

        calibrated_probs = self.calibrator.calibrate(extracted_logits)
        best_choice = max(calibrated_probs, key=calibrated_probs.get)
        confidence = calibrated_probs[best_choice]

        effective_thresh = self._get_effective_threshold(len(options))
        abstained = False
        if (allow_abstain and best_choice == "UNKNOWN") or confidence < effective_thresh:
            abstained = True

        final_value = "UNKNOWN" if abstained else best_choice
        tentative_val = best_choice if (abstained and best_choice != "UNKNOWN") else None

        return ChoiceDecision(
            value=final_value,
            probabilities=calibrated_probs,
            confidence=confidence,
            abstained=abstained,
            tentative_value=tentative_val,
            raw_logits=extracted_logits,
        )

    def decide_noul(
        self,
        state: Dict[str, Any],
        assertion: str,
    ) -> NoulDecision:
        """Evaluates a binary truth assertion judgment."""
        options = ["TRUE", "FALSE"]
        decision = self.decide_choice(
            state=state,
            candidates=options,
            criteria=f"Evaluate whether the following assertion is strictly TRUE or FALSE: {assertion}",
            allow_abstain=False
        )
        p_true = decision.probabilities.get("TRUE", 0.5)
        value = p_true >= 0.5
        conf = p_true if value else (1.0 - p_true)
        return NoulDecision(
            value=value,
            probability_true=p_true,
            confidence=conf,
            abstained=decision.abstained
        )

    def create_hybrid_gateway(
        self,
        cloud_engine: Any,
        local_tau: Optional[float] = None,
        alpha: float = 1.25,
        min_confidence: float = 0.72,
        failure_threshold: int = 3,
        recovery_timeout: float = 10.0,
    ) -> Any:
        """Creates a resilient HybridJevGateway using this client as the Tier 1 Local Edge Engine."""
        from openjevpro.gateway import HybridJevGateway, CircuitBreaker
        cb = CircuitBreaker(failure_threshold=failure_threshold, recovery_timeout=recovery_timeout)
        return HybridJevGateway(
            local_engine=self,
            cloud_engine=cloud_engine,
            local_tau=local_tau,
            alpha=alpha,
            min_confidence=min_confidence,
            circuit_breaker=cb,
        )

# Backward compatibility alias
OpenJevClient = OpenJevProClient
