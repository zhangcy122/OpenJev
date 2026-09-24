import json
import math
from typing import Dict, Any, Type, Union, List, Optional
from enum import Enum
import requests

from openjevpro.schemas import ChoiceDecision, NoulDecision
from openjevpro.calibrator import TemperatureCalibrator

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
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.calibrator = TemperatureCalibrator(temperature=temperature_scaling)
        self.abstain_threshold = abstain_threshold

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

    def _decide_choice_openai(
        self,
        state: Dict[str, Any],
        options: List[str],
        criteria: Union[str, Dict[str, str]],
        allow_abstain: bool
    ) -> ChoiceDecision:
        """Evaluates categorical choice via vLLM/OpenAI completions logprobs."""
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
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 20
        }

        resp = requests.post(f"{self.base_url}/completions", headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        choice_logprobs = data["choices"][0].get("logprobs", {}).get("top_logprobs", [{}])[0]

        extracted_logits: Dict[str, float] = {}
        for letter, opt in option_map.items():
            l_prob = choice_logprobs.get(letter) or choice_logprobs.get(f" {letter}") or -100.0
            extracted_logits[opt] = float(l_prob)

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

# Backward compatibility alias
OpenJevClient = OpenJevProClient
