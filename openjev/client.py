import json
from typing import Dict, Any, Type, Union, List, Optional
from enum import Enum
import requests

from openjev.schemas import ChoiceDecision, NoulDecision
from openjev.calibrator import TemperatureCalibrator

class OpenJevClient:
    """Client for querying open-source LLM APIs with Jev-style typed probabilistic decisions."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "Qwen/Qwen3-4B-Instruct",
        temperature_scaling: float = 1.25,
        abstain_threshold: float = 0.45,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.calibrator = TemperatureCalibrator(temperature=temperature_scaling)
        self.abstain_threshold = abstain_threshold

    def decide_choice(
        self,
        state: Dict[str, Any],
        candidates: Union[Type[Enum], List[str]],
        criteria: str = "",
        allow_abstain: bool = True
    ) -> ChoiceDecision:
        """Evaluates a categorical choice decision across the given candidates."""
        if isinstance(candidates, type) and issubclass(candidates, Enum):
            options = [e.value for e in candidates]
        else:
            options = list(candidates)

        if allow_abstain and "UNKNOWN" not in options:
            options.append("UNKNOWN")

        # Map options to single-token letters A, B, C...
        letters = [chr(65 + i) for i in range(len(options))]
        option_map = {letter: opt for letter, opt in zip(letters, options)}
        reverse_map = {opt: letter for letter, opt in option_map.items()}

        options_prompt = "\n".join([f"{letter}. {opt}" for letter, opt in option_map.items()])
        prompt = (
            f"Given the following state:\n{json.dumps(state, ensure_ascii=False, indent=2)}\n\n"
            f"Evaluation criteria:\n{criteria}\n\n"
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

        # Extract logprobs of option letters
        choice_logprobs = data["choices"][0].get("logprobs", {}).get("top_logprobs", [{}])[0]
        
        extracted_logits: Dict[str, float] = {}
        for letter, opt in option_map.items():
            # Check letter with or without leading whitespace
            l_prob = choice_logprobs.get(letter) or choice_logprobs.get(f" {letter}") or -100.0
            extracted_logits[opt] = float(l_prob)

        # Calibrate probabilities
        calibrated_probs = self.calibrator.calibrate(extracted_logits)
        best_choice = max(calibrated_probs, key=calibrated_probs.get)
        confidence = calibrated_probs[best_choice]

        abstained = False
        if best_choice == "UNKNOWN" or confidence < self.abstain_threshold:
            abstained = True

        return ChoiceDecision(
            value=best_choice,
            probabilities=calibrated_probs,
            confidence=confidence,
            abstained=abstained,
            raw_logits=extracted_logits
        )
