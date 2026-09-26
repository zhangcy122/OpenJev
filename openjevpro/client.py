import json
import math
import logging
from typing import Dict, Any, Type, Union, List, Optional, Tuple
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import requests

from openjevpro.schemas import ChoiceDecision, NoulDecision, ScoreDecision
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
        mock: bool = False,
        order_invariant_max_workers: int = 8,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.calibrator = TemperatureCalibrator(temperature=temperature_scaling)
        self.abstain_threshold = abstain_threshold
        self.use_chat = use_chat
        self.chat_template_kwargs = chat_template_kwargs or {}
        # Concurrency for order_invariant=True. Batching servers (llama.cpp, vLLM, SGLang) are not
        # batch-invariant: concurrent requests land in batches whose composition depends on thread
        # timing, and near-tied candidates can then flip between identical calls. 1 = sequential
        # and deterministic, at N x the latency; the default favours throughput.
        self.order_invariant_max_workers = max(1, int(order_invariant_max_workers))
        self.mock = mock or (base_url and base_url.startswith("mock://"))

        if self.mock:
            from openjevpro.mock import MockClient
            self._mock_client = MockClient(
                base_url=self.base_url,
                model=self.model,
                temperature_scaling=temperature_scaling,
                abstain_threshold=abstain_threshold,
            )
        else:
            self._mock_client = None

        if backend == "auto":
            if "11434" in self.base_url:
                self.backend = "ollama"
            elif "laya" in self.base_url.lower() or "8001" in self.base_url:
                self.backend = "laya"
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

    @staticmethod
    def _select_best(probs: Dict[str, float]) -> Tuple[str, float, bool]:
        """Pick the winner of a calibrated distribution without depending on candidate order.

        `max(probs, key=probs.get)` keeps the first of several equal maxima, i.e. whichever option
        the caller happened to list first. Ties are broken on the sorted label instead. A fully
        uniform distribution carries no signal at all (typically every candidate hit a fallback
        score), so it is reported as degenerate and resolved to UNKNOWN rather than to an arbitrary
        first option.

        Returns (best_choice, confidence, degenerate).
        """
        if not probs:
            return "UNKNOWN", 0.0, True
        top = max(probs.values())
        if top - min(probs.values()) <= 1e-12:
            logger.warning(
                "All %d candidates received the same score, so the decision has no signal; "
                "returning UNKNOWN. Check that the server returns logprobs for the answer tokens.",
                len(probs),
            )
            return "UNKNOWN", top, True
        best = max(sorted(probs), key=probs.get)
        return best, probs[best], False

    def decide_choice(
        self,
        state: Dict[str, Any],
        candidates: Union[Type[Enum], List[str]],
        criteria: Union[str, Dict[str, str]] = "",
        allow_abstain: bool = True,
        order_invariant: bool = False,
    ) -> ChoiceDecision:
        """Evaluates a categorical choice decision across the given candidates with calibrated probabilities."""
        if self.mock and self._mock_client is not None:
            return self._mock_client.decide_choice(
                state=state,
                candidates=candidates,
                criteria=criteria,
                allow_abstain=allow_abstain,
                order_invariant=order_invariant,
            )
        if isinstance(candidates, type) and issubclass(candidates, Enum):
            options = [e.value for e in candidates]
        else:
            options = list(candidates)

        if allow_abstain and "UNKNOWN" not in options:
            options.append("UNKNOWN")

        if order_invariant:
            return self._decide_choice_order_invariant(state, options, criteria, allow_abstain)

        if self.backend == "ollama":
            return self._decide_choice_ollama(state, options, criteria, allow_abstain)
        elif self.backend == "laya":
            return self._decide_choice_laya(state, options, criteria, allow_abstain)
        else:
            return self._decide_choice_openai(state, options, criteria, allow_abstain)

    def _score_single_candidate_ollama(
        self,
        state: Dict[str, Any],
        candidate: str,
        criteria: Union[str, Dict[str, str]]
    ) -> float:
        """Scores a single candidate option independently using Ollama JSON response."""
        if isinstance(criteria, dict):
            cand_crit = criteria.get(candidate)
            if cand_crit:
                crit_text = f"Criteria for '{candidate}': {cand_crit}"
            else:
                crit_text = "\n".join([f"- {k}: {v}" for k, v in sorted(criteria.items())])
        else:
            crit_text = str(criteria)

        prompt = (
            f"You are a probabilistic decision engine evaluating candidate suitability.\n\n"
            f"State / Context:\n{json.dumps(state, ensure_ascii=False, indent=2)}\n\n"
            f"Evaluation Criteria:\n{crit_text}\n\n"
            f"Candidate Option: '{candidate}'\n\n"
            f"Rate the relative likelihood score (0.0 to 10.0) that this candidate option is the correct decision.\n"
            f"Output ONLY a valid JSON object matching this schema:\n"
            f'{{"score": 0.0}}'
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

        content = data.get("message", {}).get("content", "").strip()
        if "```" in content:
            parts = content.split("```")
            for p in parts:
                p_clean = p.strip()
                if p_clean.startswith("json"):
                    p_clean = p_clean[4:].strip()
                if p_clean.startswith("{") and p_clean.endswith("}"):
                    content = p_clean
                    break
        try:
            parsed = json.loads(content)
            return float(parsed.get("score", 0.0))
        except Exception:
            return 0.0

    def _score_single_candidate_openai(
        self,
        state: Dict[str, Any],
        candidate: str,
        criteria: Union[str, Dict[str, str]]
    ) -> float:
        """Scores a single candidate option independently using isolated binary logprobs."""
        if isinstance(criteria, dict):
            cand_crit = criteria.get(candidate)
            if cand_crit:
                crit_text = f"Criteria for '{candidate}': {cand_crit}"
            else:
                crit_text = "\n".join([f"- {k}: {v}" for k, v in sorted(criteria.items())])
        else:
            crit_text = str(criteria)

        prompt = (
            f"Given the following state:\n{json.dumps(state, ensure_ascii=False, indent=2)}\n\n"
            f"Evaluation criteria:\n{crit_text}\n\n"
            f"Evaluate candidate option: '{candidate}'\n"
            f"Is this candidate the single best and correct category for this state?\n"
            f"A. YES\n"
            f"B. NO\n\n"
            f"Reply with ONLY the option letter (A or B):"
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data: Optional[Dict[str, Any]] = None
        if self.use_chat:
            system_msg = "You are a precise binary classification evaluator. Reply with ONLY the option letter (A or B)."
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
                # Top level, not under "extra_body": extra_body is an openai-python SDK convention
                # that the SDK merges into the request body. Sent with requests it arrives as a
                # literal key that llama.cpp and vLLM ignore, so enable_thinking=False never applied.
                chat_payload["chat_template_kwargs"] = self.chat_template_kwargs

            endpoint_used = f"{self.base_url}/chat/completions"
            resp = requests.post(endpoint_used, headers=headers, json=chat_payload, timeout=30)
            if resp.status_code in (404, 405):
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

        prob_a = choice_logprobs.get("A", choice_logprobs.get(" A"))
        prob_b = choice_logprobs.get("B", choice_logprobs.get(" B"))

        if prob_a is not None and prob_b is not None:
            return float(prob_a) - float(prob_b)
        elif prob_a is not None:
            return float(prob_a)
        elif prob_b is not None:
            return -float(prob_b) - 5.0
        else:
            logger.warning(
                "Neither 'A' nor 'B' in top logprobs for candidate '%s' (top tokens: %s); "
                "falling back to the generated text.",
                candidate,
                list(choice_logprobs.keys())[:10],
            )
            # Fallback when logprobs are absent
            content_str = ""
            if "message" in choice_data and "content" in choice_data["message"]:
                content_str = (choice_data["message"]["content"] or "").strip()
            elif "text" in choice_data:
                content_str = (choice_data["text"] or "").strip()
            
            if content_str.upper().startswith("A") or "YES" in content_str.upper():
                return 2.0
            elif content_str.upper().startswith("B") or "NO" in content_str.upper():
                return -2.0
            return 0.0

    def _score_single_candidate(
        self,
        state: Dict[str, Any],
        candidate: str,
        criteria: Union[str, Dict[str, str]]
    ) -> float:
        """Evaluates a single candidate likelihood in an isolated prompt without competitor options."""
        if self.backend == "ollama":
            return self._score_single_candidate_ollama(state, candidate, criteria)
        elif self.backend == "laya":
            crit_dict = dict(criteria) if isinstance(criteria, dict) else {"criteria": str(criteria)}
            resp = requests.post(
                f"{self.base_url}/decision/choice",
                json={
                    "model": self.model,
                    "state": state,
                    "candidates": [candidate, "UNKNOWN"],
                    "criteria": crit_dict,
                },
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            raw_logits = data.get("raw_logits") or {}
            return float(raw_logits.get(candidate, 0.0))
        else:
            return self._score_single_candidate_openai(state, candidate, criteria)

    def _decide_choice_order_invariant(
        self,
        state: Dict[str, Any],
        options: List[str],
        criteria: Union[str, Dict[str, str]],
        allow_abstain: bool
    ) -> ChoiceDecision:
        """Evaluates categorical choice with mathematical order invariance by scoring each candidate in isolation."""
        if len(options) > 20:
            logger.warning(
                "Candidate option count %d exceeds recommended limit (20) for order_invariant evaluation.",
                len(options)
            )

        # See order_invariant_max_workers in __init__: >1 is faster but not deterministic on
        # batching servers.
        max_workers = min(len(options), self.order_invariant_max_workers)
        extracted_logits: Dict[str, float] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_opt = {
                executor.submit(self._score_single_candidate, state, opt, criteria): opt
                for opt in options
            }
            for future in future_to_opt:
                opt = future_to_opt[future]
                try:
                    score = future.result()
                    extracted_logits[opt] = float(score)
                except Exception as e:
                    logger.warning("Scoring failed for candidate '%s': %s", opt, e)
                    extracted_logits[opt] = -100.0

        calibrated_probs = self.calibrator.calibrate(extracted_logits)
        best_choice, confidence, degenerate = self._select_best(calibrated_probs)

        effective_thresh = self._get_effective_threshold(len(options))
        abstained = degenerate
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

    def _decide_choice_laya(
        self,
        state: Dict[str, Any],
        options: List[str],
        criteria: Union[str, Dict[str, str]],
        allow_abstain: bool
    ) -> ChoiceDecision:
        """Evaluates categorical choice via Laya (ModernBERT System 1) endpoint with temperature calibration."""
        crit_dict = dict(criteria) if isinstance(criteria, dict) else {"criteria": str(criteria)}
        if allow_abstain and "UNKNOWN" not in crit_dict:
            crit_dict["UNKNOWN"] = "None of the other categories apply, or query is out-of-scope/unrelated."

        payload = {
            "model": self.model,
            "state": state,
            "candidates": options,
            "criteria": crit_dict,
        }

        resp = requests.post(f"{self.base_url}/decision/choice", json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        raw_probs = data.get("probabilities") or {opt: 1.0 / len(options) for opt in options}
        extracted_logits = data.get("raw_logits")
        if not extracted_logits:
            extracted_logits = {k: math.log(max(float(v), 1e-6)) for k, v in raw_probs.items()}

        calibrated_probs = self.calibrator.calibrate(extracted_logits)
        best_choice, confidence, degenerate = self._select_best(calibrated_probs)

        effective_thresh = self._get_effective_threshold(len(options))
        abstained = degenerate
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
        best_choice, confidence, degenerate = self._select_best(calibrated_probs)

        effective_thresh = self._get_effective_threshold(len(options))
        abstained = degenerate
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
                # Top level, not under "extra_body": extra_body is an openai-python SDK convention
                # that the SDK merges into the request body. Sent with requests it arrives as a
                # literal key that llama.cpp and vLLM ignore, so enable_thinking=False never applied.
                chat_payload["chat_template_kwargs"] = self.chat_template_kwargs

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
        best_choice, confidence, degenerate = self._select_best(calibrated_probs)

        effective_thresh = self._get_effective_threshold(len(options))
        abstained = degenerate
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

    def decide_score(
        self,
        state: Dict[str, Any],
        criteria: str = "",
        levels: Optional[List[str]] = None,
    ) -> ScoreDecision:
        """Evaluates an ordinal severity/quality score."""
        if self.mock and self._mock_client is not None:
            return self._mock_client.decide_score(state=state, criteria=criteria, levels=levels)

        if levels is None:
            levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

        decision = self.decide_choice(
            state=state,
            candidates=levels,
            criteria=criteria,
            allow_abstain=False,
            order_invariant=True,
        )

        n = len(levels)
        expected_score = sum(
            (idx / max(n - 1, 1)) * decision.probabilities.get(lvl, 0.0)
            for idx, lvl in enumerate(levels)
        )

        return ScoreDecision(
            expected_score=round(expected_score, 4),
            level_probabilities=decision.probabilities,
            confidence=decision.confidence,
            abstained=decision.abstained,
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

    def create_decision_flywheel(
        self,
        reasoning_engine: Optional[Any] = None,
        reasoning_model: str = "Qwen/Qwen3-14B-Thinking",
        crystallization_store: Optional[Any] = None,
        auto_crystallize: bool = True,
        escalate_threshold: Optional[float] = None,
    ) -> Any:
        """Creates a DeliberativeDecisionFlywheel coupling this client with System 2 exploration."""
        from openjevpro.flywheel import DeliberativeDecisionFlywheel
        return DeliberativeDecisionFlywheel(
            fast_engine=self,
            reasoning_engine=reasoning_engine,
            reasoning_model=reasoning_model,
            crystallization_store=crystallization_store,
            auto_crystallize=auto_crystallize,
            escalate_threshold=escalate_threshold,
        )

    def decide_with_crystallization(
        self,
        state: Dict[str, Any],
        candidates: Union[Type[Enum], List[str]],
        criteria: Union[str, Dict[str, str]] = "",
        allow_abstain: bool = True,
        order_invariant: bool = False,
        reasoning_engine: Optional[Any] = None,
        reasoning_model: str = "Qwen/Qwen3-14B-Thinking",
        store: Optional[Any] = None,
    ) -> ChoiceDecision:
        """Evaluates choice via cognitive flywheel: fast-path with System 2 exploration and crystallization."""
        flywheel = self.create_decision_flywheel(
            reasoning_engine=reasoning_engine,
            reasoning_model=reasoning_model,
            crystallization_store=store,
            auto_crystallize=True,
        )
        return flywheel.evaluate_choice(
            state=state,
            candidates=candidates,
            criteria=criteria,
            allow_abstain=allow_abstain,
            order_invariant=order_invariant,
        )

# Backward compatibility alias
OpenJevClient = OpenJevProClient

