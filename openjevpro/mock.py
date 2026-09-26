"""Zero-GPU Mock Decision Engine and MockClient for local testing and zero-config evaluation."""

import math
import hashlib
import time
from typing import Dict, Any, Type, Union, List, Optional
from enum import Enum

from openjevpro.schemas import ChoiceDecision, NoulDecision, ScoreDecision
from openjevpro.calibrator import TemperatureCalibrator

class MockClient:
    """Zero-dependency, offline deterministic mock client providing instant simulation of OpenJevPro primitives."""

    def __init__(
        self,
        base_url: str = "mock://localhost",
        model: str = "mock-simulator-v1",
        temperature_scaling: float = 1.0,
        abstain_threshold: Union[float, str] = 0.40,
        **kwargs: Any,
    ):
        self.base_url = base_url
        self.model = model
        self.calibrator = TemperatureCalibrator(temperature=temperature_scaling)
        self.abstain_threshold = abstain_threshold

    def _get_effective_threshold(self, n_options: int) -> float:
        if isinstance(self.abstain_threshold, str) and self.abstain_threshold.lower() == "auto":
            return 1.25 / max(n_options, 1)
        return float(self.abstain_threshold)

    def _extract_state_text(self, state: Dict[str, Any]) -> str:
        parts: List[str] = []
        for k, v in state.items():
            if isinstance(v, (str, int, float, bool)):
                parts.append(str(v).lower())
            elif isinstance(v, (list, tuple)):
                parts.append(" ".join(str(x).lower() for x in v))
            elif isinstance(v, dict):
                parts.append(self._extract_state_text(v))
        return " ".join(parts)

    _SEMANTIC_ASSOCIATIONS = {
        "billing": {"bill", "invoice", "refund", "charge", "payment", "fee", "cost", "card", "subscription", "price"},
        "tech_support": {"bug", "issue", "error", "crash", "help", "broken", "fail", "slow", "debug", "install", "technical", "timeout", "leak", "memory", "exception", "database", "connection", "server"},
        "security_fraud": {"security", "fraud", "unauthorized", "login", "hack", "breach", "threat", "suspicious", "stolen", "ip", "attack"},
        "human_review": {"review", "agent", "human", "manual", "escalate", "dispute", "appeal", "complex", "supervisor"},
    }

    def _score_single_candidate(self, state_text: str, candidate: str, criteria: str) -> float:
        """Isolated deterministic scoring of a candidate against state context.
        Independent of other candidate presence or ordering."""
        cand_lower = candidate.lower()
        if cand_lower == "unknown":
            return 0.5

        tokens = cand_lower.replace("_", " ").replace("-", " ").split()
        score = 0.0

        # Direct token match
        for token in tokens:
            if len(token) > 2 and token in state_text:
                score += 3.5

        # Exact substring match bonus
        if cand_lower in state_text or cand_lower.replace("_", " ") in state_text:
            score += 3.0

        # Semantic domain association match
        for category, keywords in self._SEMANTIC_ASSOCIATIONS.items():
            if any(t in category for t in tokens):
                matched_kw = sum(1 for kw in keywords if kw in state_text)
                score += matched_kw * 1.8

        # Criteria alignment bonus
        if criteria and any(token in criteria.lower() for token in tokens):
            score += 1.5

        # Deterministic pseudo-random semantic hash perturbation in [-0.2, 0.2]
        hash_seed = f"{state_text}::{cand_lower}"
        h_val = int(hashlib.sha256(hash_seed.encode("utf-8")).hexdigest()[:8], 16)
        noise = ((h_val / 0xFFFFFFFF) - 0.5) * 0.4
        score += noise

        return score

    def decide_choice(
        self,
        state: Dict[str, Any],
        candidates: Union[Type[Enum], List[str]],
        criteria: Union[str, Dict[str, str]] = "",
        allow_abstain: bool = True,
        order_invariant: bool = True,
    ) -> ChoiceDecision:
        """Evaluates categorical choice with strict isolated candidate scoring and commutative softmax."""
        if isinstance(candidates, type) and issubclass(candidates, Enum):
            options = [e.value for e in candidates]
        else:
            options = list(candidates)

        if allow_abstain and "UNKNOWN" not in options:
            options.append("UNKNOWN")

        state_text = self._extract_state_text(state)
        crit_str = json_criteria = ""
        if isinstance(criteria, dict):
            crit_str = " ".join(f"{k}: {v}" for k, v in criteria.items())
        else:
            crit_str = str(criteria)

        # Isolated scoring: each option is evaluated strictly against (state, option)
        raw_logits: Dict[str, float] = {}
        for opt in options:
            raw_logits[opt] = self._score_single_candidate(state_text, opt, crit_str)

        # Commutative Softmax normalization
        calibrated_probs = self.calibrator.calibrate(raw_logits)
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
            raw_logits=raw_logits,
        )

    def decide_noul(
        self,
        state: Dict[str, Any],
        assertion: str,
    ) -> NoulDecision:
        """Evaluates a binary truth assertion judgment offline."""
        options = ["TRUE", "FALSE"]
        decision = self.decide_choice(
            state=state,
            candidates=options,
            criteria=f"Evaluate whether the following assertion is strictly TRUE or FALSE: {assertion}",
            allow_abstain=False,
            order_invariant=True,
        )
        p_true = decision.probabilities.get("TRUE", 0.5)
        value = p_true >= 0.5
        conf = p_true if value else (1.0 - p_true)
        return NoulDecision(
            value=value,
            probability_true=p_true,
            confidence=conf,
            abstained=decision.abstained,
        )

    def decide_score(
        self,
        state: Dict[str, Any],
        criteria: str = "",
        levels: Optional[List[str]] = None,
    ) -> ScoreDecision:
        """Evaluates an ordinal severity/quality score."""
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
