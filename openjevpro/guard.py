"""
TypeSafe Jev Runtime Calibration & Adaptive Safety Guard Harness.

Provides adaptive temperature calibration and dual-threshold abstention
to safeguard predictions from TypeSafe Jev and neural decision engines.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from openjevpro.calibrator import TemperatureCalibrator
from openjevpro.harness import BaseDecisionEngine


@dataclass
class GuardDecision:
    """Structured decision output produced by TypeSafeJevGuardHarness."""
    choice: str
    confidence: float
    is_abstained: bool
    tentative_choice: str
    threshold_used: float
    calibrated_probs: Dict[str, float]
    raw_probs: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class TypeSafeJevGuardHarness(BaseDecisionEngine):
    """Adaptive safety harness wrapping raw prediction probabilities or decision engines.

    Applies:
    1. Pseudo-logit inversion: z_i = ln(max(p_i, epsilon))
    2. Temperature calibration scaling via TemperatureCalibrator
    3. Adaptive dual-threshold abstention: tau = max(min_confidence, alpha / K)
    """

    def __init__(
        self,
        engine: Optional[Any] = None,
        calibrator: Optional[TemperatureCalibrator] = None,
        alpha: float = 1.25,
        min_confidence: float = 0.72,
        epsilon: float = 1e-6,
        name: Optional[str] = None,
    ):
        self.engine = engine
        self.calibrator = calibrator if calibrator is not None else TemperatureCalibrator(temperature=1.25)
        self.alpha = float(alpha)
        self.min_confidence = float(min_confidence)
        self.epsilon = float(epsilon)
        self.name = name or (f"Guarded({getattr(engine, 'name', 'TypeSafeJev')})" if engine else "TypeSafe Jev Guard Harness")

    def compute_threshold(
        self,
        num_classes: int,
        alpha: Optional[float] = None,
        min_confidence: Optional[float] = None,
    ) -> float:
        """Computes dynamic adaptive threshold: tau = max(min_confidence, alpha / K).

        If num_classes <= 0, returns min_confidence.
        """
        eff_alpha = self.alpha if alpha is None else float(alpha)
        eff_min_conf = self.min_confidence if min_confidence is None else float(min_confidence)

        if num_classes <= 0:
            return eff_min_conf
        return max(eff_min_conf, eff_alpha / float(num_classes))

    def evaluate_probabilities(
        self,
        raw_probs: Dict[str, float],
        candidates: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        alpha: Optional[float] = None,
    ) -> GuardDecision:
        """Applies calibration and adaptive safety guard on a given probability distribution."""
        eff_min_conf = self.min_confidence if min_confidence is None else float(min_confidence)
        eff_alpha = self.alpha if alpha is None else float(alpha)

        # Handle empty probability inputs
        if not raw_probs and not candidates:
            return GuardDecision(
                choice="UNKNOWN",
                confidence=0.0,
                is_abstained=True,
                tentative_choice="UNKNOWN",
                threshold_used=eff_min_conf,
                calibrated_probs={},
                raw_probs={},
                metadata={"reason": "empty_input"},
            )

        # Normalize probability map
        prob_map: Dict[str, float] = {}
        if candidates:
            for c in candidates:
                prob_map[c] = float(raw_probs.get(c, 0.0))
        else:
            for k, v in raw_probs.items():
                prob_map[k] = float(v)

        total_prob = sum(prob_map.values())
        if total_prob > 0:
            prob_map = {k: v / total_prob for k, v in prob_map.items()}
        else:
            uniform = 1.0 / len(prob_map) if prob_map else 0.0
            prob_map = {k: uniform for k in prob_map}

        # Step 1: Invert to pseudo-logits
        pseudo_logits: Dict[str, float] = {
            k: math.log(max(v, self.epsilon)) for k, v in prob_map.items()
        }

        # Step 2: Temperature calibration
        if self.calibrator is not None:
            calibrated_probs = self.calibrator.calibrate(pseudo_logits)
        else:
            calibrated_probs = dict(prob_map)

        # Step 3: Determine top candidate
        num_classes = len(calibrated_probs)
        tau = self.compute_threshold(num_classes=num_classes, alpha=eff_alpha, min_confidence=eff_min_conf)

        if not calibrated_probs:
            return GuardDecision(
                choice="UNKNOWN",
                confidence=0.0,
                is_abstained=True,
                tentative_choice="UNKNOWN",
                threshold_used=tau,
                calibrated_probs={},
                raw_probs=prob_map,
                metadata={"reason": "empty_distribution"},
            )

        best_cand = max(calibrated_probs.keys(), key=lambda k: calibrated_probs[k])
        best_prob = calibrated_probs[best_cand]

        # Step 4: Dual-threshold abstention evaluation
        if best_cand == "UNKNOWN":
            # Direct out-of-scope prediction from model
            return GuardDecision(
                choice="UNKNOWN",
                confidence=best_prob,
                is_abstained=True,
                tentative_choice="UNKNOWN",
                threshold_used=tau,
                calibrated_probs=calibrated_probs,
                raw_probs=prob_map,
                metadata={"reason": "explicit_unknown"},
            )

        if best_prob < tau:
            # Low confidence or high entropy -> abstain
            return GuardDecision(
                choice="UNKNOWN",
                confidence=best_prob,
                is_abstained=True,
                tentative_choice=best_cand,
                threshold_used=tau,
                calibrated_probs=calibrated_probs,
                raw_probs=prob_map,
                metadata={"reason": "below_threshold", "gap": tau - best_prob},
            )

        # High confidence pass-through
        return GuardDecision(
            choice=best_cand,
            confidence=best_prob,
            is_abstained=False,
            tentative_choice=best_cand,
            threshold_used=tau,
            calibrated_probs=calibrated_probs,
            raw_probs=prob_map,
            metadata={"reason": "passed_threshold"},
        )

    def guard(
        self,
        result: Any,
        candidates: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        alpha: Optional[float] = None,
    ) -> GuardDecision:
        """Guards a decision result (dict or probability mapping)."""
        if isinstance(result, GuardDecision):
            return result

        if isinstance(result, dict):
            if "probabilities" in result and isinstance(result["probabilities"], dict):
                return self.evaluate_probabilities(
                    raw_probs=result["probabilities"],
                    candidates=candidates,
                    min_confidence=min_confidence,
                    alpha=alpha,
                )
            return self.evaluate_probabilities(
                raw_probs=result,
                candidates=candidates,
                min_confidence=min_confidence,
                alpha=alpha,
            )

        raise TypeError(f"Cannot guard object of type {type(result)}; expected dict or GuardDecision")

    def evaluate_choice(
        self,
        state: Dict[str, Any],
        candidates: List[str],
        criteria: Dict[str, str],
        allow_abstain: bool = True,
        min_confidence: Optional[float] = None,
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Evaluates choice using wrapped engine, then applies calibration and safety guarding."""
        t0 = time.time()
        if self.engine is None:
            raise RuntimeError("No underlying engine configured in TypeSafeJevGuardHarness to evaluate_choice.")

        raw_res = self.engine.evaluate_choice(
            state=state,
            candidates=candidates,
            criteria=criteria,
            allow_abstain=allow_abstain,
        )
        latency_ms = (time.time() - t0) * 1000

        raw_probs = raw_res.get("probabilities", {})
        guard_res = self.evaluate_probabilities(
            raw_probs=raw_probs,
            candidates=candidates,
            min_confidence=min_confidence,
            alpha=alpha,
        )

        return {
            "choice": guard_res.choice,
            "confidence": guard_res.confidence,
            "probabilities": guard_res.calibrated_probs,
            "abstained": guard_res.is_abstained,
            "tentative_choice": guard_res.tentative_choice,
            "raw_probabilities": guard_res.raw_probs,
            "threshold_used": guard_res.threshold_used,
            "latency_ms": raw_res.get("latency_ms", latency_ms),
            "guard_decision": guard_res,
        }
