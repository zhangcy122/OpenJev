"""Deliberative Decision Flywheel & Crystallization Operator for OpenJevPro.

Implements the "Explore First, Crystallize Later" cognitive flywheel:
1. System 1 (Fast-Path / Laya / Jev): Sub-35ms calibrated probabilistic classification.
2. System 2 (Deliberation / Thinking LLM): Conditional counterfactual exploration for ambiguous or abstained queries.
3. Crystallization Operator: Compiles exploratory reasoning traces into refined criteria and precedent memory,
   permanently promoting subsequent similar queries to the sub-35ms fast path.
"""

import time
import json
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, asdict

from openjevpro.schemas import ChoiceDecision

logger = logging.getLogger("openjevpro.flywheel")


@dataclass
class CrystallizationReceipt:
    """Receipt documenting System 2 exploration and System 1 crystallization."""
    escalated: bool
    target_choice: Optional[str] = None
    reasoning_trace: Optional[str] = None
    discriminative_rule: Optional[str] = None
    crystallized: bool = False
    fast_path_latency_ms: float = 0.0
    system_2_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CrystallizationStore:
    """In-memory and persistent store for refined criteria and decision precedents."""

    def __init__(self, max_rule_chars_per_candidate: int = 400):
        self.max_rule_chars = max_rule_chars_per_candidate
        # candidate -> list of refined discriminative rules
        self._criteria_augments: Dict[str, List[str]] = {}
        # precedent key -> precedent record
        self._precedents: Dict[str, Dict[str, Any]] = {}

    def get_augmented_criteria(self, base_criteria: Union[str, Dict[str, str]], candidate: str) -> str:
        """Returns the base criterion augmented with crystallized discriminative rules."""
        base_text = ""
        if isinstance(base_criteria, dict):
            base_text = base_criteria.get(candidate, "")
        elif isinstance(base_criteria, str):
            base_text = base_criteria

        rules = self._criteria_augments.get(candidate, [])
        if not rules:
            return base_text

        rule_suffix = " Distilled boundaries: " + "; ".join(rules)
        combined = f"{base_text}. {rule_suffix}".strip(". ") + "."
        if len(combined) > self.max_rule_chars:
            return combined[:self.max_rule_chars] + "..."
        return combined

    def merge_all_criteria(
        self,
        base_criteria: Union[str, Dict[str, str]],
        candidates: List[str]
    ) -> Dict[str, str]:
        """Merges all stored criteria augmentations into a dictionary for all candidates."""
        out: Dict[str, str] = {}
        for c in candidates:
            out[c] = self.get_augmented_criteria(base_criteria, c)
        return out

    def add_rule(self, candidate: str, rule: str) -> bool:
        """Adds a refined discriminative rule for the candidate."""
        clean_rule = rule.strip().strip(".")
        if not clean_rule:
            return False
        rules = self._criteria_augments.setdefault(candidate, [])
        if clean_rule not in rules:
            rules.append(clean_rule)
            return True
        return False

    def record_precedent(self, key: str, choice: str, reasoning: str, rule: str):
        """Records a decision precedent for auditing and retrieval."""
        self._precedents[key] = {
            "choice": choice,
            "reasoning": reasoning,
            "rule": rule,
            "timestamp": time.time(),
        }

    def has_precedent(self, key: str) -> bool:
        return key in self._precedents

    def clear(self):
        self._criteria_augments.clear()
        self._precedents.clear()


class CrystallizationOperator:
    """Distills System 2 reasoning traces into System 1 criteria and precedent memory."""

    def __init__(self, store: Optional[CrystallizationStore] = None):
        self.store = store or CrystallizationStore()

    def distill(
        self,
        state: Dict[str, Any],
        candidates: List[str],
        exploration: Dict[str, Any]
    ) -> bool:
        """Extracts discriminative rules and updates criteria store."""
        choice = exploration.get("winning_candidate") or exploration.get("choice")
        rule = exploration.get("discriminative_rule") or ""
        reasoning = exploration.get("justification") or exploration.get("reasoning_trace") or ""

        if not choice or choice not in candidates:
            return False

        updated = False
        if rule:
            updated = self.store.add_rule(choice, rule)

        # Record precedent indexed by state hash or text snippet
        key = json.dumps(state, sort_keys=True)
        self.store.record_precedent(key, choice, reasoning, rule)
        return updated or bool(rule)


class DeliberativeDecisionFlywheel:
    """Cognitive flywheel orchestrator coupling System 1 fast path with System 2 deliberation."""

    def __init__(
        self,
        fast_engine: Any,
        reasoning_engine: Optional[Union[Callable, Any]] = None,
        reasoning_model: str = "Qwen/Qwen3-14B-Thinking",
        crystallization_store: Optional[CrystallizationStore] = None,
        auto_crystallize: bool = True,
        escalate_threshold: Optional[float] = None,
    ):
        self.fast_engine = fast_engine
        self.reasoning_engine = reasoning_engine
        self.reasoning_model = reasoning_model
        self.store = crystallization_store or CrystallizationStore()
        self.operator = CrystallizationOperator(store=self.store)
        self.auto_crystallize = auto_crystallize
        self.escalate_threshold = escalate_threshold

    def _explore_system_2(
        self,
        state: Dict[str, Any],
        candidates: List[str],
        criteria: Union[str, Dict[str, str]],
        tentative_choice: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executes System 2 deliberative exploration on ambiguous or abstained queries."""
        # 1. Custom callable reasoning engine
        if callable(self.reasoning_engine):
            return self.reasoning_engine(state, candidates, criteria, tentative_choice)

        # 2. Reasoning engine object with evaluate_choice or explore
        if self.reasoning_engine and hasattr(self.reasoning_engine, "explore"):
            return self.reasoning_engine.explore(state, candidates, criteria)

        # 3. Default structured deliberation prompt using client backend
        prompt = (
            f"You are a master decision classifier performing counterfactual deliberation.\n\n"
            f"Context / State:\n{json.dumps(state, ensure_ascii=False, indent=2)}\n\n"
            f"Candidate Categories:\n" + "\n".join([f"- {c}" for c in candidates if c != "UNKNOWN"]) + "\n\n"
            f"Evaluation Criteria:\n{json.dumps(criteria, ensure_ascii=False) if isinstance(criteria, dict) else str(criteria)}\n\n"
            f"The fast decision engine abstained due to ambiguity (tentative: '{tentative_choice}').\n"
            f"Deliberate carefully across all alternatives. Return ONLY a valid JSON object matching:\n"
            f'{{"winning_candidate": "<best candidate>", "justification": "<concise causal argument>", "discriminative_rule": "<one-sentence rule distinguishing this category from near-misses>"}}'
        )

        # If fast_engine has request dispatch capability (e.g. OpenJevProClient)
        if hasattr(self.fast_engine, "base_url") and hasattr(self.fast_engine, "backend"):
            import requests
            headers = {"Authorization": f"Bearer {getattr(self.fast_engine, 'api_key', 'EMPTY')}"}
            payload = {
                "model": self.reasoning_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            }
            endpoint = f"{self.fast_engine.base_url}/chat/completions"
            try:
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                if "```" in content:
                    parts = content.split("```")
                    for p in parts:
                        p_c = p.strip()
                        if p_c.startswith("json"):
                            p_c = p_c[4:].strip()
                        if p_c.startswith("{") and p_c.endswith("}"):
                            content = p_c
                            break
                return json.loads(content)
            except Exception as e:
                logger.warning("System 2 API call failed, falling back to tentative choice: %s", e)

        # Fallback heuristic deliberation
        fallback_win = tentative_choice or (candidates[0] if candidates else "UNKNOWN")
        return {
            "winning_candidate": fallback_win,
            "justification": f"Deliberative fallback resolved to {fallback_win}",
            "discriminative_rule": f"Distinguish {fallback_win} by state context keywords."
        }

    def evaluate_choice(
        self,
        state: Dict[str, Any],
        candidates: Union[Any, List[str]],
        criteria: Union[str, Dict[str, str]] = "",
        allow_abstain: bool = True,
        order_invariant: bool = False,
    ) -> ChoiceDecision:
        """Evaluates choice via System 1 fast path, escalating to System 2 upon ambiguity and crystallizing."""
        # Unpack candidates enum if needed
        from enum import Enum
        if isinstance(candidates, type) and issubclass(candidates, Enum):
            cand_list = [e.value for e in candidates]
        else:
            cand_list = list(candidates)

        # Step 1: Merge any crystallized criteria augmentations
        merged_criteria = self.store.merge_all_criteria(criteria, cand_list)

        # Step 2: System 1 Fast-Path Evaluation
        t0 = time.time()
        decision: ChoiceDecision = self.fast_engine.decide_choice(
            state=state,
            candidates=cand_list,
            criteria=merged_criteria,
            allow_abstain=allow_abstain,
            order_invariant=order_invariant,
        )
        fast_duration_ms = (time.time() - t0) * 1000.0

        # Step 3: Check Escalation Trigger
        effective_cutoff = self.escalate_threshold
        if effective_cutoff is None and hasattr(self.fast_engine, "_get_effective_threshold"):
            effective_cutoff = self.fast_engine._get_effective_threshold(len(cand_list))
        elif effective_cutoff is None:
            effective_cutoff = 0.45

        needs_escalation = decision.abstained or (decision.confidence < effective_cutoff)

        if not needs_escalation:
            # Fast-path accepted! Sub-35ms return
            decision.escalated = False
            decision.crystallization_receipt = CrystallizationReceipt(
                escalated=False,
                target_choice=decision.value,
                fast_path_latency_ms=fast_duration_ms,
                total_latency_ms=fast_duration_ms
            ).to_dict()
            return decision

        # Step 4: System 2 Deliberative Exploration
        logger.info("Decision confidence marginal (%.2f) or abstained. Escalating to System 2...", decision.confidence)
        t_sys2_0 = time.time()
        exploration = self._explore_system_2(
            state=state,
            candidates=cand_list,
            criteria=merged_criteria,
            tentative_choice=decision.tentative_value or decision.value
        )
        sys2_duration_ms = (time.time() - t_sys2_0) * 1000.0

        target_win = exploration.get("winning_candidate") or decision.tentative_value or "UNKNOWN"
        justification = exploration.get("justification") or ""
        rule = exploration.get("discriminative_rule") or ""

        # Step 5: Crystallization Operator
        crystallized = False
        if self.auto_crystallize and target_win in cand_list and target_win != "UNKNOWN":
            crystallized = self.operator.distill(state, cand_list, exploration)

            # Re-evaluate with crystallized criteria to confirm fast path activation
            refreshed_criteria = self.store.merge_all_criteria(criteria, cand_list)
            try:
                refreshed_decision = self.fast_engine.decide_choice(
                    state=state,
                    candidates=cand_list,
                    criteria=refreshed_criteria,
                    allow_abstain=allow_abstain,
                    order_invariant=order_invariant,
                )
                if not refreshed_decision.abstained:
                    refreshed_decision.escalated = True
                    refreshed_decision.crystallization_receipt = CrystallizationReceipt(
                        escalated=True,
                        target_choice=refreshed_decision.value,
                        reasoning_trace=justification,
                        discriminative_rule=rule,
                        crystallized=crystallized,
                        fast_path_latency_ms=fast_duration_ms,
                        system_2_latency_ms=sys2_duration_ms,
                        total_latency_ms=fast_duration_ms + sys2_duration_ms
                    ).to_dict()
                    return refreshed_decision
            except Exception as e:
                logger.debug("Fast-path re-evaluation failed: %s", e)

        # If re-evaluation did not clear abstention, return explored decision directly
        calibrated_probs = dict(decision.probabilities)
        if target_win in calibrated_probs:
            calibrated_probs[target_win] = max(calibrated_probs.get(target_win, 0.0), 0.90)
            # Re-normalize
            total_p = sum(calibrated_probs.values())
            calibrated_probs = {k: v / total_p for k, v in calibrated_probs.items()}

        receipt = CrystallizationReceipt(
            escalated=True,
            target_choice=target_win,
            reasoning_trace=justification,
            discriminative_rule=rule,
            crystallized=crystallized,
            fast_path_latency_ms=fast_duration_ms,
            system_2_latency_ms=sys2_duration_ms,
            total_latency_ms=fast_duration_ms + sys2_duration_ms
        )

        return ChoiceDecision(
            value=target_win,
            probabilities=calibrated_probs,
            confidence=calibrated_probs.get(target_win, 0.90),
            abstained=False,
            tentative_value=None,
            raw_logits=decision.raw_logits,
            escalated=True,
            crystallization_receipt=receipt.to_dict()
        )
