"""
OpenJevPro Benchmark Harness
Standardized evaluation harness comparing:
1. TypeSafe Jev (Proprietary System 1 decision engine)
2. OpenJevPro (Open-source calibrated typed probabilistic decision engine)
3. Direct Structured Output (Raw Open-Source LLM + JSON Schema constraints)
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

from openjevpro.client import OpenJevProClient
from openjevpro.schemas import ChoiceDecision

class BaseDecisionEngine:
    """Abstract base class for decision engines in the harness."""
    name: str

    def evaluate_choice(
        self,
        state: Dict[str, Any],
        candidates: List[str],
        criteria: Dict[str, str],
        allow_abstain: bool = True,
    ) -> Dict[str, Any]:
        raise NotImplementedError

class TypeSafeJevEngine(BaseDecisionEngine):
    """Engine interacting with TypeSafe AI Jev official API."""
    def __init__(self, api_key: str, endpoint: str = "https://api.typesafe.ai/v1/systemone", model: str = "jev-latest"):
        self.name = "TypeSafe Jev (Commercial)"
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model

    def evaluate_choice(
        self,
        state: Dict[str, Any],
        candidates: List[str],
        criteria: Dict[str, str],
        allow_abstain: bool = True,
    ) -> Dict[str, Any]:
        import requests
        t0 = time.time()
        # State can be a dict or string
        state_repr = state.get("query") if isinstance(state, dict) and "query" in state else json.dumps(state, ensure_ascii=False)
        
        crit = dict(criteria)
        if allow_abstain and "UNKNOWN" not in crit:
            crit["UNKNOWN"] = "None of the other categories apply, or query is out-of-scope/unrelated."

        payload = {
            "model": self.model,
            "state": state_repr,
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": "Select the single best category matching the state context.",
                    "criteria": crit,
                }
            }
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(self.endpoint, headers=headers, json=payload, timeout=20)
        latency_ms = (time.time() - t0) * 1000
        resp.raise_for_status()
        data = resp.json()
        ans = data["answers"]["decision"]
        choice_val = ans["choice"]
        return {
            "choice": choice_val,
            "confidence": ans.get("confidence", 1.0),
            "probabilities": ans.get("probabilities", {}),
            "abstained": choice_val == "UNKNOWN",
            "latency_ms": latency_ms,
        }

class OpenJevProEngine(BaseDecisionEngine):
    """Engine using OpenJevProClient with temperature calibration and selective prediction."""
    def __init__(self, client: OpenJevProClient, name: str = "OpenJevPro (Calibrated)"):
        self.name = name
        self.client = client

    def evaluate_choice(
        self,
        state: Dict[str, Any],
        candidates: List[str],
        criteria: Dict[str, str],
        allow_abstain: bool = True,
    ) -> Dict[str, Any]:
        t0 = time.time()
        decision: ChoiceDecision = self.client.decide_choice(
            state=state,
            candidates=candidates,
            criteria=criteria,
            allow_abstain=allow_abstain
        )
        latency_ms = (time.time() - t0) * 1000
        return {
            "choice": decision.value,
            "confidence": decision.confidence,
            "probabilities": decision.probabilities,
            "abstained": decision.abstained,
            "tentative_value": decision.tentative_value,
            "latency_ms": latency_ms,
        }

class DirectStructuredEngine(BaseDecisionEngine):
    """Engine using raw open-source LLM + JSON schema (uncalibrated, no abstention)."""
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "gpt-oss:20b-cloud"):
        self.name = f"Direct LLM ({model} + JSON)"
        self.base_url = base_url.rstrip("/")
        self.model = model

    def evaluate_choice(
        self,
        state: Dict[str, Any],
        candidates: List[str],
        criteria: Dict[str, str],
        allow_abstain: bool = True,
    ) -> Dict[str, Any]:
        import requests
        t0 = time.time()
        crit = dict(criteria)
        if allow_abstain and "UNKNOWN" not in crit:
            crit["UNKNOWN"] = "None of the other categories apply, or query is out-of-scope/unrelated."

        crit_text = "\n".join([f"- {k}: {v}" for k, v in crit.items()])
        prompt = (
            f"Classify the following query into exactly one of the allowed categories:\n\n"
            f"Input: {json.dumps(state, ensure_ascii=False)}\n\n"
            f"Categories:\n{crit_text}\n\n"
            f"Output ONLY a JSON object: {{\"choice\": \"<one of the allowed category keys>\", \"confidence\": <float between 0 and 1>}}"
        )
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "format": "json",
                "think": False,
                "stream": False,
            },
            timeout=25
        )
        latency_ms = (time.time() - t0) * 1000
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
        choice_val = parsed.get("choice") or parsed.get("intent", "").strip()
        is_abstained = (choice_val == "UNKNOWN")
        return {
            "choice": choice_val,
            "confidence": float(parsed.get("confidence", 0.5)),
            "probabilities": {},
            "abstained": is_abstained,
            "latency_ms": latency_ms,
        }

class OpenJevProHarness:
    """Benchmark runner orchestrating dataset evaluation and comparative reporting."""

    def __init__(self, engines: List[BaseDecisionEngine]):
        self.engines = {e.name: e for e in engines}

    def evaluate_item(
        self,
        item: Dict[str, Any],
        candidates: List[str],
        criteria: Dict[str, str],
        allow_abstain: bool = True,
    ) -> Dict[str, Any]:
        query = item.get("text") or item.get("query")
        gt = item.get("category") or item.get("ground_truth")
        state = {"query": query}

        results = {
            "text": query,
            "ground_truth": gt,
            "engine_results": {}
        }

        # Evaluate engines concurrently
        with ThreadPoolExecutor(max_workers=len(self.engines)) as executor:
            future_to_engine = {
                executor.submit(eng.evaluate_choice, state, candidates, criteria, allow_abstain): name
                for name, eng in self.engines.items()
            }
            for fut in future_to_engine:
                name = future_to_engine[fut]
                try:
                    res = fut.result()
                    results["engine_results"][name] = res
                except Exception as e:
                    results["engine_results"][name] = {
                        "choice": "ERROR",
                        "confidence": 0.0,
                        "probabilities": {},
                        "abstained": True,
                        "latency_ms": 0.0,
                        "error": str(e)
                    }
        return results

    def run(
        self,
        dataset: List[Dict[str, Any]],
        candidates: List[str],
        criteria: Dict[str, str],
        checkpoint_file: Optional[str] = None,
        max_workers: int = 4,
        allow_abstain: bool = True,
    ) -> Dict[str, Any]:
        """Runs the benchmark across all items and compiles metric statistics."""
        records = []
        completed_texts = set()

        if checkpoint_file and os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    records = data.get("records", [])
                    completed_texts = {r["text"] for r in records}
            except Exception:
                pass

        remaining = [item for item in dataset if (item.get("text") or item.get("query")) not in completed_texts]

        def save():
            if checkpoint_file:
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump({"records": records}, f, indent=2, ensure_ascii=False)

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.evaluate_item, item, candidates, criteria, allow_abstain) for item in remaining]
            for i, fut in enumerate(futures, start=len(records) + 1):
                res = fut.result()
                records.append(res)
                save()
                eng_res = res.get("engine_results", {})
                j_c = next((v.get("choice") for k, v in eng_res.items() if "Jev (" in k), "N/A")
                oj_c = next((v.get("choice") for k, v in eng_res.items() if "OpenJev" in k), "N/A")
                d_c = next((v.get("choice") for k, v in eng_res.items() if "Direct" in k), "N/A")
                print(f"[{i:02d}/{len(dataset)}] GT: {res['ground_truth']:<22} | Jev: {j_c:<18} | OpenJev: {oj_c:<18} | Direct: {d_c:<18}", flush=True)

        total_duration = time.time() - start_time
        summary = self._compute_summary(records)
        summary["total_duration_sec"] = total_duration

        final_output = {
            "summary": summary,
            "records": records,
        }

        if checkpoint_file:
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(final_output, f, indent=2, ensure_ascii=False)

        return final_output

    @staticmethod
    def compute_ece(records: List[Dict[str, Any]], engine_name: str, n_bins: int = 10) -> float:
        """Computes Expected Calibration Error (ECE) over equal-width confidence bins."""
        bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
        confidences = []
        accuracies = []
        for r in records:
            res = r["engine_results"].get(engine_name, {})
            conf = float(res.get("confidence", 0.0))
            is_corr = 1.0 if (res.get("choice") == r["ground_truth"] or (res.get("abstained") and r["ground_truth"] == "UNKNOWN")) else 0.0
            confidences.append(conf)
            accuracies.append(is_corr)

        ece = 0.0
        total = len(confidences)
        if total == 0:
            return 0.0
        for i in range(n_bins):
            low, high = bin_boundaries[i], bin_boundaries[i + 1]
            bin_indices = [idx for idx, c in enumerate(confidences) if (low <= c < high if i < n_bins - 1 else low <= c <= high)]
            if not bin_indices:
                continue
            bin_acc = sum(accuracies[idx] for idx in bin_indices) / len(bin_indices)
            bin_conf = sum(confidences[idx] for idx in bin_indices) / len(bin_indices)
            ece += (len(bin_indices) / total) * abs(bin_acc - bin_conf)
        return round(ece, 4)

    def _compute_summary(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(records)
        if total == 0:
            return {}

        engine_names = list(self.engines.keys())
        jev_name = next((name for name in engine_names if "Jev" in name and "Open" not in name), engine_names[0])

        summary = {
            "total_samples": total,
            "engine_metrics": {}
        }

        for name in engine_names:
            correct = 0
            agreed_with_jev = 0
            abstained_cnt = 0
            latencies = []

            for r in records:
                e_res = r["engine_results"].get(name, {})
                choice = e_res.get("choice")
                gt = r["ground_truth"]

                if choice == gt:
                    correct += 1
                if e_res.get("abstained"):
                    abstained_cnt += 1

                jev_choice = r["engine_results"].get(jev_name, {}).get("choice")
                if choice == jev_choice:
                    agreed_with_jev += 1

                lat = e_res.get("latency_ms", 0.0)
                if lat > 0:
                    latencies.append(lat)

            sorted_lat = sorted(latencies) if latencies else [0.0]
            p50 = sorted_lat[int(0.50 * len(sorted_lat))]
            p95 = sorted_lat[int(0.95 * len(sorted_lat))]
            mean_lat = sum(latencies) / len(latencies) if latencies else 0.0

            summary["engine_metrics"][name] = {
                "accuracy": round((correct / total) * 100, 2),
                "correct_count": correct,
                "jev_agreement_rate": round((agreed_with_jev / total) * 100, 2),
                "abstained_count": abstained_cnt,
                "abstained_rate": round((abstained_cnt / total) * 100, 2),
                "ece": self.compute_ece(records, name),
                "mean_latency_ms": round(mean_lat, 1),
                "p50_latency_ms": round(p50, 1),
                "p95_latency_ms": round(p95, 1),
            }

        return summary
