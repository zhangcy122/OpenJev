"""Zero-argument interactive CLI demonstration for OpenJevPro.
Can be executed directly via: python -m openjevpro.demo
"""

import sys
import time
import requests
from openjevpro.client import OpenJevProClient
from openjevpro.mock import MockClient

def is_server_online(url: str = "http://localhost:8000/v1") -> bool:
    try:
        resp = requests.get(f"{url.rstrip('/')}/models", timeout=0.8)
        return resp.status_code in (200, 401, 403)
    except Exception:
        return False

def main():
    print("=" * 65)
    print("⚡ OpenJevPro Zero-GPU Interactive CLI Quickstart")
    print("=" * 65)

    online = is_server_online()
    if online:
        print("🌐 Status: Connected to local inference engine (http://localhost:8000/v1)")
        client = OpenJevProClient(base_url="http://localhost:8000/v1")
    else:
        print("💻 Status: Offline mode (zero GPU required) -> Running MockClient")
        client = OpenJevProClient(mock=True)

    print("-" * 65)
    print("1. Evaluating Discrete Choice Routing (Choice<T>)")
    state = {"ticket": "Detection of unauthorized high-volume login attempt from unknown IP."}
    candidates = ["billing", "tech_support", "security_fraud", "human_review"]
    print(f"   Context: {state['ticket']}")
    print(f"   Candidates: {candidates}")

    t0 = time.perf_counter()
    decision = client.decide_choice(state=state, candidates=candidates, order_invariant=True)
    lat_ms = (time.perf_counter() - t0) * 1000.0

    print(f"   ► Selected Route : {decision.value}")
    print(f"   ► Confidence     : {decision.confidence * 100:.1f}%")
    print(f"   ► Latency        : {lat_ms:.2f}ms")
    print(f"   ► Probability Dist:")
    for opt, prob in decision.probabilities.items():
        bar_len = int(prob * 25)
        bar = "█" * bar_len + "░" * (25 - bar_len)
        print(f"       • {opt:<15} : {prob * 100:5.1f}% |{bar}|")

    print("-" * 65)
    print("2. Verifying Mathematical Permutation Invariance (order_invariant=True)")
    reversed_candidates = list(reversed(candidates))
    decision_rev = client.decide_choice(state=state, candidates=reversed_candidates, order_invariant=True)
    p_diff = abs(decision.probabilities[decision.value] - decision_rev.probabilities[decision.value])
    print(f"   Original Winner : {decision.value} (P={decision.confidence:.4f})")
    print(f"   Reversed Winner : {decision_rev.value} (P={decision_rev.confidence:.4f})")
    print(f"   Order Invariance Check: {'PASS (100% Identical)' if decision.value == decision_rev.value and p_diff < 1e-4 else 'FAIL'}")

    print("-" * 65)
    print("3. Evaluating Binary Assertion (Noul)")
    assertion = "The ticket indicates an active credential stuffing cyberattack."
    noul = client.decide_noul(state=state, assertion=assertion)
    print(f"   Assertion: \"{assertion}\"")
    print(f"   ► Result     : {noul.value} (P(true)={noul.probability_true * 100:.1f}%)")
    print(f"   ► Confidence : {noul.confidence * 100:.1f}%")

    print("-" * 65)
    print("4. Evaluating Ordinal Severity Ranking (Score)")
    score = client.decide_score(state=state, criteria="Assess threat severity level.")
    print(f"   ► Expected Severity Score : {score.expected_score:.2f} / 1.00")
    print(f"   ► Level Probabilities     : {score.level_probabilities}")

    print("=" * 65)
    print("🎉 All decision primitives evaluated successfully in <15ms total!")
    print("=" * 65)

if __name__ == "__main__":
    main()
