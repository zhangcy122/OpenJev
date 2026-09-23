#!/usr/bin/env python3
"""
OpenJevPro Benchmark Harness Runner
Runs comparative evaluation:
1. TypeSafe Jev
2. OpenJevPro Harness (Calibrated + Selective Abstention)
3. Direct Open LLM (gpt-oss:20b + Raw JSON Schema)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import json
import time
import urllib.request
from typing import List, Dict, Any

from openjevpro.client import OpenJevProClient
from openjevpro.harness import (
    OpenJevProHarness,
    TypeSafeJevEngine,
    OpenJevProEngine,
    DirectStructuredEngine,
)

CRITERIA = {
    "card_arrival": "Questions about physical card delivery, dispatch or arrival status",
    "change_pin": "Changing, resetting or setting physical card PIN code",
    "top_up_failed": "Account top-up, deposit or transfer into balance failed or rejected",
    "lost_or_stolen_card": "Reporting a lost, stolen, or compromised physical card",
    "transfer_fee_charged": "Inquiries or complaints about fees or charges for money transfers",
    "extra_charge_on_statement": "Inquiries about unrecognized, duplicate or extra charges on statements",
}

def load_jev_api_key() -> str:
    key = os.environ.get("JEV_API_KEY")
    if key:
        return key.strip()
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("JEV_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("\"'")
    raise ValueError("JEV_API_KEY not found in environment or .env file")

def prepare_dataset() -> List[Dict[str, str]]:
    """Fetches Banking77 test samples and adds out-of-scope samples to test abstention."""
    url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
    req = urllib.request.urlopen(url)
    lines = [line.decode("utf-8") for line in req.readlines()]
    reader = csv.reader(lines)
    next(reader)

    target_cats = set(CRITERIA.keys())
    counts = {c: 0 for c in target_cats}
    dataset = []

    # 5 samples per category = 30 in-domain samples
    for row in reader:
        if len(row) >= 2:
            text, cat = row[0].strip(), row[1].strip()
            if cat in target_cats and counts[cat] < 5:
                counts[cat] += 1
                dataset.append({"text": text, "category": cat, "type": "in_domain"})

    # 6 Out-of-Scope / Ambiguous test samples
    oos_samples = [
        {"text": "Write a python script to calculate the Fibonacci series.", "category": "UNKNOWN", "type": "out_of_scope"},
        {"text": "What is the weather like in Tokyo right now?", "category": "UNKNOWN", "type": "out_of_scope"},
        {"text": "Thank you so much, your team was super helpful today!", "category": "UNKNOWN", "type": "chitchat"},
        {"text": "Can I trade Bitcoin and Ethereum on your mobile app?", "category": "UNKNOWN", "type": "out_of_scope"},
        {"text": "How many employees work at the company headquarters?", "category": "UNKNOWN", "type": "out_of_scope"},
        {"text": "Please book a table for four at the Italian bistro tonight.", "category": "UNKNOWN", "type": "out_of_scope"},
    ]
    dataset.extend(oos_samples)
    return dataset

def main():
    print("=" * 75)
    print("OpenJevPro Benchmark Harness: 3-Way Comparative Evaluation")
    print("1. TypeSafe Jev (Commercial Reference)")
    print("2. OpenJevPro (Calibrated Posterior + Selective Abstention Layer)")
    print("3. Direct LLM (gpt-oss:20b + Raw JSON Schema)")
    print("=" * 75)

    jev_api_key = load_jev_api_key()
    print("✓ Loaded JEV_API_KEY")

    # Initialize Engines
    jev_engine = TypeSafeJevEngine(api_key=jev_api_key)
    model_name = os.environ.get("OLLAMA_MODEL", "gemma4:cloud")
    openjev_client = OpenJevProClient(
        base_url="http://localhost:11434",
        model=model_name,
        temperature_scaling=1.35,
        abstain_threshold=0.40,
        backend="ollama",
    )
    openjev_engine = OpenJevProEngine(client=openjev_client, name=f"OpenJevPro (Calibrated {model_name})")
    direct_engine = DirectStructuredEngine(base_url="http://localhost:11434", model=model_name)

    harness = OpenJevProHarness(engines=[jev_engine, openjev_engine, direct_engine])

    dataset = prepare_dataset()
    print(f"✓ Prepared test suite: {len(dataset)} items (30 in-domain Banking77 + 6 out-of-scope)")

    output_path = os.path.join(os.path.dirname(__file__), "harness_benchmark_results.json")
    print(f"\nRunning harness evaluation with 6 workers...")
    
    t0 = time.time()
    results = harness.run(
        dataset=dataset,
        candidates=list(CRITERIA.keys()),
        criteria=CRITERIA,
        checkpoint_file=output_path,
        max_workers=6
    )
    duration = time.time() - t0

    print(f"\n✓ Completed in {duration:.2f} seconds!")
    summary = results["summary"]
    engine_metrics = summary["engine_metrics"]

    print("\n" + "=" * 75)
    print("📊 OPENJEVPRO HARNESS BENCHMARK RESULTS")
    print("=" * 75)
    header = f"{'Engine':<28} | {'Accuracy':<10} | {'Jev Agree':<10} | {'Abstained':<10} | {'Mean Latency':<12} | {'P95 Latency':<12}"
    print(header)
    print("-" * 75)

    for eng_name, m in engine_metrics.items():
        acc_str = f"{m['accuracy']}%"
        agree_str = f"{m['jev_agreement_rate']}%"
        abst_str = f"{m['abstained_count']} ({m['abstained_rate']}%)"
        mean_lat = f"{m['mean_latency_ms']} ms"
        p95_lat = f"{m['p95_latency_ms']} ms"
        print(f"{eng_name:<28} | {acc_str:<10} | {agree_str:<10} | {abst_str:<10} | {mean_lat:<12} | {p95_lat:<12}")

    print("-" * 75)

    # Detailed OOS analysis
    print("\n🛡️ OUT-OF-SCOPE & ABSTENTION BEHAVIOR (Selective Prediction)")
    print("-" * 75)
    for r in results["records"]:
        if r["ground_truth"] == "UNKNOWN":
            print(f"Query: \"{r['text']}\"")
            for eng_name in harness.engines:
                res = r["engine_results"].get(eng_name, {})
                choice = res.get("choice")
                abst = "ABSTAINED" if res.get("abstained") else "PREDICTED"
                conf = res.get("confidence", 0.0)
                print(f"  - {eng_name:<26}: Choice={choice:<15} [{abst}] (Conf={conf:.2f})")
            print()

if __name__ == "__main__":
    main()
