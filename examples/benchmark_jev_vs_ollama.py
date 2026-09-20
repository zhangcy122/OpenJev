#!/usr/bin/env python3
"""
Benchmark: TypeSafe Jev vs Open-Source LLM (Ollama gpt-oss:20b-cloud) + Structured Output
Evaluates whether a generic open-source LLM with constrained JSON output can approach Jev zero-shot.
Dataset: PolyAI Banking77 (60 balanced samples across 6 intent categories)
"""

import os
import csv
import json
import time
import urllib.request
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# Category descriptions matching standard routing criteria
CRITERIA = {
    "card_arrival": "Questions about card delivery, tracking or arrival status",
    "change_pin": "Changing, resetting or forgotten card PIN",
    "top_up_failed": "Account top up or deposit failed or rejected",
    "lost_or_stolen_card": "Reporting a lost, stolen, or compromised physical card",
    "transfer_fee_charged": "Inquiries or disputes about money transfer fees or charges",
    "extra_charge_on_statement": "Inquiries about unexpected or duplicate charges on the statement",
}

def load_jev_api_key() -> str:
    """Reads JEV_API_KEY from .env file or environment variable."""
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

def fetch_banking77_dataset(samples_per_category: int = 10) -> List[Dict[str, str]]:
    """Fetches a balanced subset of Banking77 test set from PolyAI's official repository."""
    url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
    req = urllib.request.urlopen(url)
    lines = [line.decode("utf-8") for line in req.readlines()]
    reader = csv.reader(lines)
    next(reader)  # Skip header

    target_cats = set(CRITERIA.keys())
    counts = {c: 0 for c in target_cats}
    dataset = []

    for row in reader:
        if len(row) >= 2:
            text, cat = row[0].strip(), row[1].strip()
            if cat in target_cats and counts[cat] < samples_per_category:
                counts[cat] += 1
                dataset.append({"text": text, "category": cat})

    return dataset

def query_jev(state_text: str, api_key: str) -> Dict[str, Any]:
    """Queries TypeSafe Jev systemone endpoint."""
    import requests
    t0 = time.time()
    url = "https://api.typesafe.ai/v1/systemone"
    payload = {
        "model": "jev-latest",
        "state": state_text,
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "Classify the customer utterance into the single best banking category.",
                "criteria": CRITERIA,
            }
        }
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=20)
    latency_ms = (time.time() - t0) * 1000
    resp.raise_for_status()
    data = resp.json()
    choice = data["answers"]["intent"]["choice"]
    confidence = data["answers"]["intent"].get("confidence", 1.0)
    probs = data["answers"]["intent"].get("probabilities", {})
    return {
        "choice": choice,
        "confidence": confidence,
        "probabilities": probs,
        "latency_ms": latency_ms,
        "tokens": data.get("usage", {}).get("input_tokens", 0) + data.get("usage", {}).get("output_tokens", 0),
    }

def query_ollama_structured(state_text: str, model_name: str = "gpt-oss:20b-cloud") -> Dict[str, Any]:
    """Queries Ollama with JSON structured output constraint."""
    import requests
    t0 = time.time()
    options_desc = "\n".join([f"- {k}: {v}" for k, v in CRITERIA.items()])
    prompt = (
        f"You are an intent classification engine. Classify this banking customer query into exactly one category.\n\n"
        f"Customer Query: \"{state_text}\"\n\n"
        f"Allowed Categories:\n{options_desc}\n\n"
        f"Output ONLY a valid JSON object matching this schema: "
        f'{{"intent": "<one of the allowed category keys>", "confidence": <float 0.0 to 1.0>}}'
    )
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=20)
    latency_ms = (time.time() - t0) * 1000
    resp.raise_for_status()
    data = resp.json()
    raw_content = data["message"]["content"]
    parsed = json.loads(raw_content)
    choice = parsed.get("intent", "").strip()
    confidence = float(parsed.get("confidence", 0.0))
    return {
        "choice": choice,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "completion_tokens": data.get("eval_count", 0),
    }

def evaluate_single(item: Dict[str, str], api_key: str) -> Dict[str, Any]:
    """Runs Jev and Ollama concurrently on a single sample."""
    text = item["text"]
    ground_truth = item["category"]

    with ThreadPoolExecutor(max_workers=2) as sub_exec:
        fut_jev = sub_exec.submit(query_jev, text, api_key)
        fut_ollama = sub_exec.submit(query_ollama_structured, text)
        jev_res = fut_jev.result()
        ollama_res = fut_ollama.result()

    return {
        "text": text,
        "ground_truth": ground_truth,
        "jev_choice": jev_res["choice"],
        "jev_correct": jev_res["choice"] == ground_truth,
        "jev_latency": jev_res["latency_ms"],
        "jev_confidence": jev_res["confidence"],
        "ollama_choice": ollama_res["choice"],
        "ollama_correct": ollama_res["choice"] == ground_truth,
        "ollama_latency": ollama_res["latency_ms"],
        "ollama_confidence": ollama_res["confidence"],
        "agreement": jev_res["choice"] == ollama_res["choice"],
    }

def main():
    print("=" * 70)
    print("OpenJevPro Benchmark: TypeSafe Jev vs Ollama (gpt-oss:20b) + Structured Output")
    print("Hypothesis: Can generic open LLM + schema constraint match Jev without training?")
    print("=" * 70)

    api_key = load_jev_api_key()
    print("✓ Loaded JEV_API_KEY from .env")

    print("\nFetching Banking77 dataset (6 categories, 10 samples each = 60 samples)...")
    dataset = fetch_banking77_dataset(samples_per_category=10)
    print(f"✓ Loaded {len(dataset)} samples successfully.")

    output_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    results = []
    processed_texts = set()

    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
                results = saved.get("records", [])
                processed_texts = {r["text"] for r in results}
                print(f"✓ Resumed from checkpoint: {len(results)} samples already completed.")
        except Exception as e:
            print(f"Note: Could not parse previous checkpoint: {e}")

    remaining = [item for item in dataset if item["text"] not in processed_texts]
    print(f"\nRemaining to evaluate: {len(remaining)}/{len(dataset)}")

    start_time = time.time()

    def save_checkpoint():
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"records": results}, f, indent=2, ensure_ascii=False)

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(evaluate_single, item, api_key): item for item in remaining}
        for i, fut in enumerate(futures, start=len(results) + 1):
            res = fut.result()
            results.append(res)
            save_checkpoint()
            mark_j = "✓" if res["jev_correct"] else "✗"
            mark_o = "✓" if res["ollama_correct"] else "✗"
            agreed = "MATCH" if res["agreement"] else "DIFF"
            print(f"[{i:02d}/{len(dataset)}] GT: {res['ground_truth']:<25} | Jev({mark_j}): {res['jev_choice']:<20} | Ollama({mark_o}): {res['ollama_choice']:<20} | [{agreed}]")

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Benchmark completed in {total_time:.2f}s ({total_time/60:.2f} min)")
    print("=" * 70)

    # Metrics calculation
    total = len(results)
    jev_correct_cnt = sum(1 for r in results if r["jev_correct"])
    ollama_correct_cnt = sum(1 for r in results if r["ollama_correct"])
    agreement_cnt = sum(1 for r in results if r["agreement"])

    jev_acc = (jev_correct_cnt / total) * 100
    ollama_acc = (ollama_correct_cnt / total) * 100
    agreement_rate = (agreement_cnt / total) * 100

    jev_latencies = [r["jev_latency"] for r in results]
    ollama_latencies = [r["ollama_latency"] for r in results]

    jev_mean_lat = sum(jev_latencies) / total
    ollama_mean_lat = sum(ollama_latencies) / total

    jev_sorted_lat = sorted(jev_latencies)
    ollama_sorted_lat = sorted(ollama_latencies)
    jev_p95_lat = jev_sorted_lat[int(0.95 * total)]
    ollama_p95_lat = ollama_sorted_lat[int(0.95 * total)]

    print(f"\n📊 SUMMARY RESULTS (N = {total} samples)")
    print("-" * 70)
    print(f"{'Metric':<30} | {'TypeSafe Jev':<18} | {'Ollama (gpt-oss:20b)':<18}")
    print("-" * 70)
    print(f"{'Classification Accuracy':<30} | {jev_acc:>6.2f}% ({jev_correct_cnt}/{total})   | {ollama_acc:>6.2f}% ({ollama_correct_cnt}/{total})")
    print(f"{'Decision Agreement Rate':<30} | {agreement_rate:>6.2f}% ({agreement_cnt}/{total})   | {agreement_rate:>6.2f}% ({agreement_cnt}/{total})")
    print(f"{'Mean Latency':<30} | {jev_mean_lat:>6.1f} ms           | {ollama_mean_lat:>6.1f} ms")
    print(f"{'P95 Latency':<30} | {jev_p95_lat:>6.1f} ms           | {ollama_p95_lat:>6.1f} ms")
    print("-" * 70)

    # Disagreement analysis
    disagreements = [r for r in results if not r["agreement"]]
    print(f"\n🔍 DISAGREEMENT ANALYSIS ({len(disagreements)} cases)")
    for d in disagreements:
        print(f"- Query: \"{d['text']}\"")
        print(f"  Ground Truth: {d['ground_truth']}")
        print(f"  Jev:    {d['jev_choice']} (Correct: {d['jev_correct']})")
        print(f"  Ollama: {d['ollama_choice']} (Correct: {d['ollama_correct']})")

    # Save artifact
    output_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_samples": total,
                "jev_accuracy": jev_acc,
                "ollama_accuracy": ollama_acc,
                "agreement_rate": agreement_rate,
                "jev_mean_latency_ms": jev_mean_lat,
                "ollama_mean_latency_ms": ollama_mean_lat,
                "total_duration_sec": total_time,
            },
            "records": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved full results to: {output_path}")

if __name__ == "__main__":
    main()
