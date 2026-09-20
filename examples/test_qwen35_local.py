import os
import json
import time
import requests

def load_key():
    with open(".env") as f:
        for line in f:
            if line.startswith("JEV_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return None

api_key = load_key()

samples = [
    {"query": "I ordered a card 10 days ago but it has not arrived yet.", "gt": "card_arrival"},
    {"query": "How do I change my security PIN code?", "gt": "change_pin"},
    {"query": "Why was I charged an extra fee on this international transfer?", "gt": "transfer_fee_charged"},
    {"query": "Write a python script to calculate fibonacci numbers.", "gt": "UNKNOWN"}
]

categories = {
    "card_arrival": "Physical card delivery or tracking status",
    "change_pin": "Changing or resetting card PIN code",
    "transfer_fee_charged": "Fees or charges on money transfers",
    "UNKNOWN": "Out of scope or unrelated request"
}

crit_text = "\n".join([f"- {k}: {v}" for k, v in categories.items()])

print(f"{'Query':<45} | {'Ground Truth':<15} | {'TypeSafe Jev':<15} | {'Qwen3.5 Local':<15} | {'Match'}")
print("-" * 105)

for s in samples:
    q = s["query"]
    gt = s["gt"]
    
    # 1. Jev
    t0 = time.time()
    r_j = requests.post("https://api.typesafe.ai/v1/systemone", headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }, json={
        "model": "jev-latest",
        "state": q,
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "Select matching category",
                "criteria": categories
            }
        }
    }, timeout=15)
    j_time = (time.time() - t0) * 1000
    j_choice = r_j.json()["answers"]["intent"]["choice"]
    
    # 2. Local Qwen 3.5
    prompt = f"Classify the following query into exactly one category.\nQuery: \"{q}\"\nCategories:\n{crit_text}\nOutput JSON ONLY: {{\"choice\": \"<category>\"}}"
    t0 = time.time()
    r_q = requests.post("http://localhost:11434/api/chat", json={
        "model": "qwen3.5:0.8b",
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",
        "think": False,
        "options": {"num_predict": 25},
        "stream": False
    }, timeout=45)
    q_time = (time.time() - t0) * 1000
    raw = r_q.json()["message"]["content"]
    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        q_choice = json.loads(clean).get("choice", clean)
    except Exception:
        q_choice = clean
        
    match = "MATCH" if j_choice == q_choice else "DIFF"
    q_disp = (q[:40] + "...") if len(q) > 40 else q
    print(f"{q_disp:<45} | {gt:<15} | {j_choice:<15} | {q_choice:<15} | {match}")
    print(f"  ↳ Latency: Jev={j_time:.1f}ms | Qwen3.5 Local (CPU)={q_time:.1f}ms")
