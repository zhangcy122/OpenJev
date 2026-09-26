# OpenJevPro

<p align="left">
  <b>English</b> | <a href="README_zh.md">简体中文</a>
</p>

> **The High-Throughput Deterministic Decision Engine for AI Agents**  
> Turn slow, expensive, prompt-biased LLM calls (>1.5s, $0.02/call) into **sub-35ms, strictly typed, calibrated, 100% order-invariant decision primitives** (`Choice<T>`, `Noul`, `Score`). Features a self-evolving **"Explore First, Crystallize Later" cognitive flywheel** that distills System 2 reasoning traces into permanent System 1 fast paths at zero repeated cost.  
> 🌐 **Official Website**: [https://openjev.pro](https://openjev.pro)

---

## ⚡ Quickstart in 60 Seconds

### 1. Installation

```bash
pip install openjevpro
```

*(Or clone for bleeding-edge source: `git clone https://github.com/zhangcy122/OpenJev.git && cd OpenJev && pip install -r requirements.txt`)*

### 2. Zero-GPU Instant CLI Demo (5 Seconds)

No GPU or backend inference server required. Run the interactive terminal demonstration with full receipts (<15ms):

```bash
python3 -m openjevpro.demo
```

### 3. High-Throughput Agent Decision (Zero Order Bias)

```python
from openjevpro import OpenJevProClient

# Production: Point to your local vLLM, SGLang, Ollama, or Laya endpoint
client = OpenJevProClient(base_url="http://localhost:8000/v1")

# Offline development & CI/CD unit testing (zero GPU required):
# client = OpenJevProClient(mock=True)

# Sub-35ms categorical decision with 100% mathematical order invariance
decision = client.decide_choice(
    state={"ticket": "Unauthorized login attempt detected from an unknown IP address."},
    candidates=["billing", "technical_support", "security_fraud", "human_review"],
    order_invariant=True,  # Eliminates prompt positioning bias across all N! candidate permutations
)

print(f"Decision: {decision.value}")        # 'security_fraud'
print(f"Confidence: {decision.confidence}")  # 0.985 (Calibrated)
print(f"Abstained: {decision.abstained}")    # False
```

---

## 🎯 Why OpenJevPro?

Modern AI agents require high-frequency deterministic choices: **tool selection, intent routing, state machine transitions, and safety gatekeeper checks**. 

However, calling full generative LLMs for routing creates three critical liabilities:
1. **Excessive Latency & Cost**: 800ms–2500ms per routing step, burning hundreds of output tokens and dollars per million requests.
2. **Schema Instability**: Random JSON formatting glitches and markdown wrap failures that crash production pipelines.
3. **Severe Option Order Bias**: In causal autoregressive models, permuting the candidate order in the prompt causes classification results to flip by **15%–20%**, failing compliance and auditability.

```
                              OpenJevPro Unified Architecture
                              
                    ┌───────────────────────────────────────────────┐
                    │            Core Decision Primitives           │
                    │   Choice<T>  •  Noul (Binary)  •  Score       │
                    │   Sub-35ms  •  Strict Types  •  Calibrated    │
                    └───────────────────────┬───────────────────────┘
                                            │
               ┌────────────────────────────┴────────────────────────────┐
               ▼                                                         ▼
     [ Flagship Feature 1 ]                                    [ Flagship Feature 2 ]
  100% Order Invariance (1.00)                             Deliberative Decision Flywheel
  Isolated Candidate Scoring                               "Explore First, Crystallize Later"
  Zero Position Bias in Regulated Audits                    System 2 LLM Distilled to System 1
               │                                                         │
               └────────────────────────────┬────────────────────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
          [ Deployment Strategy 1 ]                     [ Deployment Strategy 2 ]
         Pure Local / Open-Source                     Enterprise Hybrid Gateway
         vLLM / SGLang / Laya 322M                    TypeSafe Jev + CircuitBreaker SLA
         100% On-Premise, $0 API Cost                 60%+ Cloud Cost Cut, 0 Silent Errors
```

---

## 🔥 Twin Flagship Capabilities

### 1. 100% Mathematical Order Invariance (`order_invariant=True`)

In conventional single-prompt evaluations (e.g., `Option A: ... Option B: ...`), LLMs exhibit strong attention bias toward early tokens or recency tokens. On borderline samples, shuffling the options flips the decision.

OpenJevPro guarantees **100% mathematical order invariance (1.00 distinct winners across all $N!$ candidate orderings)**:
- Evaluates each candidate option in an **isolated forward scoring pass** without competitor tokens present in context.
- Normalizes scores across candidates via a strictly commutative softmax operator.

```python
# Evaluates candidates in isolated passes via parallel workers
decision = client.decide_choice(
    state={"query": "Where is my debit card? It never arrived."},
    candidates=["card_arrival", "lost_or_stolen_card", "change_pin"],
    order_invariant=True,  # Guarantees 1.00 permutation invariance (ΔP < 10^-4)
)
```

*(Note: For single-pass zero-drift execution in sub-35ms, use the dedicated **Laya ModernBERT-large 322M** encoder, which is natively bidirectional).*

### 2. Deliberative Decision Flywheel ("Explore First, Crystallize Later")

Solves the trade-off between cheap/fast classifiers and expensive/slow reasoning LLMs (o1, DeepSeek R1):
- **Fast Path (System 1)**: 90%+ of standard high-confidence queries execute in **<35ms**.
- **Deliberative Reasoning (System 2)**: Borderline, low-confidence, or out-of-distribution queries automatically escalate to a thinking model for deep causal deduction.
- **Crystallization Operator**: Distills the reasoning chain into discriminative boundary rules and precedent memory, **permanently promoting subsequent similar queries to the sub-35ms fast path with zero repeated LLM cost**.

```python
from openjevpro.flywheel import DeliberativeDecisionFlywheel, CrystallizationStore

flywheel = DeliberativeDecisionFlywheel(
    fast_engine=client,                  # System 1: sub-35ms local engine
    reasoning_engine=thinking_llm,        # System 2: DeepSeek R1 / o1 reasoning model
    crystallization_store=CrystallizationStore(),
    auto_crystallize=True,
)

# First ambiguous query escalates to System 2, crystallizes the decision boundary,
# and promotes the pattern. All subsequent requests run on System 1 fast path (<35ms)!
decision = flywheel.evaluate_choice(
    state={"query": "Card package intercepted in transit"},
    candidates=["card_arrival", "lost_or_stolen_card"],
)
```

---

## 📦 Two Pragmatic Deployment Strategies

### Strategy 1: Pure Local / Open-Source (Edge & Private Cluster)
Run completely offline with zero external cloud dependencies:
* **Open LLMs (Qwen3 / Gemma 4 / DeepSeek)**: Run locally via vLLM, SGLang, or Ollama with decode-time grammar masks.
* **Laya Engine (`LayaEngine`)**: Native support for Convai's open-weight **ModernBERT-large 322M** encoder. Achieves ~33ms bidirectional inference with zero prompt drift.

```python
from openjevpro.harness import LayaEngine

# Sub-35ms local inference via Laya ModernBERT microservice
laya = LayaEngine(endpoint="http://localhost:8001/v1", model="convai/laya-modernbert-large")
result = laya.evaluate_choice(
    state={"query": "Cancel order #4829 and issue refund"},
    candidates=["cancel_order", "track_shipment", "contact_seller"],
)
print(f"Choice: {result['choice']}, Latency: {result['latency_ms']:.1f}ms")
```

### Strategy 2: Enterprise Hybrid Gateway & Cost Arbitrage
Co-exist with existing commercial TypeSafe Jev or proprietary cloud LLM APIs:
* **TypeSafeJevGuardHarness**: Intercepts overconfident misclassifications on borderline samples via pseudo-logit scaling and dynamic cutoff $\tau = \max(\tau_{\min}, \alpha / K)$, lifting selective precision to **100.00%**.
* **HybridJevGateway**: Routes 70%+ of high-frequency intent queries to local Tier 1 models (<20ms, $0 cloud cost). Only edge cases escalate to cloud APIs, slashing monthly bills by **60%+**.
* **CircuitBreaker SLA**: Automatically catches cloud timeouts and HTTP 429 rate limits, gracefully degrading to calibrated local decisions without unhandled 500 exceptions.

```python
from openjevpro.gateway import HybridJevGateway, CircuitBreaker

gateway = HybridJevGateway(
    edge_engine=local_client,           # Tier 1 Local (<35ms, $0)
    cloud_engine=commercial_jev_engine, # Tier 2 Cloud API
    circuit_breaker=CircuitBreaker(failure_threshold=3, recovery_time=30.0),
    confidence_threshold=0.80
)
decision = gateway.decide_choice(state={"query": "What are your international fees?"}, candidates=tiers)
```

---

## 💡 System Primitives & Mechanics

| Primitive | Description | Output Guarantee |
| :--- | :--- | :--- |
| **`Choice<T>`** | Multi-class categorical decision over an enum set | Strict enum matching + normalized probability distribution |
| **`order_invariant=True`** | Isolated candidate scoring with commutative softmax | 100% mathematical permutation invariance across all $N!$ orderings |
| **`Noul`** | Binary truth judgment (`TRUE` / `FALSE`) | Calibrated $P(\text{true})$ + uncertainty interval |
| **`Score`** | Ordinal evaluation across predefined severity/rank tiers | Probability mass across tiers + expected score |
| **`TypeSafeJevGuardHarness`** | Runtime calibration & adaptive safety guard | Pseudo-logit inversion, temperature scaling & dual-threshold cutoff |
| **`DeliberativeDecisionFlywheel`**| Self-evolving cognitive flywheel (System 2 $\to$ System 1) | Sub-35ms promoted fast path + zero repeated deliberation cost |
| **`MockClient`** (`mock=True`) | Zero-GPU offline deterministic simulation client | Zero network or GPU dependency; offline CI unit testing with exact receipts |

All primitives incorporate first-class **Abstention & Fallback Options** (`UNKNOWN`, `OUT_OF_SCOPE`, `HUMAN_REVIEW`) to eliminate artificial probability spikes caused by closed candidate sets.

---

## 📊 Empirical Benchmarks & Validation

Empirical validation on **PolyAI Banking77** (30 in-domain queries across 6 financial categories + 6 out-of-scope queries) demonstrates that OpenJevPro with adaptive calibration matches and exceeds commercial dedicated endpoints:

| Decision Engine | Naive Accuracy | Selective Accuracy (On Answered) | Tentative Top-1 Accuracy | Out-of-Scope Rejection | Latency (P50) | ECE (10-bin) | Reliability Architecture |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **OpenJevPro (`gemma4:cloud`)** | 94.4% (34/36) | **100.0% (28/28)** | **100.0% (36/36)** | **100.0% (6/6)** | 1,084 ms | **0.025** *(Calibrated)* | **Temperature Calibrated + Selective Abstention + Auto-repair JSON** |
| **Direct Open LLM (`gemma4:cloud`)** | **97.2%** (35/36) | 96.7% (29/30) | 97.2% (35/36) | **100.0% (6/6)** | 630 ms | 0.029 *(Raw JSON)* | Direct JSON Schema (No confidence calibration) |
| **TypeSafe Jev (1.13.0 Live)** | **97.2%** (35/36) | 96.7% (29/30) | 97.2% (35/36) | **100.0% (6/6)** | 740 ms | 0.026 *(Proprietary)* | Dedicated System 1 / RLCD (Symmetric UNKNOWN verified) |

### Independent Validation
1. 📈 **[JevBench v1](https://benchmarkheaven.com/jev-models)**: 242-case cross-comparison of Jev vs general LLMs vs openjev-sglang showing statistical parity.
2. 🚀 **[ekzhang/openjev-sglang](https://github.com/ekzhang/openjev-sglang)**: Open implementation of SGLang constrained-grammar decoding replication against Jev.
3. 🔬 **[iammrduncan/typesafe-ai-benchmark](https://github.com/iammrduncan/typesafe-ai-benchmark)**: Open reproducibility study testing Qwen 3.8 27B / Cerebras schema-constrained structured output vs Jev.

---

## 🤖 Recommended Base Models (2025–2026 Tiers)

| Tier | Model | Architecture & Active Params | Context | Recommended Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 0: Dedicated System 1 (<35ms)** | **Laya (Convai)** | **ModernBERT-large (322M)** • Non-autoregressive | 8K | **Ultra-fast intent classifier**: ~33ms, zero cloud cost, local HTTP microservice (Apache 2.0) |
| **Tier 1: Edge & Ultra-Fast Gate (<50ms)** | **Qwen3-1.7B / 4B** | Dense (1.7B / 4B) • `/no_think` switch | 32K | Default lightweight router, binary gates, high-QPS routing |
| | **Gemma 4 E2B / E4B** | Dense (2.3B / 4.5B effective) | 128K | Edge & local multimodal (vision + text) classification |
| **Tier 2: Production Workflow Routing (50–120ms)** | **Qwen3-30B-A3B** | MoE (30B total, 3B active / token) | 128K | **Golden Standard**: 3B inference cost with 30B representation capacity |
| | **Mistral Small 3.2 / 4** | 24B Dense / 119B MoE (6B active) | 128K–256K | General enterprise workflow classification and agent routing |
| **Tier 3: Complex Arbitration & Flywheel System 2** | **DeepSeek-V4.1-Flash / R1** | MoE (552B total) • API/Cluster | 1M | Hard-sample counterfactual exploration, Crystallization operator compilation |

---

## ⚖️ License

* **Core Engine & Open Primitives**: PolyForm Noncommercial License 1.0.0 (Free for research, evaluation, and non-commercial development).
* **Commercial Deployment**: Commercial production licenses are available for enterprise agent infrastructure. Please refer to [LICENSE.md](LICENSE.md) or visit [https://openjev.pro](https://openjev.pro).
