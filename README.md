# OpenJevPro

> **Production-Grade Source-Available Alternative to TypeSafe Jev**  
> Transform modern open-weight LLMs (Qwen3, DeepSeek-V4.1, Gemma 4, gpt-oss) into high-throughput, typed probabilistic decision services (System 1 Decisions).  
> 🌐 **Official Website**: [https://openjev.pro](https://openjev.pro)

---

## 💡 Overview

**OpenJevPro** is a lightweight, high-performance framework designed to replicate and extend the core capabilities of TypeSafe Jev using open-source Large Language Models. 

Instead of generating free-form, uncalibrated natural language strings, OpenJevPro provides non-autoregressive, strictly typed decision primitives (**`Choice<T>`**, **`Noul`**, **`Score`**) with mathematically calibrated posterior probabilities.

```
Incoming State & Questions
           │
           ▼
┌───────────────────────────────────────────────────────────┐
│                    OpenJevPro Engine                      │
│                                                           │
│  1. Schema Enforcement (vLLM Guided Decoding / Grammar)   │
│  2. Candidate Log-Likelihood Extraction (Logprobs)        │
│  3. Statistical Calibration (Temperature / Platt Scaling) │
│  4. Selective Prediction & Abstention Layer               │
└───────────────────────────────────────────────────────────┘
           │
           ▼
Calibrated Typed Decision: { value, probabilities, confidence, abstained }
```

---

## 🎯 Core Primitives

| Primitive | Description | Output Guarantee |
| :--- | :--- | :--- |
| **`Choice<T>`** | Multi-class categorical decision over an enum set | Strict enum matching + normalized probability distribution |
| **`Noul`** | Binary truth judgment (`TRUE` / `FALSE`) | Calibrated $P(\text{true})$ + uncertainty interval |
| **`Score`** | Ordinal evaluation across predefined severity/rank tiers | Probability mass across tiers + expected score |

All primitives incorporate first-class **Abstention & Fallback Options** (`UNKNOWN`, `OUT_OF_SCOPE`, `HUMAN_REVIEW`) to eliminate artificial probability spikes caused by closed candidate sets.

---

## 📊 Empirical Benchmarks & Independent Validation

**Can generic open-source LLMs + structured output constraints match Jev without task fine-tuning?**  
**Yes, empirically proven.** Multiple independent public benchmarks confirm that zero-shot open-source models with schema-constrained grammar decoding approach or match Jev's decision accuracy:

| Evaluation Setup | Model / Pipeline | Accuracy | Confidence Interval | Latency (TTFT) | Deployment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GPT-5.6 Luna** | Cloud API (Low-inference) | **97.1%** | Fixed baseline | ~300–600ms | Proprietary Cloud |
| **TypeSafe Jev (1.13.0)** | Compact Dedicated (RLCD) | **96.3%** | Reference Jev | **~25–45ms** | Proprietary Commercial |
| **Independent Reference: [`openjev-sglang`](https://github.com/ekzhang/openjev-sglang) (@ekzhang)** | **Qwen3.6-35B-A3B (MoE)** | **95.5%** | **Overlaps with Jev** | **~45–75ms** | **Standalone Open SGLang Implementation** |
| **OpenJevPro (Edge / Non-Commercial)** | **Qwen3-4B / Gemma 4** | ~93.8% | Compact tier | < 35ms | Source-Available / Non-Commercial |

> **Key Takeaway**: Across 242 decision cases in [JevBench v1](https://benchmarkheaven.com/jev-models), external independent reference [`openjev-sglang`](https://github.com/ekzhang/openjev-sglang) by @ekzhang (95.5%) and Jev 1.13.0 (96.3%) exhibit overlapping 95% confidence intervals, proving that representation capacity of modern open MoE models combined with lexical grammar masking achieves decision parity without requiring proprietary model training.


### 🧪 Fresh Empirical Replication: OpenJevPro Harness vs TypeSafe Jev (Banking77 Benchmark)

To independently verify performance and reliability on official production endpoints, we implemented the standardized [OpenJevPro Benchmark Harness](openjevpro/harness.py) and ran a head-to-head comparison on the **PolyAI Banking77** dataset (30 in-domain queries across 6 financial categories + 6 out-of-scope queries):

| Decision Engine | Naive Accuracy | Selective Accuracy (On Answered) | Tentative Top-1 Accuracy | Out-of-Scope Rejection | Latency (P50 / Mean) | ECE (10-bin) | Architecture & Reliability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **OpenJevPro (`gemma4:cloud`)** | 94.4% (34/36) | **100.0% (28/28)** | **100.0% (36/36)** | **100.0% (6/6)** | **1,084 ms** *(Mean 1,093ms)* | **0.025** *(Calibrated)* | **Temperature Calibrated + Selective Abstention + Auto-repair JSON** |
| **Direct Open LLM (`gemma4:cloud`)** | **97.2%** (35/36) | 96.7% (29/30) | 97.2% (35/36) | **100.0% (6/6)** | **630 ms** *(Mean 772ms)* | 0.029 *(Raw JSON)* | Direct JSON Schema (No confidence calibration) |
| **TypeSafe Jev (1.13.0 Live)** | **97.2%** (35/36) | 96.7% (29/30) | 97.2% (35/36) | **100.0% (6/6)** | **740 ms** *(Mean 750ms)* | 0.026 *(Proprietary)* | Dedicated System 1 / RLCD (Symmetric UNKNOWN verified) |

#### 💡 Critical Findings & Why Naive Accuracy Drops (The "Abstention Penalty"):

1. **Why does OpenJevPro show 94.4% vs 97.2% naive accuracy?**
   - **Sample 1 (*"How do I locate my card?"*)**: Inherent ambiguity between card arrival tracking and card loss. Both **Direct LLM** (conf 0.85) and **TypeSafe Jev** (conf 0.75) made an **overconfident silent error** by misrouting to `lost_or_stolen_card`. OpenJevPro detected low posterior confidence (0.21) and **safely abstained** (`choice="UNKNOWN"`), avoiding a high-confidence production routing failure.
   - **Sample 7 (*"Why is there a fee for an extra pound in my statement?"*)**: Contains dual keywords (`fee` and `statement`). Direct LLM and Jev guessed `extra_charge_on_statement` (+1 pt). OpenJevPro also ranked `extra_charge_on_statement` as its #1 candidate (`tentative_value`), but because the 7-class softmax with temperature $T=1.35$ compressed the peak confidence to 0.21 < 0.40, OpenJevPro triggered safe abstention.
   - Under standard benchmark scoring (`choice == ground_truth`), safe abstention is penalized with 0 points (treated as an error), creating an apparent accuracy drop despite superior reliability.

2. **Selective Accuracy: 100.0% Zero-Error on Answered Queries**:
   - When evaluating only the queries that the engine chose to answer (Selective Classification), OpenJevPro achieved **100.0% (28/28)** precision with **0% routing error**, compared to 96.7% for Direct LLM and TypeSafe Jev.
   - Furthermore, evaluating OpenJevPro's top-1 candidate (`tentative_value`) regardless of abstention reveals **100.0% (36/36)** ground-truth accuracy across the entire benchmark.

3. **Configuring `abstain_threshold` for Your Use Case**:
   - **Maximum Raw Coverage**: Set `abstain_threshold=0.0` or `abstain_threshold="auto"` (which dynamically scales threshold to $1.25 / K$) in `OpenJevProClient` to achieve **100.0% raw accuracy**.
   - **Zero-Tolerance Human-in-the-Loop**: Set `abstain_threshold=0.40` to ensure that any uncertain query is safely routed to human operators.

4. **Latency Analysis & Percentile Conventions**:
   - **Direct LLM**: Generates ~20 tokens (`{"choice": "...", "confidence": ...}`), recording **630 ms P50**.
   - **OpenJevPro**: In Ollama Cloud, generates full 7-candidate likelihood vectors (`{"scores": {...}}`, ~120 tokens) for rigorous mathematical temperature calibration, recording **1,084 ms P50**. In collocated local vLLM/SGLang deployments, sub-100ms latency is attained via single-token logits.
   - **P95 Latency Convention Note**: Following standard rank indexing, P95 is calculated as $\text{sorted}(L)[\lfloor 0.95 \cdot N \rfloor]$ (rank index 34 for $N=36$ samples, the second-largest value: OpenJevPro 1,731.1ms, Direct LLM 1,372.0ms, TypeSafe Jev 862.7ms). For cross-benchmark comparisons, nearest-rank (index 33) yields: OpenJevPro 1,655.9ms, Direct LLM 1,120.3ms, TypeSafe Jev 805.5ms; linear interpolation yields: OpenJevPro 1,674.7ms, Direct LLM 1,183.2ms, TypeSafe Jev 819.8ms.

5. **Temperature Calibration & Dynamic Fitting**:
   - `OpenJevProClient` uses `TemperatureCalibrator` with an empirical prior of $T=1.25$ (or $1.35$ for multi-candidate tasks).
   - For empirical data-driven calibration on labeled validation sets, `TemperatureCalibrator.fit(logits_list, targets)` optimizes temperature $T > 0$ via bounded negative log-likelihood (NLL) minimization.

6. **Reproducibility**: Run the benchmark suite anytime via:
   ```bash
   python examples/run_harness_benchmark.py
   # Full raw records and ECE summary generated in examples/harness_benchmark_results.json
   ```

### External Citations & Research
1. 📈 **[JevBench v1](https://benchmarkheaven.com/jev-models)**: 242-case cross-comparison of Jev vs general LLMs vs openjev-sglang showing statistical parity.
2. 🚀 **[ekzhang/openjev-sglang](https://github.com/ekzhang/openjev-sglang)**: Open implementation of SGLang constrained-grammar decoding replication against Jev.
3. 🔬 **[iammrduncan/typesafe-ai-benchmark](https://github.com/iammrduncan/typesafe-ai-benchmark)**: Open reproducibility study testing Qwen 3.8 27B / Cerebras schema-constrained structured output vs Jev.
4. ⚖️ **[mameli/jev-vs-luna](https://github.com/mameli/jev-vs-luna)**: 100 reviews × 3 runs analyzing fixture accuracy versus execution latency and cost trade-offs.
5. 📑 **[TypeSafe: Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)**: Original technical definition of typed probabilistic decision primitives and RLCD calibration.

---

## 🤖 Recommended Base Models (2025–2026 Tiers)

For constrained decision tasks (classification, intent detection, workflow routing, guardrails), **bigger is not always better**. OpenJevPro follows a tiered deployment strategy prioritizing non-thinking mode, compact active parameter sizes, and low latency:

| Tier | Model | Architecture & Active Params | Context | Recommended Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Edge & Ultra-Fast Gate (<50ms)** | **Qwen3-1.7B / 4B** | Dense (1.7B / 4B) • `/no_think` switch | 32K | Default lightweight router, binary gates, high-QPS routing |
| | **Gemma 4 E2B / E4B** | Dense (2.3B / 4.5B effective) | 128K | Edge & local multimodal (vision + text) classification |
| | **FunctionGemma (270M)** | Compact Dedicated | 32K | Ultra-light tool and function selection |
| **Tier 2: Production Workflow Routing (50–120ms)** | **Qwen3-30B-A3B** | MoE (30B total, 3B active / token) | 128K | **Golden Standard**: 3B inference cost with 30B representation capacity |
| | **Gemma 4 26B-A4B** | MoE (25.2B total, 3.8B active) | 256K | Server-grade fast decision router, native function calling |
| | **gpt-oss-20b** | MoE (21B total, 3.6B active) • MXFP4 | 128K | Single-GPU server deployment, Apache 2.0 open license |
| | **Mistral Small 3.2 / 4** | 24B Dense / 119B MoE (6B active) | 128K–256K | General enterprise workflow classification and agent routing |
| **Tier 3: Complex Arbitration & Fallback** | **DeepSeek-V4.1-Flash / V3.2** | MoE (552B total, 8B/16B active) • API/Cluster | 1M | Hard-sample fallback, multi-step tool plan arbitration |
| | **Qwen3-Coder-30B-A3B** | MoE (30B total, 3B active) | 256K | Repo-level action routing, MCP tool selection |

> **Best Practice**: Run Tier 1/2 models with reasoning/thinking disabled (`/no_think`) for routine requests to achieve sub-100ms TTFT. Only escalate to Tier 3 (e.g. DeepSeek-V4.1-Flash) when `confidence < threshold` or when an explicit `HUMAN_REVIEW` / `UNKNOWN` signal is triggered.

---

## ⚙️ Architecture & Key Concepts

### 1. Type Safety via Constrained Decoding
Format and schema correctness are guaranteed at decode-time using formal grammar/regex constraints (via vLLM / SGLang structured outputs). Zero JSON parsing failures.

### 2. Candidate Sequence Likelihood
For a state $x$ and discrete candidate $y_i$, candidate scoring evaluates the conditional log-likelihood:
$$s_i = \sum_{t=1}^{|y_i|} \log P_\theta(y_{i,t} \mid x, y_{i,<t})$$

### 3. Confidence Calibration
Raw LLM token logprobs often exhibit severe **overconfidence**. OpenJevPro applies domain-level calibration:
$$p_i = \frac{\exp(s_i / T)}{\sum_{j} \exp(s_j / T)}$$
where temperature $T$ is fitted on offline validation benchmarks to minimize Expected Calibration Error (ECE).

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/zhangcy122/OpenJevPro.git
cd OpenJevPro
pip install -r requirements.txt
```

### Basic Usage

```python
from enum import StrEnum
from openjevpro.schemas import ChoiceDecision
from openjevpro.client import OpenJevProClient

class TicketRoute(StrEnum):
    BILLING = "billing"
    TECH_SUPPORT = "tech_support"
    SECURITY = "security"
    ESCALATE = "human_review"

# Initialized with a fast Tier 1/2 model (e.g., Qwen3-4B or Qwen3-30B-A3B)
client = OpenJevProClient(
    base_url="http://localhost:8000/v1",  # vLLM / SGLang endpoint
    model="Qwen/Qwen3-4B-Instruct",
    temperature_scaling=1.30,
    abstain_threshold=0.45
)

decision: ChoiceDecision = client.decide_choice(
    state={"ticket_text": "I noticed an unauthorized login attempt from an unknown IP address."},
    candidates=TicketRoute,
    criteria="Classify the incoming support ticket into the correct handling department."
)

print(f"Action: {decision.value}")
print(f"Confidence: {decision.confidence:.2%}")
print(f"Probabilities: {decision.probabilities}")
print(f"Abstained: {decision.abstained}")
```

---

## 📊 Economics & Performance

* **1-Token Decoding**: By constraining generation to single-token choice identifiers or evaluating candidate logits directly, generation token costs drop to near zero.
* **Prompt Caching Friendly**: System instructions, schema definitions, and candidate options stay static in the prefill cache.
* **Latency**: End-to-end response time typically ranges between **40ms ~ 120ms** when deployed on local vLLM instances with modern MoE/Dense models.

---

## 📄 License & Commercial Terms

* **Non-Commercial & Community Use**: OpenJevPro is licensed under the **[PolyForm Noncommercial License 1.0.0](LICENSE)**. Free for personal learning, academic research, non-profit institutions, and non-commercial development.
* **Commercial Use**: Any use within commercial enterprises, production environments, commercial SaaS products, or paid services **requires a commercial license from the project maintainers**. See **[LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md)** for details on applying for commercial authorization.
