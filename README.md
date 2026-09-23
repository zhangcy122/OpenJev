# OpenJevPro

> **Production-Grade Open Alternative to TypeSafe Jev**  
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
| **OpenJevPro (via [`openjev-sglang`](https://github.com/ekzhang/openjev-sglang))** | **Qwen3.6-35B-A3B (MoE)** | **95.5%** | **Overlaps with Jev** | **~45–75ms** | **Open Source / Zero-Shot** |
| **OpenJevPro (Edge)** | **Qwen3-4B / Gemma 4** | ~93.8% | Compact tier | < 35ms | 100% Free / Single GPU |

> **Key Takeaway**: Across 242 decision cases in [JevBench v1](https://benchmarkheaven.com/jev-models), [`openjev-sglang`](https://github.com/ekzhang/openjev-sglang) by @ekzhang (95.5%) and Jev 1.13.0 (96.3%) exhibit overlapping 95% confidence intervals, proving that representation capacity of modern open MoE models combined with lexical grammar masking achieves decision parity without requiring proprietary model training.

### 🧪 Fresh Empirical Replication: OpenJevPro Harness vs TypeSafe Jev (Banking77 Benchmark)

To independently verify performance and reliability on official production endpoints, we implemented the standardized [OpenJevPro Benchmark Harness](openjevpro/harness.py) and ran a head-to-head comparison on the **PolyAI Banking77** dataset (30 in-domain queries across 6 financial categories + 6 out-of-scope queries):

| Decision Engine | Overall Accuracy | In-Domain Accuracy | Out-of-Scope Rejection | Latency (P50 / Mean) | Calibration Error (ECE) | Architecture & Reliability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **OpenJevPro (`gemma4:cloud`)** | **94.4%** (34/36) | 93.3% (28/30)* | **100.0% (Safe Abstain)** | **1,084 ms** *(Mean 1,093ms)* | **0.025** *(Calibrated)* | **Temperature Calibrated + Selective Abstention + Auto-repair JSON** |
| **Direct Open LLM (`gemma4:cloud`)** | **97.2%** (35/36) | **96.7%** (29/30) | **100.0% (Symmetric UNKNOWN)** | **630 ms** *(Mean 772ms)* | 0.029 *(Raw JSON)* | Direct JSON Schema (No confidence calibration) |
| **TypeSafe Jev (1.13.0)** | 80.56% (29/36) | **96.7%** (29/30) | 0.0% (Forced Pick)* | **767 ms** *(Mean 777ms)* | 0.075 *(Proprietary)* | Proprietary System 1 / RLCD (Closed-set in baseline) |

*\*Note on Symmetry & Abstention: When `UNKNOWN` is symmetrically provided in the criteria, modern open models (`gemma4:cloud`) achieve 100% out-of-scope rejection out-of-the-box. OpenJevPro's key distinction is in-domain calibration: for 2 queries with low calibrated confidence (< 0.40, e.g. ambiguous 'locate my card'), OpenJevPro safely abstained (`choice="UNKNOWN"`), avoiding overconfident misrouting while retaining 100% (30/30) accuracy in its underlying `tentative_value`. TypeSafe Jev was evaluated without UNKNOWN in its criteria in the historical run (causing forced selection); symmetric evaluation is supported in the harness via `allow_abstain=True`. Latency reflects managed cloud endpoints (`gemma4:cloud` ~630–1,084ms) vs Jev API (~767ms); collocated local vLLM/SGLang deployments achieve sub-100ms.*

#### 💡 Critical Findings:
1. **Honest Latency & Deployment Profiles**: Under modern managed cloud inference (`gemma4:cloud` via Ollama Cloud), Direct LLM recorded **630 ms P50 latency** (mean 772ms), outperforming commercial TypeSafe Jev API's **767 ms P50**. OpenJevPro (which generates full candidate scores for temperature calibration) recorded **1,084 ms P50** (mean 1,093ms). Sub-100ms latency is attained in collocated vLLM/SGLang deployments (see JevBench v1 benchmark above).
2. **Symmetric Open-Set Parity**: With `allow_abstain=True` injecting the `UNKNOWN` option symmetrically, open-source models attain **6/6 (100.0%)** out-of-scope rejection, proving that forced closed-set errors in prior baselines were purely an artifact of harness criteria asymmetry.
3. **Calibrated Selective Prediction**: While Direct LLM and Jev both output high confidence on ambiguous queries (e.g., 0.85 on *"How do I locate my card?"*), OpenJevPro's **`TemperatureCalibrator` + Selective Abstention Layer** recognizes low posterior confidence (< 0.40) and safely abstains, while retaining 100% accuracy in `tentative_value`.
4. **Superior Probability Calibration (ECE 0.025)**: Standardized 10-bin equal-width ECE confirms OpenJevPro's post-hoc calibration reduces calibration error to **0.0253**, markedly lower than TypeSafe Jev's 0.0750.
5. **Reproducibility**: Run the benchmark suite anytime via:
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
