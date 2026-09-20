# OpenJev

> **Open-Source Alternative to TypeSafe Jev**  
> Transform modern open-weight LLMs (Qwen3, DeepSeek-V3.2, Gemma 4, gpt-oss) into high-throughput, typed probabilistic decision services (System 1 Decisions).

---

## 💡 Overview

**OpenJev** is a lightweight, high-performance framework designed to replicate and extend the core capabilities of TypeSafe Jev using open-source Large Language Models. 

Instead of generating free-form, uncalibrated natural language strings, OpenJev provides non-autoregressive, strictly typed decision primitives (**`Choice<T>`**, **`Noul`**, **`Score`**) with mathematically calibrated posterior probabilities.

```
Incoming State & Questions
           │
           ▼
┌───────────────────────────────────────────────────────────┐
│                    OpenJev Engine                         │
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

## 🤖 Recommended Base Models (2025–2026 Tiers)

For constrained decision tasks (classification, intent detection, workflow routing, guardrails), **bigger is not always better**. OpenJev follows a tiered deployment strategy prioritizing non-thinking mode, compact active parameter sizes, and low latency:

| Tier | Model | Architecture & Active Params | Context | Recommended Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Edge & Ultra-Fast Gate (<50ms)** | **Qwen3-1.7B / 4B** | Dense (1.7B / 4B) • `/no_think` switch | 32K | Default lightweight router, binary gates, high-QPS routing |
| | **Gemma 4 E2B / E4B** | Dense (2.3B / 4.5B effective) | 128K | Edge & local multimodal (vision + text) classification |
| | **FunctionGemma (270M)** | Compact Dedicated | 32K | Ultra-light tool and function selection |
| **Tier 2: Production Workflow Routing (50–120ms)** | **Qwen3-30B-A3B** | MoE (30B total, 3B active / token) | 128K | **Golden Standard**: 3B inference cost with 30B representation capacity |
| | **Gemma 4 26B-A4B** | MoE (25.2B total, 3.8B active) | 256K | Server-grade fast decision router, native function calling |
| | **gpt-oss-20b** | MoE (21B total, 3.6B active) • MXFP4 | 128K | Single-GPU server deployment, Apache 2.0 open license |
| | **Mistral Small 3.2 / 4** | 24B Dense / 119B MoE (6B active) | 128K–256K | General enterprise workflow classification and agent routing |
| **Tier 3: Complex Arbitration & Fallback** | **DeepSeek-V3.2** | MoE (671B total, 37B active) • DSA | 128K | Hard-sample fallback, multi-step tool plan arbitration |
| | **Qwen3-Coder-30B-A3B** | MoE (30B total, 3B active) | 256K | Repo-level action routing, MCP tool selection |

> **Best Practice**: Run Tier 1/2 models with reasoning/thinking disabled (`/no_think`) for routine requests to achieve sub-100ms TTFT. Only escalate to Tier 3 (e.g. DeepSeek-V3.2) when `confidence < threshold` or when an explicit `HUMAN_REVIEW` / `UNKNOWN` signal is triggered.

---

## ⚙️ Architecture & Key Concepts

### 1. Type Safety via Constrained Decoding
Format and schema correctness are guaranteed at decode-time using formal grammar/regex constraints (via vLLM / SGLang structured outputs). Zero JSON parsing failures.

### 2. Candidate Sequence Likelihood
For a state $x$ and discrete candidate $y_i$, candidate scoring evaluates the conditional log-likelihood:
$$s_i = \sum_{t=1}^{|y_i|} \log P_\theta(y_{i,t} \mid x, y_{i,<t})$$

### 3. Confidence Calibration
Raw LLM token logprobs often exhibit severe **overconfidence**. OpenJev applies domain-level calibration:
$$p_i = \frac{\exp(s_i / T)}{\sum_{j} \exp(s_j / T)}$$
where temperature $T$ is fitted on offline validation benchmarks to minimize Expected Calibration Error (ECE).

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/zhangcy122/OpenJev.git
cd OpenJev
pip install -r requirements.txt
```

### Basic Usage

```python
from enum import StrEnum
from openjev.schemas import ChoiceDecision
from openjev.client import OpenJevClient

class TicketRoute(StrEnum):
    BILLING = "billing"
    TECH_SUPPORT = "tech_support"
    SECURITY = "security"
    ESCALATE = "human_review"

# Initialized with a fast Tier 1/2 model (e.g., Qwen3-4B or Qwen3-30B-A3B)
client = OpenJevClient(
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

## 📄 License

MIT License.
