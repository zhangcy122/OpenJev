# OpenJev

> **Open-Source Alternative to TypeSafe Jev**  
> Transform open-source LLMs (Qwen 2.5, DeepSeek, Llama 3) into high-throughput, typed probabilistic decision services (System 1 Decisions).

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

client = OpenJevClient(
    base_url="http://localhost:8000/v1",  # vLLM or OpenAI-compatible API
    model="Qwen/Qwen2.5-7B-Instruct",
    temperature_scaling=1.35
)

decision: ChoiceDecision = client.decide_choice(
    state={"ticket_text": "I noticed an unauthorized login attempt from an unknown IP."},
    candidates=TicketRoute,
    criteria="Classify the incoming support ticket into the correct handling team."
)

print(f"Action: {decision.value}")
print(f"Confidence: {decision.confidence:.2%}")
print(f"Probabilities: {decision.probabilities}")
print(f"Abstained: {decision.abstained}")
```

---

## 📊 Economics & Performance

* **1-Token Decoding**: By constraining generation to single-token choice identifiers or evaluating logprobs directly, generation token costs drop to near zero.
* **Prompt Caching Friendly**: System instructions, schema definitions, and rules stay static in the prefill cache.
* **Latency**: End-to-end response time typically ranges between **50ms ~ 150ms** when deployed on local vLLM instances.

---

## 📄 License

MIT License.
