# OpenJevPro

<p align="left">
  <a href="README.md">English</a> | <b>简体中文</b>
</p>

> **面向 AI 智能体的高并发、确定性类型安全决策引擎**  
> 针对意图路由、工具分发等离散决策场景，提供 35ms 以内延迟、严格类型约束与后验概率标定的决策原语（`Choice<T>`、`Noul`、`Score`）。支持选项排列无关性（消除提示词位置偏差），并内置双系统结晶机制：常规请求走快路径，边缘样本由推理模型分析并提炼为规则缓存，兼顾高吞吐与准确率。  
> 🌐 **官方站点**：[https://openjev.pro](https://openjev.pro)

---

## ⚡ 快速上手

### 1. 安装

```bash
pip install openjevpro
```

*(或克隆源码开发环境：`git clone https://github.com/zhangcy122/OpenJev.git && cd OpenJev && pip install -r requirements.txt`)*

### 2. 0 GPU 终端极速演算（5 秒快速体验）

无需配置显卡或启动本地推理服务，直接运行内置确定性演算演示（总耗时 <15ms）：

```bash
python3 -m openjevpro.demo
```

### 3. 智能体分类路由（选项顺序无关）

```python
from openjevpro import OpenJevProClient

# 生产部署：连接至本地 vLLM、SGLang、Ollama 或 Laya 服务端
client = OpenJevProClient(base_url="http://localhost:8000/v1")

# 本地无 GPU 开发与 CI 自动化测试：
# client = OpenJevProClient(mock=True)

# 离散分类决策，保证在不同候选顺序下判定一致
decision = client.decide_choice(
    state={"ticket": "检测到来自未知海外 IP 地址的异地异常登录尝试。"},
    candidates=["billing", "technical_support", "security_fraud", "human_review"],
    order_invariant=True,  # 开启选项顺序无关模式，消除提示词位置偏差
)

print(f"决策结果: {decision.value}")        # 'security_fraud'
print(f"校准置信度: {decision.confidence}")  # 0.985 (动态温度校准)
print(f"是否弃权: {decision.abstained}")    # False
```

---

## 🎯 为什么需要 OpenJevPro？

智能体系统在执行工具分发、意图路由或状态跳转等高频离散决策时，若直接调用通用生成式模型，容易遇到三个问题：

1. **延迟与算力开销高**：单次路由生成通常耗时 800ms 至 2500ms，为输出简短的标签消耗大量解码 Token。
2. **输出格式不稳定**：可能出现 Markdown 代码块截断或 JSON 语法解析失败，导致调用链异常中断。
3. **提示词位置偏差 (Order Bias)**：因自回归模型的注意力衰减，在多候选提示词中（如 `A. ... B. ...`），仅仅调换选项的前后顺序，模型的判定结果就可能发生 15%~20% 的变化，影响生产审计要求。

```
                              OpenJevPro 统一技术架构
                              
                    ┌───────────────────────────────────────────────┐
                    │              核心确定性决策原语               │
                    │   Choice<T>  •  Noul (二元断言)  •  Score     │
                    │   <35ms     •  严格类型安全  •  后验概率校准  │
                    └───────────────────────┬───────────────────────┘
                                            │
               ┌────────────────────────────┴────────────────────────────┐
               ▼                                                         ▼
       [ 核心特性 1 ]                                            [ 核心特性 2 ]
    候选选项排列无关性                                        双系统决策结晶机制
   孤立候选前向独立评分                                      探索与规则提炼回路
 避免因选项顺序变化导致判定翻转                              将慢推理逻辑提炼为快路径规则
               │                                                         │
               └────────────────────────────┬────────────────────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
             [ 部署方案 1 ]                                [ 部署方案 2 ]
          纯本地 / 开源独立运行                         企业混合网关与容灾降级
        vLLM / SGLang / Laya 322M                    TypeSafe Jev + 熔断保护 SLA
        完全私有化，无外部 API 依赖                   降低商业调用成本，拦截低置信度误判
```

---

## 核心特性

### 1. 选项排序无关性 (`order_invariant=True`)

在传统单次 Prompt 选项罗列中，LLM 容易受到首位或末尾位置的影响。在置信度接近的边缘样本上，选项顺序颠倒可能会改变判定结果。

OpenJevPro 通过两项机制保证在任意候选排列顺序下胜者判定一致：
- **孤立前向打分 (Isolated Candidate Scoring)**：每个候选独立执行前向计算，上下文互不干扰。
- **置换等变归一化 (Commutative Softmax)**：跨选项打分后进行可交换归一化，置换方差 $\Delta P < 10^{-4}$。

```python
decision = client.decide_choice(
    state={"query": "我的银行卡在邮寄途中一直未收到，物流停滞"},
    candidates=["card_arrival", "lost_or_stolen_card", "change_pin"],
    order_invariant=True,  # 确保传入任意排列顺序，最终判定均一致
)
```

*(注：若需单次前向无偏推理，可选用配套的 **Laya ModernBERT-large 322M** 编码器模型，其双向注意力机制不依赖自回归生成，天然避免位置偏差)*。

### 2. 双系统决策与规则结晶 ("Explore First, Crystallize Later")

兼顾高吞吐快路径与复杂样本深度推理：
- **快路径 (System 1)**：常规高置信度请求在 35ms 内完成判定。
- **深度推理 (System 2)**：歧义样本或置信度不足时，触发推理模型（如 DeepSeek R1）进行因果分析。
- **结晶算子 (Crystallization Operator)**：将推理结果提炼为判定准则写入规则库，使后续同类请求直接走快路径，降低平均延迟与调用成本。

```python
from openjevpro.flywheel import DeliberativeDecisionFlywheel, CrystallizationStore

flywheel = DeliberativeDecisionFlywheel(
    fast_engine=client,                  # 快路径引擎 (System 1)
    reasoning_engine=thinking_llm,        # 深度推理模型 (System 2)
    crystallization_store=CrystallizationStore(),
    auto_crystallize=True,
)

# 首次遇到疑难请求交由推理模型分析并记录规则；
# 后续相似请求直接由快路径模型处理 (<35ms)。
decision = flywheel.evaluate_choice(
    state={"query": "快递包裹在途中疑似被冒领拦截"},
    candidates=["card_arrival", "lost_or_stolen_card"],
)
```

---

## 交付部署方案

### 方案 1：纯本地 / 开源独立运行（边缘与私有云）
完全离线运行，不依赖外部云端接口：
* **通用开源大模型 (Qwen3 / Gemma 4 / DeepSeek)**：通过 vLLM、SGLang 或 Ollama 部署，在解码阶段添加词表掩码约束。
* **Laya 专用引擎 (`LayaEngine`)**：支持 Convai **ModernBERT-large 322M** 编码器模型，约 33ms 延迟，双向非自回归分类。

```python
from openjevpro.harness import LayaEngine

# 通过 Laya ModernBERT 本地微服务实现低延迟决策
laya = LayaEngine(endpoint="http://localhost:8001/v1", model="convai/laya-modernbert-large")
result = laya.evaluate_choice(
    state={"query": "取消订单 4829 并立即原路退款"},
    candidates=["cancel_order", "track_shipment", "contact_seller"],
)
print(f"决策: {result['choice']}, 延迟: {result['latency_ms']:.1f}ms")
```

### 方案 2：企业混合网关与容灾降级
与现有的商业 TypeSafe Jev 或商业云端 API 协同工作：
* **安全护栏 (`TypeSafeJevGuardHarness`)**：针对边界模糊样本使用动态阈值 $\tau = \max(\tau_{\min}, \alpha / K)$，拦截低置信度误判。
* **成本分流网关 (`HybridJevGateway`)**：在本地边缘前置处理约 70% 高频请求（<20ms），仅疑难请求转交商业 API，降低云端调用成本。
* **SLA 熔断降级**：云端超时或触发 429 限流时自动熔断，降级回本地模型，保障下游服务可用性。

```python
from openjevpro.gateway import HybridJevGateway, CircuitBreaker

gateway = HybridJevGateway(
    edge_engine=local_client,           # 本地引擎 (<35ms)
    cloud_engine=commercial_jev_engine, # 商业 API
    circuit_breaker=CircuitBreaker(failure_threshold=3, recovery_time=30.0),
    confidence_threshold=0.80
)
decision = gateway.decide_choice(state={"query": "请问你们跨境汇款的手续费怎么算？"}, candidates=tiers)
```

---

## 核心决策原语

| 原语 | 说明 | 输出确定性保证 |
| :--- | :--- | :--- |
| **`Choice<T>`** | 离散枚举类型的多分类决策 | 词表掩码约束输出空间 + 归一化概率分布 |
| **`order_invariant=True`** | 孤立打分与置换等变归一化 | 保证任意 $N!$ 候选排列顺序下胜者判定一致 |
| **`Noul`** | 二元真值断言评估（`TRUE` / `FALSE`） | 标定后验概率 $P(\text{true})$ + 不确定度区间 |
| **`Score`** | 多档有序等级评分 | 跨档位概率质量分布 + 期望分值 |
| **`TypeSafeJevGuardHarness`** | 运行期动态概率标定与安全护栏 | 伪对数反推、温度缩放与动态双阈值拦截 |
| **`DeliberativeDecisionFlywheel`**| 双系统决策回路 (System 2 $\to$ System 1) | 疑难案例规则提炼并缓存，后续请求走快路径 |
| **`MockClient`** (`mock=True`) | 零 GPU 确定性离线仿真客户端 | 零网络与零显卡依赖，用于本地开发、极速演示与 CI 单元测试 |

---

## 📊 基准评测结果

在 Banking77 数据集（36 条测试样本，含 6 条域外样本）上的对比实测数据：

| 决策引擎 | 朴素准确率 | 选择性精度（已应答样本） | 潜在 Top-1 准确率 | 域外拒答率 | 延迟 (P50) | ECE 标定误差 | 可靠性机制 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **OpenJevPro (`gemma4:cloud`)** | 94.4% (34/36) | **100.0% (28/28)** | **100.0% (36/36)** | **100.0% (6/6)** | 1,084 ms | **0.025** | 温度标定 + 主动弃权 + JSON 自动修复 |
| **Direct Open LLM (`gemma4:cloud`)** | **97.2%** (35/36) | 96.7% (29/30) | 97.2% (35/36) | **100.0% (6/6)** | 630 ms | 0.029 | 直接 JSON Schema 生成（无置信度标定） |
| **TypeSafe Jev (1.13.0 Live)** | **97.2%** (35/36) | 96.7% (29/30) | 97.2% (35/36) | **100.0% (6/6)** | 740 ms | 0.026 | 专用小模型 / RLCD 标定 |

---

## 🤖 推荐模型分级 (2025–2026)

| 级别 | 模型名称 | 架构与激活参数 | 上下文 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 0: 专用快速模型 (<35ms)** | **Laya (Convai)** | **ModernBERT-large (322M)** • 非自回归 | 8K | 高频意图路由：~33ms 延迟，本地轻量 HTTP 服务 (Apache 2.0) |
| **Tier 1: 边缘与轻量门控 (<50ms)** | **Qwen3-1.7B / 4B** | Dense (1.7B / 4B) • `/no_think` 模式 | 32K | 轻量级路由、二元断言门控、高并发接口 |
| | **Gemma 4 E2B / E4B** | Dense (2.3B / 4.5B) | 128K | 边缘与本地多模态（视觉 + 文本）分类路由 |
| **Tier 2: 通用工作流路由 (50–120ms)** | **Qwen3-30B-A3B** | MoE (30B 总量，单 Token 激活 3B) | 128K | 综合平衡：激活 3B 参数，具备较好的长文本与意图理解能力 |
| | **Mistral Small 3.2 / 4** | 24B Dense / 119B MoE (6B 激活) | 128K–256K | 通用复杂工作流分类与代理意图判定 |
| **Tier 3: 深度推理模型** | **DeepSeek-V4.1-Flash / R1** | MoE (552B 总量) • 思考模型 | 1M | 歧义与边缘样本因果分析，生成判定准则并写入规则库 |

---

## ⚖️ 授权协议

* **核心决策引擎与开源原语**：遵循 PolyForm Noncommercial 1.0.0 协议（学术研究、测试评测及非商业开发完全免费开放）。
* **企业生产商用授权**：针对生产环境大规模商业应用，提供商业专属授权许可，详情请参阅 [LICENSE.md](LICENSE.md) 或访问 [https://openjev.pro](https://openjev.pro)。
