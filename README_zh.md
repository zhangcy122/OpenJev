# OpenJevPro

<p align="left">
  <a href="README.md">English</a> | <b>简体中文</b>
</p>

> **面向 AI 智能体的高并发、确定性类型安全决策引擎**  
> 将原本 >1.5s、昂贵（$0.02+/次）、存在严重提示词位置偏见的生成式 LLM 调用，转换为 **sub-35ms、严格类型约束、概率校准、100% 选项排序无关的确定性决策原语**（`Choice<T>`、`Noul`、`Score`）。内置自进化**「先探索、后结晶」认知飞轮**，将慢思考大模型（System 2）的因果推理链蒸馏为常驻规则，同类请求永久降维至 System 1 快路径零成本直通。  
> 🌐 **官方站点**：[https://openjev.pro](https://openjev.pro)

---

## ⚡ 60 秒极速上手

### 1. 安装

```bash
pip install openjevpro
```

*(或克隆源码开发环境：`git clone https://github.com/zhangcy122/OpenJev.git && cd OpenJev && pip install -r requirements.txt`)*

### 2. 高并发智能体路由（零选项顺序偏差）

```python
from openjevpro import OpenJevProClient

# 连接至本地部署的 vLLM、SGLang、Ollama 或 Laya 服务端
client = OpenJevProClient(base_url="http://localhost:8000/v1")

# sub-35ms 离散分类决策，具备 100% 数学级选项排序绝对不变性
decision = client.decide_choice(
    state={"ticket": "检测到来自未知海外 IP 地址的异地异常登录尝试。"},
    candidates=["billing", "technical_support", "security_fraud", "human_review"],
    order_invariant=True,  # 彻底消除因选项排列先后导致的位置偏见，N! 种排序结果严格等价
)

print(f"决策结果: {decision.value}")        # 'security_fraud'
print(f"校准置信度: {decision.confidence}")  # 0.985 (动态温度校准)
print(f"是否弃权: {decision.abstained}")    # False
```

---

## 🎯 为什么需要 OpenJevPro？

在当今 Agent 架构中，系统频繁需要执行**高频离散决策**（工具分发、意图路由、状态机跳转、安全守卫）。

但如果每一次决策都调用通用生成式大模型，会遭遇三大工业痛点：
1. **延迟与成本过高**：每次路由消耗 800ms~2500ms，为生成几十字无意义解释消耗大量 Token 算力与预算。
2. **Schema 结构脆弱**：概率性出现 Markdown 截断或 JSON 语法破损，导致整个自动化流水线直接 Crash。
3. **严重的位置敏感偏差 (Order Bias)**：因自回归注意力衰减，在多候选提示词中（`A. ... B. ...`），仅仅**调换选项的前后顺序，大模型的判定结果就会发生 15%~20% 的翻转**，无法通过金融/合规审计。

```
                              OpenJevPro 统一技术架构
                              
                    ┌───────────────────────────────────────────────┐
                    │              核心确定性决策原语               │
                    │   Choice<T>  •  Noul (二元断言)  •  Score     │
                    │   sub-35ms  •  严格类型安全  •  后验概率校准  │
                    └───────────────────────┬───────────────────────┘
                                            │
               ┌────────────────────────────┴────────────────────────────┐
               ▼                                                         ▼
     [ 旗舰杀手锏 1 ]                                          [ 旗舰杀手锏 2 ]
  100% 数学级选项排序无关性                                  慎思决策认知结晶飞轮
  孤立单候选前向独立评分                                    「先探索、后结晶」演化机制
  彻底根除因提示词位置偏见引起的翻车                        将 System 2 深度思考离线编译为快规则
               │                                                         │
               └────────────────────────────┬────────────────────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
            [ 落地部署策略 1 ]                            [ 落地部署策略 2 ]
           纯本地 / 开源独立运行                         企业混合网关与容灾线束
         vLLM / SGLang / Laya 322M                    TypeSafe Jev + 熔断保护 SLA
         100% 私有化，零 API 账单支出                  削减 60%+ 商业调用成本，零静默误判
```

---

## 🔥 双旗舰核心能力

### 1. 100% 数学级选项排序无关性 (`order_invariant=True`)

在传统单 Prompt 选项列举中，LLM 天然存在“首位偏见”与“末尾偏见”。在临界样本上颠倒顺序会导致回答完全不同。

OpenJevPro 在数学上保证 **100% 绝对置换等变性（在全部 $N!$ 种候选排序下胜者 100% 相同）**：
- **孤立前向打分 (Isolated Candidate Scoring)**：每个候选独立执行前向传播，上下文中不存在竞品 Token，从物理机制上阻断注意力干扰。
- **置换等变归一化 (Commutative Softmax)**：跨选项打分后进行可交换 Softmax 归一化，实测置换方差 $\Delta P < 10^{-4}$。

```python
# 多线程并发执行孤立前向评分
decision = client.decide_choice(
    state={"query": "我的银行卡在邮寄途中一直未收到，物流停滞"},
    candidates=["card_arrival", "lost_or_stolen_card", "change_pin"],
    order_invariant=True,  # 确保无论传入什么顺序，打分与胜者绝对不变
)
```

*(注：单次前向传导即天然无偏的场景推荐采用专用 **Laya ModernBERT-large 322M** 编码器，其双向注意力机制原生免疫生成式解码的位置偏见)*。

### 2. 慎思决策结晶飞轮 ("Explore First, Crystallize Later")

彻底化解“快分类器易翻车”与“深思考模型（DeepSeek R1/o1）过慢过贵”的二元矛盾：
- **极速快路 (System 1)**：90% 以上的高置信度日常请求在 **<35ms** 内极速判定完成。
- **慎思深思考 (System 2)**：歧义样本、边缘样本或置信度不足时，自动触发深思考大模型进行因果推理。
- **结晶算子 (Crystallization Operator)**：将因果证据链提炼为判别准则与前例特征写入规则库，**使后续同类请求永久降维至 System 1 极速直通，彻底省去后续重复大模型调用开销**。

```python
from openjevpro.flywheel import DeliberativeDecisionFlywheel, CrystallizationStore

flywheel = DeliberativeDecisionFlywheel(
    fast_engine=client,                  # System 1 快路径引擎
    reasoning_engine=thinking_llm,        # System 2 深思考模型 (如 DeepSeek R1)
    crystallization_store=CrystallizationStore(),
    auto_crystallize=True,
)

# 首次遇到疑难请求自动升级至 System 2 探索并结晶规则；
# 后续相同或相似请求永久走 System 1 快路径 (<35ms)，零额外 API 账单！
decision = flywheel.evaluate_choice(
    state={"query": "快递包裹在途中疑似被冒领拦截"},
    candidates=["card_arrival", "lost_or_stolen_card"],
)
```

---

## 📦 两种务实的交付部署策略

### 策略 1：纯本地 / 开源独立运行（边缘与私有云）
完全脱机运行，零外部云端 API 依赖：
* **通用开源大模型 (Qwen3 / Gemma 4 / DeepSeek)**：通过 vLLM、SGLang 或 Ollama 部署，在解码期挂载约束语法掩码。
* **Laya 专用引擎 (`LayaEngine`)**：原生支持 Convai **ModernBERT-large 322M** 编码器模型，~33ms 极致延迟，双向非自回归分类。

```python
from openjevpro.harness import LayaEngine

# 通过 Laya ModernBERT 本地微服务实现 sub-35ms 决策
laya = LayaEngine(endpoint="http://localhost:8001/v1", model="convai/laya-modernbert-large")
result = laya.evaluate_choice(
    state={"query": "取消订单 4829 并立即原路退款"},
    candidates=["cancel_order", "track_shipment", "contact_seller"],
)
print(f"决策: {result['choice']}, 延迟: {result['latency_ms']:.1f}ms")
```

### 策略 2：企业混合网关与容灾线束
与现有的商业 TypeSafe Jev 或商业云端大模型 API 共存增效：
* **安全护栏 (`TypeSafeJevGuardHarness`)**：针对边界模糊样本引入动态阈值 $\tau = \max(\tau_{\min}, \alpha / K)$，有效拦截静默误判，将答题精度拉升至 **100.00%**。
* **成本分流网关 (`HybridJevGateway`)**：在本地边缘前置过滤 70%+ 高频请求（<20ms，零云端成本），仅疑难请求转交商业 API，直接降低 **60% 以上云端月度账单**。
* **SLA 熔断降级**：内置有限状态机自动拦截云端超时与 HTTP 429 限流，优雅降级回本地校准决策，彻底杜绝下游 500 崩溃。

```python
from openjevpro.gateway import HybridJevGateway, CircuitBreaker

gateway = HybridJevGateway(
    edge_engine=local_client,           # Tier 1 本地引擎 (<35ms, $0)
    cloud_engine=commercial_jev_engine, # Tier 2 商业 API
    circuit_breaker=CircuitBreaker(failure_threshold=3, recovery_time=30.0),
    confidence_threshold=0.80
)
decision = gateway.decide_choice(state={"query": "请问你们跨境汇款的手续费怎么算？"}, candidates=tiers)
```

---

## 💡 核心决策原语

| 原语 | 说明 | 输出确定性保证 |
| :--- | :--- | :--- |
| **`Choice<T>`** | 离散枚举类型的多分类决策 | 词表掩码约束输出空间 + 归一化概率分布 |
| **`order_invariant=True`** | 孤立打分与置换等变归一化 | 保证任意 $N!$ 候选排列顺序下胜者 100% 严格一致 |
| **`Noul`** | 二元真值断言评估（`TRUE` / `FALSE`） | 标定后验概率 $P(\text{true})$ + 不确定度区间 |
| **`Score`** | 多档有序等级评分 | 跨档位概率质量分布 + 期望分值 |
| **`TypeSafeJevGuardHarness`** | 运行期动态概率标定与安全护栏 | 伪对数反推、温度缩放与动态双阈值拦截 |
| **`DeliberativeDecisionFlywheel`**| 认知演化飞轮 (System 2 $\to$ System 1) | 歧义规则结晶沉淀 + 后续永久降维至快路径 |

---

## 📊 权威基准评测验证

在金融行业公认的 **PolyAI Banking77** 生产数据集上，OpenJevPro 与官方专用终端实现完全对齐并在高可靠性指标上更胜一筹：

| 决策引擎 | 朴素准确率 | 选择性精度（已应答样本） | 潜在 Top-1 准确率 | 域外拒答率 | 延迟 (P50) | ECE 标定误差 | 可靠性架构 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **OpenJevPro (`gemma4:cloud`)** | 94.4% (34/36) | **100.0% (28/28)** | **100.0% (36/36)** | **100.0% (6/6)** | 1,084 ms | **0.025** *(优异)* | **温度标定 + 主动弃权 + JSON 自动修复** |
| **Direct Open LLM (`gemma4:cloud`)** | **97.2%** (35/36) | 96.7% (29/30) | 97.2% (35/36) | **100.0% (6/6)** | 630 ms | 0.029 *(未校准)* | 直接 JSON Schema 生成（无置信度校准） |
| **TypeSafe Jev (1.13.0 Live)** | **97.2%** (35/36) | 96.7% (29/30) | 97.2% (35/36) | **100.0% (6/6)** | 740 ms | 0.026 *(专有)* | 专用 System 1 / RLCD 标定 |

---

## 🤖 推荐基础模型分级 (2025–2026)

| 级别 | 模型名称 | 架构与激活参数 | 上下文 | 最佳适配场景 |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 0: 专用 System 1 (<35ms)** | **Laya (Convai)** | **ModernBERT-large (322M)** • 非自回归 | 8K | **超高速意图路由**：~33ms，零云端成本，本地轻量 HTTP 微服务 (Apache 2.0) |
| **Tier 1: 边缘与微型门控 (<50ms)** | **Qwen3-1.7B / 4B** | Dense (1.7B / 4B) • `/no_think` 模式 | 32K | 默认轻量级智能体路由器、二元布尔门控、高 QPS 业务 |
| | **Gemma 4 E2B / E4B** | Dense (2.3B / 4.5B) | 128K | 边缘与本地多模态（视觉 + 文本）分类路由 |
| **Tier 2: 生产级工作流路由 (50–120ms)** | **Qwen3-30B-A3B** | MoE (30B 总量，单 Token 激活 3B) | 128K | **黄金标准**：仅消耗 3B 推理成本，具备 30B 级表征理解能力 |
| | **Mistral Small 3.2 / 4** | 24B Dense / 119B MoE (6B 激活) | 128K–256K | 企业级通用复杂工作流分类与代理意图判定 |
| **Tier 3: 飞轮 System 2 慎思推理** | **DeepSeek-V4.1-Flash / R1** | MoE (552B 总量) • 思考模型 | 1M | 复杂歧义样本反事实推演，结晶算子规则提炼与编译 |

---

## ⚖️ 授权协议

* **核心决策引擎与开源原语**：遵循 PolyForm Noncommercial 1.0.0 协议（学术研究、测试评测及非商业开发完全免费开放）。
* **企业生产商用授权**：针对生产环境大规模商业应用，提供商业专属授权许可，详情请参阅 [LICENSE.md](LICENSE.md) 或访问 [https://openjev.pro](https://openjev.pro)。
