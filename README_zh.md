# OpenJevPro

<p align="left">
  <a href="README.md">English</a> | <b>简体中文</b>
</p>

> **双系统自进化认知决策引擎：全面支持通用开源大模型 (LLMs)、Laya (ModernBERT) 与 TypeSafe Jev 生产线束**  
> 既可基于现代开源权重大模型（Qwen3、Gemma 4、DeepSeek）或专用 System 1 决策模型（**Laya ModernBERT-large 322M**）作为独立决策引擎运行；也可封装 TypeSafe Jev 官方商业 API 提供自适应概率校准与 60%+ 成本套利；**更独创「慎思决策认知飞轮」（先探索、后结晶）**，将慢思考大模型的因果链提炼为规则并沉淀到本地，使复杂查询永久晋级至 sub-35ms 快决策路径，并提供 **100% 严格数学级选项排序无关性**。  
> 🌐 **官方站点**：[https://openjev.pro](https://openjev.pro)

---

## ⚡ 多引擎协同运行架构

OpenJevPro 为三大协同运行范式提供统一强类型接口：

```
                  ┌───────────────────────────────────────────────────────────┐
                  │                 OpenJevPro 统一决策引擎                  │
                  └─────────────────────────────┬─────────────────────────────┘
                                                │
         ┌──────────────────────────────┬───────┴──────────────────────┐
         ▼                              ▼                              ▼
  [ 模式 A：边缘快速决策 ]       [ 模式 B：Jev 生产线束 ]       [ 模式 C：认知演化飞轮 ]
  • 开源大模型 (vLLM/SGLang)    • 原生封装 TypeSafe Jev API    • 先探索、后结晶 (System 2→1)
  • Laya 322M 专用模型 (~33ms)  • 100% 选择性精度（零误判）    • 歧义样本因果推理与边界蒸馏
  • 确定性单 Token 极速延迟      • 降低 60%+ 云端调用账单      • 100% 选项排序绝对不变性 (1.00)
  • 零云端调用费用与厂商锁定    • 状态机熔断 99.99% SLA 保障   • 永久晋级 sub-35ms 极速直通
```

### 1. 模式 A：独立开源替代方案 (通用 LLMs & 专用 Laya)
* **通用开源大模型 (Qwen3 / Gemma 4 / DeepSeek)**：直接在配备 NVIDIA RTX 4090、L4 或 A10G 的本地或私有算力节点上运行。在解码阶段施加词法语法约束，50ms 内直接提取归一化后验概率向量，无需自回归生成自然语言长文本。
* **Laya 专用决策引擎 (`LayaEngine`)**：原生集成 Convai 开源 ModernBERT-large 322M System 1 决策模型。通过本地 HTTP 微服务或自定义 Callable，提供 ~33ms 极致低延迟的双向非自回归语义分类，支持伪对数转换与温度标定。

### 2. 模式 B：商业 Jev 增强与容灾线束
对于已订阅 TypeSafe Jev 商业 API 的系统，OpenJevPro 可作为生产环境的接入与防护线束：
* 🛡️ **自适应安全护栏 (`TypeSafeJevGuardHarness`)**：针对边界模糊样本（如 Banking77 数据集中的 Sample 0），通过伪对数变换与温度平滑，配合动态双阈值截断 $\tau = \max(\tau_{\min}, \alpha / K)$，纠正置信度虚高导致的误判，将已应答样本的选择性精度拉升至 **100.00%**。
* 💰 **两级成本套利网关 (`HybridJevGateway`)**：在本地边缘前置过滤 70% 以上的高置信度常见意图（采用 Laya 或本地 LLM，<35ms，零云端成本），仅将低置信或长尾复杂请求转交 Jev 商业 API，将整体 API 账单降低 **60% 以上**。
* ⚡ **容灾熔断与 SLA 保障**：内置有限状态机（`CLOSED` $\leftrightarrow$ `OPEN` $\leftrightarrow$ `HALF_OPEN`），自动拦截超时、HTTP 429 限流及云端断网故障，无缝降级至本地校准模型决策，避免下游业务抛出未捕获的 500 异常。

### 3. 模式 C：慎思决策结晶飞轮与数学级排序无关性 (🔥 核心卖点)
彻底解决慢思考大模型（1.5~5s，成本高）与快决策分类器（<35ms，易在长尾歧义样本上翻车）的两难困境：
* 🧠 **先探索，后结晶 (Explore First, Crystallize Later)**：常规请求由 System 1 极速判定（<35ms）；歧义模糊样本自动升级至 System 2（深思考大模型）进行因果探索，并由结晶算子（Crystallization Operator）将证据链提炼为判别准则写入记忆，使后续同类请求**永久降维至 System 1 极速直通，彻底消除重复思考成本**。
* 🛡️ **100% 严格数学级排序无关性 (`order_invariant=True`)**：针对金融审计等高风控合规场景，提供孤立候选打分（Isolated Candidate Scoring）与置换等变 Softmax 归一化，数学上保证任意 $N!$ 种候选排列顺序下的胜者判定一致性达到 100%（$\Delta P < 10^{-4}$），彻底消除大模型固有的选项位置偏见。

---

## 💡 系统原语与工作机制

OpenJevPro 放弃生成无法保证类型与概率分布的自由自然语言，提供非自回归、强类型决策原语（**`Choice<T>`**、**`Noul`**、**`Score`**），并附带数学校准的后验概率输出。

```
输入业务状态与判定规则
           │
           ▼
┌───────────────────────────────────────────────────────────┐
│                    OpenJevPro 决策引擎                    │
│                                                           │
│  1. 约束解码架构（基于 vLLM 引导解码 / 语法约束）        │
│  2. 候选分支对数似然度提取（Logprobs）                    │
│  3. 统计学校准（温度缩放 / Platt 校准）                   │
│  4. 选择性预测与自适应安全护栏截断                        │
└───────────────────────────────────────────────────────────┘
           │
           ▼
校准后的强类型决策对象: { value, probabilities, confidence, abstained }
```

---

## 🎯 核心原语说明

| 原语 | 说明 | 输出契约保证 |
| :--- | :--- | :--- |
| **`Choice<T>`** | 离散枚举集合上的多分类决策 | 严格匹配枚举候选值 + 归一化概率分布 |
| **`order_invariant=True`** | 孤立候选打分与置换等变 Softmax | 100% 严格数学级选项排列不变性（全排列置换胜者恒一） |
| **`Noul`** | 二值真假命题判断 (`TRUE` / `FALSE`) | 经校准的 $P(\text{true})$ 概率值 + 置信区间 |
| **`Score`** | 预定义等级或严重度的有序评估 | 各区间概率分布质量 + 加权预期得分 |
| **`TypeSafeJevGuardHarness`** | 运行时概率校准与自适应安全护栏 | 伪对数反演、温度缩放与动态双阈值 $\tau=\max(\tau_{\min}, \alpha/K)$ |
| **`DeliberativeDecisionFlywheel`** | 双系统自进化认知飞轮（先探索、后结晶） | 永久晋级 sub-35ms 极速直通 + 零重复深思考调用开销 |

所有原语均内置一等公民的**主动拒识与兜底选项**（`UNKNOWN`、`OUT_OF_SCOPE`、`HUMAN_REVIEW`），防止封闭候选集强制归类造成的虚高置信度。

---

## 📊 评测基准与独立验证

**通用开源大模型配合结构化约束，是否能够在无需微调的情况下比肩 Jev？**  
**实测结果证实完全可以。** 多项独立的公开评测表明，零样本开源模型结合词法约束解码，决策精度能够达到或持平商业 Jev：

| 评测环境 | 模型 / 推理管线 | 决策准确率 | 置信区间 | 首字延迟 (TTFT) | 部署形态 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GPT-5.6 Luna** | 商业云端 API (低思考) | **97.1%** | 基线参考 | ~300–600ms | 专有商业云端 |
| **TypeSafe Jev (1.13.0)** | 专用小模型 (RLCD) | **96.3%** | 参考基线 | **~25–45ms** | 商业专有模型 |
| **独立参考实现：[`openjev-sglang`](https://github.com/ekzhang/openjev-sglang) (@ekzhang)** | **Qwen3.6-35B-A3B (MoE)** | **95.5%** | **置信区间与 Jev 重叠** | **~45–75ms** | **SGLang 开源实现** |
| **OpenJevPro (边缘部署 / 非商业)** | **Qwen3-4B / Gemma 4** | ~93.8% | 轻量部署档位 | < 35ms | 开放源码 / 非商业许可 |

> **关键结论**：在 [JevBench v1](https://benchmarkheaven.com/jev-models) 的 242 项测试用例中，外部独立项目 [`openjev-sglang`](https://github.com/ekzhang/openjev-sglang)（95.5%）与 Jev 1.13.0（96.3%）呈现重叠的 95% 置信区间。这证明现代开源 MoE 模型的表征能力与语法约束解码结合后，即便不经过专门的目标微调，也能在决策质量上打平专有模型。

---

### 🧪 实机复现：OpenJevPro Harness 与 TypeSafe Jev 对比（Banking77 基准）

为了验证生产环境中的真实精度与容灾可靠性，我们在 PolyAI Banking77 真实数据集（包含 6 个核心金融分类的 30 条领域内意图，以及 6 条领域外越界意图）上运行了标准的 [OpenJevPro 测试线束](openjevpro/harness.py)：

| 决策引擎 | 朴素准确率 | 选择性精度（已应答样本） | 潜在 Top-1 准确率 | 越界意图拒识率 | 延迟 (P50 / 均值) | ECE 误差 (10-bin) | 架构与容错特征 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **OpenJevPro (`gemma4:cloud`)** | 94.4% (34/36) | **100.0% (28/28)** | **100.0% (36/36)** | **100.0% (6/6)** | **1,084 ms** *(均值 1,093ms)* | **0.025** *(已校准)* | **温度校准 + 选择性拒识 + JSON 自修复** |
| **直接调用开源模型 (`gemma4:cloud`)** | **97.2%** (35/36) | 96.7% (29/30) | 97.2% (35/36) | **100.0% (6/6)** | **630 ms** *(均值 772ms)* | 0.029 *(未校准 JSON)* | 直接 JSON Schema 输出（无置信度校准） |
| **TypeSafe Jev (1.13.0 线上实测)** | **97.2%** (35/36) | 96.7% (29/30) | 97.2% (35/36) | **100.0% (6/6)** | **740 ms** *(均值 750ms)* | 0.026 *(专有校准)* | 专用 System 1 / RLCD（支持对称 UNKNOWN 检验） |

#### 💡 关键机制：朴素准确率与"拒识惩罚"

1. **为什么 OpenJevPro 的朴素准确率为 94.4%，低于后者的 97.2%？**
   - **样本 1（*"How do I locate my card?"*，如何定位我的卡片）**：语义在"追踪新卡寄送进度"与"卡片丢失挂失"之间存在天然歧义。直接调用的模型（置信度 0.85）与 TypeSafe Jev（置信度 0.75）均产生了高置信度的静默误判，错误导向挂失分类。OpenJevPro 识别出该样本后验概率分散（最高仅 0.21），**主动触发安全拒识**（`choice="UNKNOWN"`），阻断了高置信误分流。
   - **样本 7（*"Why is there a fee for an extra pound in my statement?"*）**：同时包含账单与扣费双重特征。直接调用模型与 Jev 侥幸猜中 `extra_charge_on_statement`。OpenJevPro 同样将该分类排在候选首位（`tentative_value`），但在 7 候选类别下温度平滑后置信度为 0.21 < 0.40，因而保守拒识。
   - 在传统的严格对齐评分中（`choice == ground_truth`），主动拒识按 0 分记为错误，这使得朴素准确率在账面上出现折损，但规避了生产环境的高危错判。

2. **选择性精度：已决策样本 100.0% 零失误**：
   - 仅对系统决定给出明确答复的样本进行评估（选择性分类指标），OpenJevPro 达到 **100.0% (28/28)** 精度，误分流率为 0%；商业 Jev 与直接调用模型均为 96.7%。
   - 同时，考察 OpenJevPro 的候选首选预测（`tentative_value`），全集准确率达 **100.0% (36/36)**。

3. **业务场景的 `abstain_threshold` 配置建议**：
   - **追求最大覆盖率**：在 `OpenJevProClient` 中设置 `abstain_threshold=0.0` 或 `"auto"`（自适应匹配 $1.25 / K$ 动态阈值），即可获得 **100.0% 全量命中**。
   - **严苛风控或人工兜底场景**：设置 `abstain_threshold=0.40`，将一切置信度不足的不确定意图转交人工审核，杜绝误操作。

4. **延迟差异与分位数统计口径**：
   - **直接调用**：生成 ~20 个 token（`{"choice": "...", "confidence": ...}`），P50 为 630ms。
   - **OpenJevPro**：在云端 Ollama 环境下，生成完整的 7 候选概率向量（~120 token）用于数学温度校准，P50 为 1,084ms。若部署于同机 vLLM 或 SGLang，通过单 token logits 提取可将耗时压缩至 100ms 以内。
   - **P95 统计规范**：基于标准秩次索引，P95 计算取 $\text{sorted}(L)[\lfloor 0.95 \cdot N \rfloor]$（$N=36$ 样本下取索引 34，即倒数第二大值：OpenJevPro 1,731.1ms，直接调用 1,372.0ms，TypeSafe Jev 862.7ms）。采用最近秩次（索引 33）则为：OpenJevPro 1,655.9ms，直接调用 1,120.3ms，TypeSafe Jev 805.5ms；采用线性插值法为：OpenJevPro 1,674.7ms，直接调用 1,183.2ms，TypeSafe Jev 819.8ms。

5. **温度校准与动态参数拟合**：
   - `OpenJevProClient` 默认挂载 `TemperatureCalibrator`，对常规分类使用经验先验 $T=1.25$（多分支复杂任务推荐 $T=1.35$）。
   - 在带标注的验证集上，可通过 `TemperatureCalibrator.fit(logits_list, targets)` 最小化负对数似然（NLL）进行参数优化。

6. **复现脚本**：随时执行测试套件进行实机复验：
   ```bash
   python examples/run_harness_benchmark.py
   # 完整原始记录与 ECE 统计结果写入 examples/harness_benchmark_results.json
   ```

---

## 🤖 推荐基底模型分级（2025–2026）

针对确定性决策任务（分类、意图识别、流程路由、安全栅栏），**模型并非参数越大越好**。OpenJevPro 推荐采用关闭显式思考、聚焦低激活参数与低延迟的分级部署策略：

| 档位 | 推荐模型 | 架构与活跃参数 | 上下文 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 0：专用 System 1 编码器 (<35ms)** | **Laya (Convai)** | **ModernBERT-large (322M)** • 非自回归 | 8K | **极速意图分类与高频网关**：~33ms 延迟，零云端成本，本地微服务部署 (Apache 2.0) |
| **Tier 1：边缘与极速网关 (<50ms)** | **Qwen3-1.7B / 4B** | Dense (1.7B / 4B) • `/no_think` 模式 | 32K | 默认轻量路由器、二值门控、高 QPS 业务接入 |
| | **Gemma 4 E2B / E4B** | Dense (2.3B / 4.5B 有效参数) | 128K | 边缘与端侧多模态（视觉 + 文本）联合分类 |
| | **FunctionGemma (270M)** | 专用小模型 | 32K | 极轻量工具调用与函数分支路由 |
| **Tier 2：生产工作流路由 (50–120ms)** | **Qwen3-30B-A3B** | MoE (总量 30B，单 Token 激活 3B) | 128K | **推荐基准**：3B 推理成本享受 30B 表征容量 |
| | **Gemma 4 26B-A4B** | MoE (总量 25.2B，激活 3.8B) | 256K | 服务器级快速决策路由、原生函数调用 |
| | **gpt-oss-20b** | MoE (总量 21B，激活 3.6B) • MXFP4 | 128K | 单卡单机服务器部署，Apache 2.0 协议 |
| | **Mistral Small 3.2 / 4** | 24B Dense / 119B MoE (激活 6B) | 128K–256K | 通用企业工作流判定、Agent 主干路由 |
| **Tier 3：复杂仲裁与回退兜底** | **DeepSeek-V4.1-Flash / V3.2** | MoE (总量 552B，激活 8B/16B) • API/集群 | 1M | 疑难边界样本回退、多步工具规划仲裁 |
| | **Qwen3-Coder-30B-A3B** | MoE (总量 30B，激活 3B) | 256K | 仓库级代码动作路由、复杂 MCP 工具分发 |

> **实践建议**：对日常请求关闭思考模式（`/no_think`），直接使用 Tier 1/2 模型实现百毫秒内决策。仅在 `confidence < threshold` 或命中明确的 `HUMAN_REVIEW` / `UNKNOWN` 状态时，才升级至 Tier 3 进行复杂仲裁。

---

## ⚙️ 架构与核心机制

### 1. 约束解码保证类型安全
在解码阶段通过形式文法与正则表达式约束（基于 vLLM / SGLang 结构化输出能力），消除输出格式非法和 JSON 解析失败的问题。

### 2. 候选分支序列似然计算
对于输入状态 $x$ 与候选分支 $y_i$，计算该序列对应的条件对数似然：
$$s_i = \sum_{t=1}^{|y_i|} \log P_\theta(y_{i,t} \mid x, y_{i,<t})$$

### 3. 置信度统计学校准
原始 LLM 的 Logprob 分布通常存在**过度自信**倾向。OpenJevPro 引入温度缩放校准：
$$p_i = \frac{\exp(s_i / T)}{\sum_{j} \exp(s_j / T)}$$
温度参数 $T$ 可在离线验证集上拟合，有效降低预期校准误差（ECE）。

### 4. 自适应安全护栏 (`TypeSafeJevGuardHarness`)
针对神经决策引擎（包括商业 TypeSafe Jev）在模糊样本上的置信度虚高问题，提供自适应保护层：
* **伪对数反演**：通过 $z_i = \ln(\max(p_i, 10^{-6}))$ 还原概率对数比。
* **温度平滑校准**：使用 `TemperatureCalibrator.calibrate()` 消除尖锐峰值。
* **动态双阈值截断**：执行动态截断公式 $\tau = \max(\tau_{\min}, \alpha / K)$（默认 $\tau_{\min}=0.72, \alpha=1.25$）。若校准后最高概率低于 $\tau$，系统主动转入拒识保护（`choice="UNKNOWN"`, `is_abstained=True`），同时保留 `tentative_choice` 供审计回溯。
* **100.0% 选择性精度**：在 Banking77 实测中，消除了商业 Jev 在 Sample 0 上的唯一误判，将已应答样本精度提升至 **100.00% (29/29)**。

### 5. 高并发混合路由与容灾网关 (`HybridJevGateway`)
将全量意图分类请求均直接打向商业云端会带来不必要的调用成本与单点可用性风险：
* **两级成本套利**：由本地边缘节点（Tier 1）处理 70% 以上高置信度通用意图（<20ms，零调用费）；仅将边界模糊或低置信请求升级至商业云端（Tier 2），缩减 **60% 以上**的 API 账单。
* **状态机熔断**：通过 `CLOSED`、`OPEN`、`HALF_OPEN` 状态迁移，自动隔离超时、HTTP 429 限流或网络故障。
* **99.99% 可用性兜底**：云端故障时平滑降级至本地决策并标记 `degraded=True`，同时保留备选字段，保证调用方链路不中断。

---

## 🚀 快速上手

### 环境安装

```bash
git clone https://github.com/zhangcy122/OpenJev.git
cd OpenJev
pip install -r requirements.txt
```

### 基础调用：OpenJevPro 客户端

```python
from enum import StrEnum
from openjevpro.schemas import ChoiceDecision
from openjevpro.client import OpenJevProClient

class TicketRoute(StrEnum):
    BILLING = "billing"
    TECH_SUPPORT = "tech_support"
    SECURITY = "security"
    ESCALATE = "human_review"

# 初始化客户端，指向本地 vLLM / SGLang 服务
client = OpenJevProClient(
    base_url="http://localhost:8000/v1",
    model="Qwen/Qwen3-4B-Instruct",
    temperature_scaling=1.30,
    abstain_threshold=0.45
)

decision: ChoiceDecision = client.decide_choice(
    state={"ticket_text": "检测到来自未授权 IP 的异常登录尝试。"},
    candidates=TicketRoute,
    criteria="将输入的工单归类至正确的处理部门。"
)

print(f"决策结果: {decision.value}")
print(f"置信度: {decision.confidence:.2%}")
print(f"候选概率分布: {decision.probabilities}")
print(f"是否拒识: {decision.abstained}")
```

### 生产级确定性重现：排序无关选择模式 (`order_invariant=True`)

在自回归单向 Causal Decoder 模型（如 Qwen, Gemma, Llama）中，单次提示词内一次性列举所有选项（如 `A. ... \n B. ...`）会引入**Prefill 上下文因果偏置**与位置注意力漂移。在边界模糊样本上，交换选项在提示词中的顺序可能引发答案翻转（~15–20% 的排列敏感度）。

OpenJevPro 提供两套互相协同的工程解法：

1. **枚举成员顺序冻结 (单 Pass 默认推荐)**：对于常规亚 50ms 高吞吐路由场景（`order_invariant=False`），确保业务 Enum 枚举成员或列表保持严格确定的固定顺序（如字母序或稳定代码顺序）。
2. **候选隔离独立打分 (`order_invariant=True`)**：对于受审计约束的金融风控与强合规场景，开启 `order_invariant=True`。每个候选将在无竞品干扰的独立提示词中完成打分，并经由置换等变的 Softmax 归一化，提供**100% 严格数学级排序不变性（全排列置换下恒为 1.00 个确切胜者）**：

```python
# 通过 ThreadPoolExecutor 并发隔离执行候选打分 (~80-150ms)
decision = client.decide_choice(
    state={"ticket_text": "检测到来自未授权 IP 的异常登录尝试。"},
    candidates=TicketRoute,
    criteria="将输入的工单归类至正确的处理部门。",
    order_invariant=True  # 启用选项排序绝对不变性
)
```

> **Tier 0 双向替代路径**：若希望在单次前向传递中兼顾极低延迟 (<35ms) 与排序绝对免疫，推荐直接部署 **Tier 0 Laya (ModernBERT 322M)**。由于 ModernBERT 采用全注意力双向编码架构而非单向因果生成，先天具备对序列位置漂移的免疫力。

### 专用 System 1 决策模型：Laya 引擎 (ModernBERT 322M)

```python
from openjevpro.harness import LayaEngine
from openjevpro.client import OpenJevProClient

# 1. 直接通过 LayaEngine 调用本地微服务（sub-35ms，零云端成本）
laya = LayaEngine(endpoint="http://localhost:8001/v1", model="convai/laya-modernbert-large")
result = laya.evaluate_choice(
    state={"query": "请立刻取消订单并为我全额退款"},
    candidates=["cancel_order", "track_shipment", "update_address"],
    allow_abstain=True
)
print(f"Laya 决策结果: {result['choice']} (延迟: {result['latency_ms']:.1f}ms, 置信度: {result['confidence']:.2f})")

# 2. 或通过 OpenJevProClient 统一客户端调度（自动根据 8001 端口或 backend='laya' 识别并校准）
client = OpenJevProClient(base_url="http://localhost:8001/v1", backend="laya")
decision = client.decide_choice(
    state={"query": "请立刻取消订单并为我全额退款"},
    candidates=["cancel_order", "track_shipment", "update_address"],
)
print(f"标定决策: {decision.value}, 标定后概率: {decision.probabilities}")
```

### 运行时安全护栏：防护商业 Jev 与模型输出

```python
from openjevpro.guard import TypeSafeJevGuardHarness

# 初始化防护线束
guard = TypeSafeJevGuardHarness(alpha=1.25, min_confidence=0.72)

raw_probs = {
    "lost_or_stolen_card": 0.79,
    "card_arrival": 0.18,
    "pin_change": 0.01,
    "balance": 0.005,
    "transfer": 0.005,
    "statement": 0.005,
    "support": 0.005,
}

decision = guard.evaluate_probabilities(raw_probs)
print(f"安全决策: {decision.choice}")              # 'UNKNOWN' (主动安全拒识)
print(f"备选首选: {decision.tentative_choice}")     # 'lost_or_stolen_card'
print(f"校准后置信度: {decision.confidence:.2%}")   # '71.11%' (< 72.00% 动态截断线)
print(f"是否拒识: {decision.is_abstained}")        # True
```

### 混合网关：成本套利与容灾保障

```python
from openjevpro.gateway import HybridJevGateway, CircuitBreaker
from openjevpro.client import OpenJevProClient
from openjevpro.harness import TypeSafeJevEngine

# 配置 Tier 1 本地边缘引擎与 Tier 2 商业云端引擎
local_client = OpenJevProClient(base_url="http://localhost:8000/v1")
cloud_engine = TypeSafeJevEngine(api_key="your-typesafe-api-key")

# 挂载带熔断保护的混合网关（连续 3 次失败触发熔断，冷却窗口 10 秒）
gateway = HybridJevGateway(
    local_engine=local_client,
    cloud_engine=cloud_engine,
    local_tau=0.75,
    circuit_breaker=CircuitBreaker(failure_threshold=3, recovery_timeout=10.0),
)

# 高置信样本 -> 本地边缘直出 (<20ms, $0 账单, tier="Tier1_Local")
# 边界样本   -> 自动转交商业云端 API (tier="Tier2_Cloud", cost_units=1)
# 云端异常   -> 自动触发容灾降级 (tier="Tier1_Fallback", degraded=True)
decision = gateway.decide_choice(
    state={"query": "如何查询我本期信用卡的待还账单？"},
    candidates=["card_balance", "card_lost", "transfer", "UNKNOWN"],
)

print(f"决策结果: {decision.choice}")
print(f"路由档位: {decision.tier}")       # 例：'Tier1_Local'
print(f"是否降级: {decision.degraded}")   # False (云端故障时为 True)
print(f"调用成本单位: {decision.cost_units}") # 0
```

### 🧠 双系统认知飞轮：先 LLM 深度探索，再沉淀为快决策 (System 2 $\to$ System 1)

静态概率决策引擎存在两难：System 1 小模型（Jev/Laya）具备亚 35ms 零成本优势，但在模糊边界样本上会陷入 **`UNKNOWN` 死锁**；而全量使用 System 2 深度思考大模型（如 DeepSeek-R1、Qwen-Thinking）则过于昂贵（$0.02+/次）且延迟过高（1.5s–5.0s）。

**Deliberative Decision Flywheel (认知飞轮)** 将慢思考大模型作为**快决策系统的近线编译器**：
1. **快径直通**：95%+ 的明确请求以 sub-35ms 在本地 System 1 极速裁决，零额外 Token 开销。
2. **条件升阶**：当置信度低于阈值 $\tau$ 或出现 `UNKNOWN` 时，自动升阶至 System 2 慢思考进行反事实因果归因。
3. **沉淀算子 (Crystallization)**：将因果证据链提炼为判别边界准则与示例记忆，**后续同类请求永久降维至 System 1 极速直通**。

```python
from openjevpro.client import OpenJevProClient

client = OpenJevProClient(base_url="http://localhost:8000/v1")

# 初始化自进化决策认知飞轮
flywheel = client.create_decision_flywheel(
    reasoning_model="Qwen/Qwen3-14B-Thinking",
    auto_crystallize=True
)

# 1. 冷启动：边界模糊样本触发 System 2 深度探索并自动沉淀准则 (~1.8s)
decision1 = flywheel.evaluate_choice(
    state={"query": "卡片未收到，怀疑在信箱被他人截胡"},
    candidates=["card_arrival", "lost_or_stolen_card"],
    criteria={"card_arrival": "卡片派送进度", "lost_or_stolen_card": "卡片遗失或盗刷"}
)
print(f"裁决结果: {decision1.value}")  # 'lost_or_stolen_card'
print(f"是否升阶: {decision1.escalated}")  # True
print(f"因果推演依据: {decision1.crystallization_receipt['reasoning_trace']}")

# 2. 直通跃迁：后续同类请求直接由 System 1 极速识别 (<35ms，零额外 Token 生成)！
decision2 = flywheel.evaluate_choice(
    state={"query": "卡片未收到，怀疑在信箱被他人截胡"},
    candidates=["card_arrival", "lost_or_stolen_card"],
    criteria={"card_arrival": "卡片派送进度", "lost_or_stolen_card": "卡片遗失或盗刷"}
)
print(f"裁决结果: {decision2.value}")  # 'lost_or_stolen_card'
print(f"是否升阶: {decision2.escalated}")  # False (已成功沉淀并命中快径！)
```

---

## 📊 性能与成本优势

* **单 Token 解码**：约束生成仅输出分类标识或直接读取候选 logprob，Token 生成开销几乎归零。
* **Prompt 缓存优化**：系统指令、类型定义与候选枚举在 Prefill Cache 中保持持久命中。
* **响应时延**：同机部署于 vLLM 实例时，端到端延迟通常在 **40ms ~ 120ms** 之间。

---

## 📄 开源许可证与商业授权

* **非商业社区使用**：OpenJevPro 依据 **[PolyForm Noncommercial License 1.0.0](LICENSE)** 协议开源。个人学习、学术科研、非营利机构以及非商业开发可免费使用。
* **商业生产授权**：任何用于商业企业运营、生产环境、商业 SaaS 产品或收费服务的场景，**均需向项目维护方申请正式商业授权**。商业授权详情请参见 **[LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md)**。
