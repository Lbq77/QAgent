# QAgent 面试讲稿（含真实评测数字）

> 用途：面试前反复看，把每个「为什么」讲成自己的话。所有数字都是 2026-11 在本机实测的，可以放心报。

---

## 0. 一句话定位

QAgent 是一个 **LangGraph 驱动的多角色量化研究 Copilot**：把一句中文研究需求，拆成「规划 → 研究 → 回测 → 人工确认 → 风险审计 → 事实核查 → 报告」的固定流水线。核心卖点不是「让 LLM 自由发挥」，而是**确定性、可核查、防未来函数**——数字全部来自工具产物，LLM 只做窄任务。

---

## 1. 架构总览

```
用户 query
   │
   ▼
planner ──(缺参数)──▶ await_input ──▶ END(blocked)
   │
   ▼
research ──(可选，内部是小型 ReAct 循环，见 §3.7)
   │
   ▼
quant ──▶ human_confirm ──(modify)──▶ 回 quant 重跑
   │           │
   │        (approve)
   ▼           ▼
(report)   risk ──▶ fact_check ──▶ report ──▶ END
   ▲
   └── 不需要研究/回测时直接跳过前面节点
```

- **7 个节点**（完整链路）：`planner → research → quant → human_confirm → risk → fact_check → report`
- **任何节点抛错** → 统一兜底到 `error_handler`，落 `status=FAILED`，不把失败伪装成成功
- **唯一回路**：`human_confirm → quant`（用户点「modify」才发生），**人驱动，不是 Agent 自主循环**

四个角色 + 两个收尾：
| 角色 | 干什么 | 是不是 LLM |
|---|---|---|
| planner | 把需求解析成结构化 plan（因子/股票池/日期/top_n） | ✅ LLM 解析 |
| research | 查行情/知识库/公司概况，产出带引用的证据 | 路由决策用 LLM，工具确定性 |
| quant | 因子计算 + 回测 + 选股 | ❌ 确定性代码 |
| risk_auditor | 审计未来函数/幸存者偏差/成本 | ❌ 确定性代码 |
| fact_check | 报告数字逐条 vs 工具产物 | ❌ 确定性代码 |
| report | 汇总成可读报告 | ✅ LLM 总结（数字仍来自工具） |

> **关键点**：全项目只有 **3 个 LLM 调用点**（planner 解析、research 路由、report 总结）。其余全是确定性工具。面试官若问「多智能体」，诚实回答：**是「角色分工 + 确定性编排」的多角色，不是多个自由对话的 LLM agent**。

---

## 2. 一次任务的完整流转（背下来，随时能画）

1. **planner**：LLM 输出严格 JSON（`needs_research / needs_backtest / factor / universe / date_range / missing_params`）。因子归一化到固定枚举（`momentum_20d` 等），未知因子 → `await_input` 拦截。
2. **research**：内部 ReAct 循环，模型逐轮选工具（§3.7），参数一律由代码从 `task_constraints` 填。行情快照按 `snapshot_id` 存 SQLite，State 只留 ID。
3. **quant**：Schema 校验策略配置 → 确定性回测（t 日信号、t+1 执行、成本非 0）。复用 research 的 `snapshot_id`，不重复拉数据。
4. **human_confirm**：LangGraph `interrupt()` 挂起，人工 approve / modify / reject。
5. **risk**：审计未来函数、幸存者偏差、成本口径。
6. **fact_check**：把报告里的收益率/回撤/夏普**逐条**和回测结果比对，对不上就标 mismatch。
7. **report**：LLM 总结 + 三档成本敏感度；总结里的数字再被 `_verify_summary_claims` 抓出来核对，对不上就弃用 LLM 总结、退回确定性模板。

---

## 3. 核心设计决策 + 为什么（面试的肉）

### 3.1 为什么固定 DAG，而不是自由 ReAct 循环？
- **轮数有上界、成本可预测**：完整链路固定 7 节点，不会失控打爆数据源。
- **可核查**：每一步产物都能回溯到工具结果，fact_check 才能逐条对账。自由多轮推理会让「这个数字哪来的」变得不可追踪。
- **金融场景的正确取舍**：宁可少一点「智能」，也要「数字对」。这是和 Vibe-Trading 最本质的分野（§5）。

### 3.2 State 只存摘要 + 引用 ID，全量外置
- `AgentState` 里是 `snapshot_id / backtest_result_ref / evidence_refs` 这类**引用**，完整行情/回测结果在 SQLite 的 `ResultStore`。
- 工具输入/输出 trace 压成 200 字摘要（`_summarize`）。
- **好处**：State 不会随任务膨胀；跨进程能按 ID 取回结果。

### 3.3 Harness：预算 / 超时 / 重试
- `Harness` 包在所有工具调用外层：`max_tool_calls=30`（预算，超了停）、`tool_timeout_seconds=30`（线程超时）、`retries=1`（超时退避重试，Schema/权限错误不重试）。
- 预算**按任务重置**，不跨任务累加（有测试保证）。

### 3.4 fact_check：专门防 LLM 编数字
- 三件事：① 证据链存在（回测有 ref、研究有 evidence）；② 三大指标（年化/回撤/夏普）逐条 vs 结果存储；③ 风险等级与配置交叉核对（low 必须是「无违规」的配置）。
- report 里的 LLM 总结还会被正则抓出所有百分比和夏普，**对不上就弃用 LLM 总结、退回确定性模板**——这是本项目「LLM 只负责措辞，不负责数字」的最硬防线。

### 3.5 未来函数 / PIT 对齐（git 历史里的重点）
- 最近一次提交就是「移除 market_cap 因子——PIT 未对齐存在未来函数，检测到即阻塞」。
- 信号窗与持有窗分离（`lookback_days` 只用前 N 个交易日算因子，其后才是持有窗）、t 日信号 t+1 日执行、成本非 0，都是防未来函数的具体手段。

### 3.6 Checkpoint：断点恢复
- 自研 `SqliteSaver`（标准库 sqlite3，语义对齐官方 `InMemorySaver`），`thread_id = task_id` 一一对应。
- 配合 `interrupt()`：新进程也能 `confirm(task_id, decision)` 恢复之前挂起的确认点。
- 诚实点：官方有 `langgraph-checkpoint-sqlite`，我自研是因为当时离线装不了——**面试若被问，直接说这是学习/演示用途的重写，不是生产必要**。

### 3.7 Research 的小型 ReAct 循环（新加的）
- 模型**每轮只选一个「无参动作」**：`{"action": "get_market_data"}` 或 `{"action": "done", "reason": ...}`。
- **参数由代码填**（symbols/日期来自 `task_constraints`），模型绝不编参数。
- 边界：最多 `max_research_rounds=5` 轮；同一工具不重复调；空结果 → 标「未查到」；解析失败 → 兜底拉一次行情。
- 上下文：工具结果只把「决策级摘要」放进**局部对话**，不进 `AgentState.messages`，完整结果外置。

---

## 4. 真实评测数字（2026-11 实测）

**测试条件**：真 LLM = SiliconFlow `deepseek-ai/DeepSeek-V4.1-Flash`；行情/知识库用桩（桩不影响轮数与 token 指标）；GraphRAG 关；6 条真实 query。

| 指标 | 数值 |
|---|---|
| research 轮数 | **平均 2.4，最长 3** |
| LLM 调用 / 任务 | **平均 3.8，最长 5** |
| token / 任务 | **平均 ~2300，最长 ~2905**（总 13821） |
| 完整链路图步数 | 7 节点（固定） |
| research-only | 3 节点 |

**面试被问「平均几轮 / 最长几轮」时的标准答法**（把「轮」拆开讲，最显专业）：

> 「轮」在我这里有三个口径：外层图步数是固定的 7 节点；内层 research 的 ReAct 循环实测平均 2.4 轮、最长 3 轮（因为只有 3 个工具 + 不重复调，天然收敛在 ≤3）；LLM 调用平均每任务 3.8 次、最长 5 次。所以它是有上界的确定性系统，不是开放式循环。

---

## 5. 和 Vibe-Trading（HKUDS）的对比

| 维度 | QAgent | Vibe-Trading |
|---|---|---|
| 定位 | 单机量化研究流水线 | 实验室产品级研究平台（30 预设团队、Swarm、MCP、前端） |
| 编排 | LangGraph 固定 DAG | 自研 ReAct `AgentLoop` |
| 工具参数 | 代码填（LLM 不编） | 模型原生 tool calling（可编，靠 schema/网关兜底） |
| 安全层 | 独立 `Harness`（预算/超时/重试） | iteration cap + Swarm deadline |
| 核心理念 | 确定性 + 可核查 | 表达力 + 多智能体协作 |

**一句话对比**：「它是自由 ReAct + 多智能体，我是固定 DAG + 事实核查。取向不同，不是高下。它适合探索式研究，我适合『数字不能错』的回测工作流。」

> 面试若被问「你参考了 Vibe-Trading 吗」，回答：**研究了它的 AgentLoop 和 Swarm，但它和我项目的定位不同——我的重点是确定性编排和防未来函数，不是多智能体规模。**

---

## 6. 高频 Q&A（可直接背）

**Q：跑一个任务平均多少轮？最长多少？**
→ §4 的标准答法。

**Q：轮数上去了，上下文怎么处理？**
→ 三件事：① State 只存摘要+引用 ID，全量外置 SQLite；② 每次 LLM 调用都是全新 `system+user`，不累积 message 历史；③ research 循环里工具结果只放决策级摘要，早期轮次可压缩。所以「轮数」和「上下文大小」是解耦的。

**Q：怎么防 LLM 编数据？**
→ 三层：① 参数由代码填，不信任 LLM 编 symbols/日期；② 数字全部来自确定性工具产物；③ fact_check + `_verify_summary_claims` 逐条核对，对不上就弃用 LLM 总结。

**Q：为什么不用 create_react_agent / bind_tools？**
→ 它们默认让模型连参数一起吐，会推翻「参数由代码填」这条安全不变量。我用手写的无参动作选择循环，代价更小、可 stub 测试。Vibe-Trading 敢让模型编，是因为它是探索型场景；我做回测，编错参数 = 错数字被当事实。

**Q：工具调用失控怎么办？**
→ `Harness` 预算（30/任务）+ 超时（30s）+ 重试（1 次退避）。超预算抛 `HarnessBudgetExceeded`，走 error_handler 落 FAILED，不崩流程。

**Q：多智能体是怎么协作的？**
→ 严格说是「多角色确定性编排」：四个角色共享一个 `AgentState`，按 DAG 顺序接力，产物通过「摘要 + 引用 ID」传递。不是多个 LLM 互相对话。

---

## 7. 诚实边界（防被追问穿帮）

面试最忌讳把桩当成果。**这些是「演示接口、未接真实」的，被问到要主动说**：

- **默认 `model_provider=stub`**：离线桩，返回写死 JSON。真跑要配 SiliconFlow/OpenAI（本机已配 SiliconFlow）。
- **行情**：默认走 yfinance（真实）；评测/测量里用了 `StubMarketDataProvider`。
- **知识库**：GraphRAG 本地索引已就绪，但未启用时 `search_financial_knowledge` 回退到桩。
- **市值/基本面、指数成分股解析**：代码里有 `fetch_market_cap`、`index` 字段，但成分股解析标记「待接入」。
- **自研 `SqliteSaver`**：学习/演示用途重写官方 checkpointer，不是生产取舍。

**面试时主动说一句**：「项目外壳是生产级的（harness、checkpoint、schema 校验、评测、PIT），但内核目前是『薄 LLM + 桩数据』——它的价值在工程方法和可核查性，不在规模。」这句话既诚实又显得你清楚边界。
