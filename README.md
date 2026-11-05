# QAgent — Multi-Agent Quant Research Copilot

基于 LangGraph 的多智能体量化研究助手（Agent 工程实践项目，量化研究为业务场景）。

> 借鉴 [HKUDS/Vibe-Trading](https://github.com/HKUDS/Vibe-Trading) 的数据源降级、数字核查、规则断言评测等思路；本项目重点放在 LangGraph 工作流、Harness、上下文管理与执行轨迹上。

## Day 1 交付：最小 Agent Runtime

当前只完成 PRD §20.1 Day 1 的骨架：

- **项目结构**：`src/` 布局 + `pyproject.toml` + pydantic-settings 配置。
- **State**：`src/qagent/state.py` 定义 Pydantic 化的 `AgentState`（PRD §6.3）与任务/节点状态枚举（§6.4）。
- **模型调用**：`src/qagent/llm.py` 提供 `StubLLM`（离线桩）与 `OpenAILLM`（真模型），工厂切换。
- **工作流骨架**：`src/qagent/graph/` 定义 8 个节点 + 条件路由，离线跑通

```
planner → research → quant → human_confirm → risk → fact_check → report → END
                ↳ error_handler → END（失败/缺参数/拒绝时）
```

节点目前是确定性桩，打通 State 传递与 node trace；真实工具（Day 3–6）、人工中断（Day 7）、SQLite Checkpoint（Day 8）会逐日替换。

## Day 2–4 进展

- **Day 2**：节点/条件路由收口，planner 真正解析 LLM 输出并落 plan + constraints；支持按需跳过、缺参阻塞、失败/拒绝收尾。
- **Day 3**：统一工具系统（`src/qagent/tools/`）——9 个工具定义 + Pydantic 输入/输出 Schema + 权限白名单 + 注册表；测试集 v1 在 `evals/`，`python evals/run_eval.py` 跑端到端评测。
- **Day 4**：行情数据源（`src/qagent/data/`）——`MarketDataService` 降级链 + `YahooProvider`（yfinance，复权价）；`get_market_data` 已接真数据，`research_node` 经 registry 拉取行情。

## 目录结构

```
src/qagent/
├── config.py        # pydantic-settings 配置
├── state.py         # AgentState + 状态枚举 + TraceRecord
├── llm.py           # StubLLM / OpenAILLM / 工厂
├── context.py       # RuntimeContext（settings + llm + tool_registry）
├── data/            # 行情数据源与降级链（yfinance）
├── tools/           # 统一工具系统（base/registry/schemas + 9 工具）
├── agents/          # 四角色定义、prompt、planner 规划逻辑
└── graph/           # nodes / routing / builder（LangGraph 图）
```

## 运行

```bash
# 1) 建虚拟环境并安装依赖
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# 2) 冒烟运行（默认 stub 模型，无需 API key）
.venv/bin/python -m qagent.runtime "研究 AAPL、MSFT、NVDA 的 20 日动量并回测"

# 3) 跑测试
.venv/bin/pytest
```

接入真实模型（硅基流动）：复制 `.env.example` 为 `.env`，设置：

```bash
QAGENT_MODEL_PROVIDER=siliconflow
SILICONFLOW_API_KEY=sk-xxx
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_MODEL=deepseek-ai/DeepSeek-V3
```

其他 OpenAI 兼容接口（DeepSeek/Moonshot 等）同理，用 `QAGENT_MODEL_PROVIDER=openai` + `QAGENT_OPENAI_BASE_URL`。

## 14 天路线（Day 1 之后）

| 天 | 内容 |
|---|---|
| D2 | Node/Edge/条件路由收口，可运行工作流 |
| D3 | 统一工具系统 + Schema 校验 + 测试集 v1 |
| D4 | 行情与 GraphRAG 工具，跑通 Research |
| D5–6 | JSON 选股、因子、回测闭环 |
| D7 | 四角色串联 + 人工确认（interrupt） |
| D8 | Harness（权限/超时/重试）+ Checkpoint 恢复 |
| D9–11 | Trace、上下文外置、数字核查/证据链/实验记录 |
| D12–14 | FastAPI+SSE、Vue3 前端、回归 + 2 个对照实验 |
