# QAgent — Multi-Agent Quant Research Copilot

基于 LangGraph 的多智能体量化研究助手：自然语言下指令，多角色 Agent 协作完成「研究 → 选股 → 回测 → 风险审计 → 报告」，所有数值由确定性工具计算，可追溯、可复现。

## 功能特性

- **多角色工作流**：Planner / Research / Quant / Risk / FactCheck 多 Agent 经 LangGraph 状态机编排，支持按需跳过、条件路由、失败收尾。
- **确定性执行**：LLM 只产出 JSON 配置，因子值、回测指标、风险结论全部由程序计算，杜绝编造数字。
- **防未来函数**：信号窗 / 持有窗分离、t+1 开盘执行、0x / 默认 / 2x 三档成本对比。
- **技术指标因子**：动量 / 波动率 / 均量之外，新增 RSI（`rsi_14`）、MACD 信号线（`macd_signal`），同样只用信号窗内数据计算，延续 PIT 原则。
- **风险指标**：回测除年化 / 回撤 / 夏普外，输出 95% 单日 VaR 与年化波动率，一并纳入 fact_check 与总结数字核查。
- **人工确认**：回测后经 interrupt 挂起，支持 approve / modify / reject 三种决策。
- **风险审计 + 数字核查**：显式检查未来函数 / t+1 执行 / 成本 / 幸存者偏差；报告数字逐条与工具产物比对，附证据链。
- **可追溯 & 可复现**：工具调用轨迹、SQLite 断点恢复、结果外置存储、实验记录。

## 技术栈

LangGraph · Pydantic · SQLite · FastAPI + SSE · Vue3

## 目录结构

```
src/qagent/
├── config.py        # pydantic-settings 配置
├── state.py         # AgentState + 状态枚举 + TraceRecord
├── llm.py           # StubLLM / OpenAICompatibleLLM / 工厂
├── context.py       # RuntimeContext（settings + llm + tool_registry）
├── harness.py       # 工具调用 Harness（预算 / 超时 / 重试）
├── checkpoint.py    # SQLite checkpointer（断点恢复）
├── store.py         # 回测结果 / 实验记录存储（SQLite）
├── backtest.py      # 确定性向量化回测引擎
├── data/            # 行情数据源与降级链（yfinance）+ 新闻/舆情源（桩 / Tavily）
├── tools/           # 统一工具系统（10 个工具）
├── agents/          # 四角色定义与规划逻辑
├── api.py           # FastAPI + SSE 服务
├── web/             # Vue3 前端单页
└── graph/           # nodes / routing / builder（LangGraph 图）
```

## 快速开始

```bash
# 1) 建虚拟环境并安装依赖
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# 2) 冒烟运行（默认 stub 模型，无需 API key）
.venv/bin/python -m qagent.runtime "研究 AAPL、MSFT、NVDA 在 2024-01-01 到 2024-12-31 的 20 日动量并回测"

# 3) 跑测试
.venv/bin/pytest

# 4) 跑对照实验（动量 vs 波动率、成本 0x vs 2x）
.venv/bin/python scripts/experiments.py

# 5) 启动 Web 服务（FastAPI + SSE + Vue3 前端）
.venv/bin/pip install -e ".[web]"
.venv/bin/uvicorn qagent.api:app --reload
# 打开 http://127.0.0.1:8000
```

## 接入真实模型

复制 `.env.example` 为 `.env`，设置 SiliconFlow（硅基流动，OpenAI 兼容接口）：

```bash
QAGENT_MODEL_PROVIDER=siliconflow
SILICONFLOW_API_KEY=sk-xxx
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_MODEL=deepseek-ai/DeepSeek-V3
```

其他 OpenAI 兼容接口（DeepSeek / Moonshot 等）同理，用 `QAGENT_MODEL_PROVIDER=openai` + `QAGENT_OPENAI_BASE_URL`。

## 新闻 / 舆情（search_news）

research 节点新增第 4 个工具 `search_news`：模型自己写检索词、查不到就换词再查（唯一允许模型给参数的只读工具）。

- **默认（未配 key）**：`StubNewsProvider` 离线桩，结果明确标记 `[stub]`，**不伪装成真实新闻**。
- **接真实新闻源**：配 Tavily API key（有免费额度）：

```bash
QAGENT_TAVILY_API_KEY=tvly-xxx
```

检索词由模型写，但新闻结果只作为**证据引用**进入研究摘要，不进入回测信号，不破坏「数字来自确定性工具」的不变量。
