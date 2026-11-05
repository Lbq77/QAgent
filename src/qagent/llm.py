"""模型调用封装（PRD §20.1 Day 1「模型调用」）。

统一接口 `invoke(system, user) -> str`，工厂 `get_llm(settings)` 按配置切换实现：
- StubLLM：确定性离线桩，返回一份合法的计划 JSON，无 key / 无网络也能跑通；
- OpenAICompatibleLLM：接 langchain `init_chat_model(model_provider="openai")`，
  覆盖 OpenAI、硅基流动(SiliconFlow)、DeepSeek、Moonshot 等一切 OpenAI 兼容接口。

model_provider 取值：
- `stub`        —— 离线桩（默认）
- `siliconflow` —— 读 Settings 里的 siliconflow_* 字段（SILICONFLOW_* 环境变量）
- `openai`      —— 读 Settings 里的 openai_* 字段
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod

# 默认计划（StubLLM 返回；结构与 PlannerOutput 一致）
DEFAULT_PLAN: dict = {
    "goal": "",
    "steps": ["research", "quant", "risk_audit", "fact_check", "report"],
    "needs_research": True,
    "needs_backtest": True,
    "needs_risk_audit": True,
    "missing_params": [],
    "market": "US",
    "universe": ["AAPL", "MSFT", "NVDA"],
    "date_range": {},
}


class LLMProvider(ABC):
    """所有 LLM 接入的统一接口。"""

    @abstractmethod
    def invoke(self, system: str, user: str) -> str:
        """给定 system/user 提示词，返回模型文本输出。"""


class StubLLM(LLMProvider):
    """离线桩：返回确定性计划 JSON。

    可注入 `raw` 自定义返回文本，用来测试非法 JSON / 缺参等失败回退路径。
    """

    def __init__(self, raw: str | None = None) -> None:
        self._raw = raw

    def invoke(self, system: str, user: str) -> str:
        if self._raw is not None:
            return self._raw
        plan = dict(DEFAULT_PLAN)
        plan["goal"] = user

        # 它将内存中的 Python 对象转化为可跨网络传输或存盘的 JSON 文本。
        return json.dumps(plan, ensure_ascii=False)


class OpenAICompatibleLLM(LLMProvider):
    """OpenAI 兼容接口适配器，用 langchain `init_chat_model` 统一接入。

    与直接 new ChatOpenAI 等价，但保留用户熟悉的 `init_chat_model(model_provider="openai")`
    写法；惰性导入，stub 环境无需这些依赖。
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        temperature: float = 0,
    ) -> None:
        from langchain.chat_models import init_chat_model  # 惰性导入

        self._model = init_chat_model(
            model=model,
            model_provider="openai",  # 硅基流动等均为 OpenAI 兼容接口，保持不变
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )

    def invoke(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        resp = self._model.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        return str(resp.content)


def get_llm(settings) -> LLMProvider:
    """按 model_provider 返回对应实现（工厂）。"""
    provider = settings.model_provider

    if provider == "stub":
        return StubLLM()

    if provider == "siliconflow":
        if not settings.siliconflow_api_key:
            raise RuntimeError("model_provider=siliconflow 但未设置 SILICONFLOW_API_KEY")
        return OpenAICompatibleLLM(
            model=settings.siliconflow_model,
            api_key=settings.siliconflow_api_key,
            base_url=settings.siliconflow_base_url,
        )

    # openai（及任意 OpenAI 兼容接口）
    if not settings.openai_api_key:
        raise RuntimeError(
            "model_provider=openai 但未设置 API key（QAGENT_OPENAI_API_KEY 或 OPENAI_API_KEY）"
        )
    return OpenAICompatibleLLM(
        model=settings.model_name,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
