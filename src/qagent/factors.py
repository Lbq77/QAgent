"""因子注册表与归一化（Day 2.5）。

LLM 只输出因子名 + 方向，程序在这里做白名单校验与方向兜底，保证确定性执行。
- 未指定因子（空）→ 默认 momentum_20d/desc；
- 显式指定但不在白名单 → 抛 UnknownFactorError（上游阻塞任务，绝不静默替换）。

`market_cap` 是小幅扩展（PRD 原文只列价量因子，用户要求支持市值因子）。
"""
from __future__ import annotations


class UnknownFactorError(ValueError):
    """用户显式指定的因子不在白名单中（不能静默替换，必须阻塞）。"""


FACTOR_REGISTRY: dict[str, dict[str, str]] = {
    "momentum_5d": {"direction": "desc"},
    "momentum_20d": {"direction": "desc"},
    "momentum_60d": {"direction": "desc"},
    "volatility_20d": {"direction": "asc"},   # 波动率越低越靠前
    "avg_volume_20d": {"direction": "desc"},  # 均量越大越靠前（流动性）
    "market_cap": {"direction": "asc"},       # 市值越小越靠前
}

# 常见口语 / 非规范表达 -> 规范名（可逐步补充）
FACTOR_ALIASES: dict[str, str] = {
    # 市值
    "市值": "market_cap",
    "小市值": "market_cap",
    "size": "market_cap",
    "marketcap": "market_cap",
    "market cap": "market_cap",
    # 动量
    "动量": "momentum_20d",
    "20日动量": "momentum_20d",
    "momentum": "momentum_20d",
    "5日动量": "momentum_5d",
    "60日动量": "momentum_60d",
    # 波动率
    "波动率": "volatility_20d",
    "20日波动率": "volatility_20d",
    "volatility": "volatility_20d",
    # 成交量 / 流动性
    "成交量": "avg_volume_20d",
    "均量": "avg_volume_20d",
    "平均成交量": "avg_volume_20d",
    "20日均量": "avg_volume_20d",
    "流动性": "avg_volume_20d",
}


def resolve_factor(factor: str | None, direction: str | None) -> dict[str, str]:
    """归一化因子：空 → 默认动量；未知 → 抛 UnknownFactorError；已知 → 白名单 + 方向兜底。"""
    raw = (factor or "").strip()
    name = raw.lower()
    if name in FACTOR_ALIASES:
        name = FACTOR_ALIASES[name]
    if not name:
        # 用户「没指定」因子 → 默认动量；这是未指定的默认值，不是替换
        return {"factor": "momentum_20d", "direction": "desc"}
    if name not in FACTOR_REGISTRY:
        raise UnknownFactorError(raw or name)
    d = (direction or "").strip().lower()
    if d not in ("asc", "desc"):
        d = FACTOR_REGISTRY[name]["direction"]
    return {"factor": name, "direction": d}


# 用户可能提到、但当前不支持的因子（检测到就阻塞，防止 LLM 擅自改选别的因子）
UNSUPPORTED_FACTOR_HINTS: list[tuple[str, str]] = [
    ("估值因子", "估值因子"),
    ("估值", "估值因子"),
    ("市盈率", "市盈率"),
    ("pe ratio", "市盈率"),
    ("p/e", "市盈率"),
    ("市净率", "市净率"),
    ("pb ratio", "市净率"),
    ("p/b", "市净率"),
    ("价值因子", "价值因子"),
    ("质量因子", "质量因子"),
    ("成长因子", "成长因子"),
    ("股息率", "股息率"),
    ("股息", "股息因子"),
    ("分红", "分红因子"),
    ("基本面", "基本面因子"),
]


def detect_unsupported_factor(text: str) -> str:
    """从用户原文识别「暂不支持的因子」，识别不到返回空串。"""
    t = text.lower()
    for kw, label in UNSUPPORTED_FACTOR_HINTS:
        if kw in t:
            return label
    return ""
