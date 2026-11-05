"""相对时间 → 具体日期（Day 2.5，确定性代码，不依赖 LLM 计算「今天」）。

支持：
- 相对时间：最近一年 / 最近三个月 / 过去五年 / 近3月 / 最近两周 …
- 明确区间：2020年至2024年 / 2024-01-01 到 2024-06-30
"""
from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

_CN_DIGITS = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}

# 相对时间：最近/过去/近 + 数量(阿拉伯或中文数字) + 单位
_REL = re.compile(
    r"(最近|过去|近)\s*([0-9零一二两三四五六七八九十]+)\s*(年|个月|月|星期|周|天|日)"
)
# 完整日期 / 年份
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_YEAR = re.compile(r"(?:19|20)\d{2}")


def _to_int(s: str) -> int | None:
    s = s.strip()
    if s.isdigit():
        return int(s)
    if s in _CN_DIGITS:
        return _CN_DIGITS[s]
    if "十" in s:
        left, _, right = s.partition("十")
        tens = _CN_DIGITS.get(left, 1) if left else 1
        ones = _CN_DIGITS.get(right, 0) if right else 0
        return tens * 10 + ones
    return None


def _add_months(d: date, n: int) -> date:
    """往前推 n 个月，天数按目标月实际天数截断。"""
    total = d.year * 12 + (d.month - 1) - n
    y, m = divmod(total, 12)
    m += 1
    last = calendar.monthrange(y, m)[1]
    return date(y, m, min(d.day, last))


def resolve_date_range(text: str, today: date | None = None) -> dict[str, str] | None:
    """从文本解析出 (start, end)，解析不到返回 None。"""
    today = today or date.today()

    # 1) 完整日期区间（优先，保留精确日期）
    dates = sorted(set(_DATE.findall(text)))
    if len(dates) >= 2:
        return {"start": dates[0], "end": dates[-1]}

    # 2) 相对时间
    m = _REL.search(text)
    if m:
        n = _to_int(m.group(2))
        unit = m.group(3)
        if n is not None:
            if unit == "年":
                start = _add_months(today, n * 12)
            elif unit in ("个月", "月"):
                start = _add_months(today, n)
            elif unit in ("星期", "周"):
                start = today - timedelta(weeks=n)
            else:  # 天 / 日
                start = today - timedelta(days=n)
            return {"start": start.isoformat(), "end": today.isoformat()}

    # 3) 年份区间
    years = sorted(set(_YEAR.findall(text)))
    if len(years) >= 2:
        return {"start": f"{years[0]}-01-01", "end": f"{years[-1]}-12-31"}

    return None
