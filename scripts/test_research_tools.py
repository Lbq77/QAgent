"""验证 research 两个工具是否真的跑通。

用法（在项目根目录）：
    .venv/bin/python scripts/test_research_tools.py

两个工具：
  1) get_market_data        —— 拉股票行情（Yahoo，需要联网）
  2) search_financial_knowledge —— GraphRAG 检索（global/local，走 SiliconFlow）
"""
from __future__ import annotations

from dotenv import load_dotenv

from qagent.tools.research import (
    GetCompanyProfile,
    GetMarketData,
    SearchFinancialKnowledge,
)

load_dotenv()


def test_company_profile() -> None:
    print("=" * 64)
    print("0) get_company_profile —— 查公司概况（yfinance）")
    print("=" * 64)
    tool = GetCompanyProfile()
    for sym in ["NVDA", "0700.HK"]:
        out = tool.call({"symbol": sym})
        print(f"  {out.symbol:10s} name={out.name!r}  sector={out.sector!r}  exchange={out.exchange!r}")
    print()


def test_market_data() -> None:
    print("=" * 64)
    print("1) get_market_data —— 拉真实行情")
    print("=" * 64)
    tool = GetMarketData()
    out = tool.call(
        {"symbols": ["NVDA", "AAPL"], "start": "2026-10-01", "end": "2026-11-01"}
    )
    print(f"source={out.source}  snapshot_id={out.snapshot_id}")
    for r in out.records[:5]:
        print(f"  {r.symbol:6s} {r.date}  close={r.close}")
    print(f"  ... 共 {len(out.records)} 条，缺失 symbol={out.missing}")
    print()


def test_knowledge_global() -> None:
    print("=" * 64)
    print("2) search_financial_knowledge —— GraphRAG global 检索")
    print("=" * 64)
    tool = SearchFinancialKnowledge()
    out = tool.call({"query": "SpaceX 的核心竞争优势是什么？", "mode": "global"})
    print(out.answer[:600])
    print()


def test_knowledge_local() -> None:
    print("=" * 64)
    print("3) search_financial_knowledge —— GraphRAG local 检索")
    print("=" * 64)
    tool = SearchFinancialKnowledge()
    out = tool.call({"query": "Elon Musk 在 SpaceX 的角色是什么？", "mode": "local"})
    print(out.answer[:600])
    print()


if __name__ == "__main__":
    test_company_profile()
    test_market_data()
    test_knowledge_global()
    test_knowledge_local()
    print("✅ 四个测试都跑完（没抛异常即代表工具链路通了）")
