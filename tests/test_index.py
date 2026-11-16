"""指数识别单元测试。"""
from qagent.agents.planner import detect_index


def test_detect_sp500():
    assert detect_index("在标普500中使用小市值因子") == "标普500"


def test_detect_nasdaq():
    assert detect_index("纳指100选股") == "纳斯达克100"


def test_detect_english():
    assert detect_index("use S&P 500") == "标普500"


def test_no_index():
    assert detect_index("AAPL MSFT 动量") == ""
