"""结果外置存储（SQLite ResultStore）+ 实验记录。"""
from __future__ import annotations

from qagent.backtest import clear_results, load_result, save_result
from qagent.store import ResultStore, list_experiments


def test_store_roundtrip(tmp_path):
    store = ResultStore(tmp_path / "r.sqlite")
    store.put("bt_1", {"metrics": {"annual_return": 0.1}, "engine_version": "v1"})
    assert store.get("bt_1")["metrics"]["annual_return"] == 0.1
    # 同 ref 覆盖
    store.put("bt_1", {"metrics": {"annual_return": 0.2}})
    assert store.get("bt_1")["metrics"]["annual_return"] == 0.2
    assert store.list()[0]["ref"] == "bt_1"
    store.close()


def test_store_persists_across_instances(tmp_path):
    """新实例（模拟新进程）读同一文件，能取回旧结果——这是跨进程可复现的地基。"""
    path = tmp_path / "r.sqlite"
    s1 = ResultStore(path)
    s1.put("bt_x", {"selected": ["AAPL"]})
    s1.close()

    s2 = ResultStore(path)
    assert s2.get("bt_x")["selected"] == ["AAPL"]
    s2.close()


def test_clear_results():
    store = ResultStore()
    store.put("a", {"x": 1})
    store.clear()
    assert store.get("a") is None
    store.close()


def test_save_load_and_experiments():
    clear_results()
    save_result("bt_exp", {"metrics": {"sharpe": 1.5}, "config": {"top_n": 3}})
    assert load_result("bt_exp")["metrics"]["sharpe"] == 1.5

    exps = list_experiments()
    assert any(e["ref"] == "bt_exp" for e in exps)
    # 实验记录自动附带引擎版本与时间戳
    assert exps[0]["engine_version"]
    assert exps[0]["created_at"]
