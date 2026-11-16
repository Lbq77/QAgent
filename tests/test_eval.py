"""测试集 v1 的 pytest 入口：确保 evals/cases.json 全部通过。"""
from __future__ import annotations

import sys
from pathlib import Path

# evals/ 在仓库根目录，不在 src 下；插入根目录让 `evals` 可导入
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.run_eval import run_all  # noqa: E402


def test_eval_cases_all_pass():
    results = run_all()
    failed = [(cid, f) for cid, ok, f in results if not ok]
    assert not failed, f"eval cases failed: {failed}"
