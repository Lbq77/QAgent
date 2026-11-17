"""回测结果 / 实验记录存储（上下文外置）。

回测产物存在进程内 dict（`backtest._RESULTS`），进程一退就丢、跨进程取不到。
换成 SQLite 持久化的 `ResultStore`：State 只存 ref，完整结果按 ref 存库里，
新进程也能 `load_result(ref)` 取回——这正是 的「State 只存摘要与引用 ID，完整结果外置」。

- `put(ref, payload)` / `get(ref)` / `list()` / `clear()`
- payload 必须是 JSON 可序列化的 dict（metrics / config / selected / data_snapshot / error）
- 每个实验附带 created_at + engine_version，天然是一份「实验记录」日志
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class ResultStore:
    """SQLite 回测结果 / 实验记录存储。"""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                ref TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def put(self, ref: str, payload: dict[str, Any]) -> None:
        created_at = payload.get("created_at") or _now_iso()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO results (ref, payload, created_at) VALUES (?,?,?)",
                (ref, json.dumps(payload, ensure_ascii=False, default=str), created_at),
            )
            self._conn.commit()

    def get(self, ref: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM results WHERE ref=?", (ref,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def list(self) -> list[dict[str, Any]]:
        """全部实验记录（含 ref / created_at），按创建时间倒序。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT ref, payload, created_at FROM results ORDER BY created_at DESC, ref DESC"
            ).fetchall()
        out = []
        for ref, payload, created_at in rows:
            rec = json.loads(payload)
            rec.setdefault("ref", ref)
            rec.setdefault("created_at", created_at)
            out.append(rec)
        return out

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM results")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# ---- 模块级默认 store（进程内共享；生产由 build_context 用 settings.results_db 配置）----

_store: ResultStore | None = None


def configure_result_store(path: str | Path | None = None) -> ResultStore:
    """配置全局结果存储路径（生产：settings.results_db；测试：可传 tmp 文件或 :memory）。"""
    global _store
    _store = ResultStore(path or ":memory:")
    return _store


def get_result_store() -> ResultStore:
    global _store
    if _store is None:
        _store = ResultStore(":memory:")
    return _store


def list_experiments() -> list[dict[str, Any]]:
    """实验记录：所有回测运行的配置 + 指标 + 时间戳。"""
    return get_result_store().list()


def save_market_snapshot(snapshot_id: str, payload: dict[str, Any]) -> None:
    """保存行情快照（records 已序列化为 dict 列表，keyed by snapshot_id）。"""
    get_result_store().put(snapshot_id, payload)


def load_market_snapshot(snapshot_id: str) -> dict[str, Any] | None:
    """按 snapshot_id 取回行情快照；不存在返回 None。"""
    return get_result_store().get(snapshot_id)
