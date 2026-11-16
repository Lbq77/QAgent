"""SQLite Checkpoint Saver（Checkpoint 恢复）。

LangGraph 官方 SQLite 后端需要额外安装 `langgraph-checkpoint-sqlite`（依赖 aiosqlite），
本环境离线装不了。这里用标准库 `sqlite3` 自实现一个同步版 `BaseCheckpointSaver`，
存储结构与官方 InMemorySaver 完全对齐：

- checkpoints 表：thread → checkpoint_ns → checkpoint_id → (checkpoint, metadata, parent)
- blobs 表：每个 channel 在某 version 下的序列化值（channel_values）
- writes 表：checkpoint 的待写 pending_writes（interrupt 恢复依赖它）

序列化复用 langgraph 自带的 `JsonPlusSerializer`（`serde.dumps_typed` / `loads_typed`），
因此与 MemorySaver 的语义逐字段一致，可被 `interrupt` / `Command(resume=...)` 正常驱动。
"""
from __future__ import annotations

import random
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
    get_checkpoint_metadata,
    writes_sort_key,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# AgentState 里出现的自定义类型（Pydantic 模型 + 枚举），在 msgpack 反序列化时显式放行，
# 否则 LangGraph 会对「未注册类型」发 future-block 警告。
_STATE_TYPES: list[tuple[str, str]] = [
    ("qagent.state", "TaskPlan"),
    ("qagent.state", "TaskConstraints"),
    ("qagent.state", "RiskReport"),
    ("qagent.state", "FactCheckResult"),
    ("qagent.state", "TraceRecord"),
    ("qagent.state", "TaskStatus"),
    ("qagent.state", "NodeStatus"),
]


def build_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=_STATE_TYPES)


class SqliteSaver(BaseCheckpointSaver[str]):
    """基于标准库 sqlite3 的同步 Checkpoint Saver（断点恢复用）。"""

    def __init__(self, path: str | Any, *, serde: SerializerProtocol | None = None) -> None:
        super().__init__(serde=serde)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                checkpoint_type TEXT NOT NULL,
                checkpoint_blob BLOB NOT NULL,
                metadata_type TEXT NOT NULL,
                metadata_blob BLOB NOT NULL,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            );
            CREATE TABLE IF NOT EXISTS blobs (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL,
                channel TEXT NOT NULL,
                version TEXT NOT NULL,
                type TEXT NOT NULL,
                blob BLOB NOT NULL,
                PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
            );
            CREATE TABLE IF NOT EXISTS writes (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                channel TEXT NOT NULL,
                type TEXT NOT NULL,
                blob BLOB NOT NULL,
                task_path TEXT NOT NULL,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- 内部读取（与 InMemorySaver 逐字段对齐） ----

    def _load_blobs(
        self, thread_id: str, checkpoint_ns: str, versions: ChannelVersions
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        with self._lock:
            for k, ver in versions.items():
                row = self._conn.execute(
                    "SELECT type, blob FROM blobs "
                    "WHERE thread_id=? AND checkpoint_ns=? AND channel=? AND version=?",
                    (thread_id, checkpoint_ns, k, str(ver)),
                ).fetchone()
                if row is None:
                    continue
                typ, blob = row
                if typ == "empty":
                    continue
                result[k] = self.serde.loads_typed((typ, blob))
        return result

    def _ordered_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> list[tuple[str, str, tuple[str, bytes], str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT task_id, idx, channel, type, blob, task_path FROM writes "
                "WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?",
                (thread_id, checkpoint_ns, checkpoint_id),
            ).fetchall()
        stored = {
            (task_id, idx): (task_id, channel, (typ, blob), task_path)
            for task_id, idx, channel, typ, blob, task_path in rows
        }
        return [
            stored[k]
            for k in sorted(stored, key=lambda k: writes_sort_key(stored[k][3], *k))
        ]

    # ---- BaseCheckpointSaver 接口 ----

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")

        if checkpoint_id := get_checkpoint_id(config):
            with self._lock:
                row = self._conn.execute(
                    "SELECT checkpoint_id, parent_checkpoint_id, checkpoint_type, "
                    "checkpoint_blob, metadata_type, metadata_blob FROM checkpoints "
                    "WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?",
                    (thread_id, checkpoint_ns, checkpoint_id),
                ).fetchone()
            if row is None:
                return None
        else:
            with self._lock:
                row = self._conn.execute(
                    "SELECT checkpoint_id, parent_checkpoint_id, checkpoint_type, "
                    "checkpoint_blob, metadata_type, metadata_blob FROM checkpoints "
                    "WHERE thread_id=? AND checkpoint_ns=? ORDER BY checkpoint_id DESC LIMIT 1",
                    (thread_id, checkpoint_ns),
                ).fetchone()
            if row is None:
                return None
            checkpoint_id = row[0]

        _, parent_id, c_type, c_blob, m_type, m_blob = row
        checkpoint_: Checkpoint = self.serde.loads_typed((c_type, c_blob))
        metadata: CheckpointMetadata = self.serde.loads_typed((m_type, m_blob))
        writes = self._ordered_writes(thread_id, checkpoint_ns, checkpoint_id)
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint={
                **checkpoint_,
                "channel_values": self._load_blobs(
                    thread_id, checkpoint_ns, checkpoint_["channel_versions"]
                ),
            },
            metadata=metadata,
            pending_writes=[
                (tid, ch, self.serde.loads_typed((typ, blob)))
                for tid, ch, (typ, blob), _ in writes
            ],
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_id,
                    }
                }
                if parent_id
                else None
            ),
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        with self._lock:
            if config:
                thread_ids = (config["configurable"]["thread_id"],)
                config_checkpoint_ns = config["configurable"].get("checkpoint_ns")
                config_checkpoint_id = get_checkpoint_id(config)
            else:
                thread_ids = tuple(
                    r[0]
                    for r in self._conn.execute(
                        "SELECT DISTINCT thread_id FROM checkpoints"
                    ).fetchall()
                )
                config_checkpoint_ns = None
                config_checkpoint_id = None

            for thread_id in thread_ids:
                ns_rows = self._conn.execute(
                    "SELECT DISTINCT checkpoint_ns FROM checkpoints WHERE thread_id=?",
                    (thread_id,),
                ).fetchall()
                for (checkpoint_ns,) in ns_rows:
                    if (
                        config_checkpoint_ns is not None
                        and checkpoint_ns != config_checkpoint_ns
                    ):
                        continue
                    rows = self._conn.execute(
                        "SELECT checkpoint_id, parent_checkpoint_id, checkpoint_type, "
                        "checkpoint_blob, metadata_type, metadata_blob FROM checkpoints "
                        "WHERE thread_id=? AND checkpoint_ns=? ORDER BY checkpoint_id DESC",
                        (thread_id, checkpoint_ns),
                    ).fetchall()
                    for row in rows:
                        checkpoint_id, parent_id, c_type, c_blob, m_type, m_blob = row
                        if config_checkpoint_id and checkpoint_id != config_checkpoint_id:
                            continue
                        if (
                            before
                            and (before_checkpoint_id := get_checkpoint_id(before))
                            and checkpoint_id >= before_checkpoint_id
                        ):
                            continue
                        metadata = self.serde.loads_typed((m_type, m_blob))
                        if filter and not all(
                            query_value == metadata.get(query_key)
                            for query_key, query_value in filter.items()
                        ):
                            continue
                        if limit is not None and limit <= 0:
                            return
                        elif limit is not None:
                            limit -= 1
                        checkpoint_ = self.serde.loads_typed((c_type, c_blob))
                        writes = self._ordered_writes(
                            thread_id, checkpoint_ns, checkpoint_id
                        )
                        yield CheckpointTuple(
                            config={
                                "configurable": {
                                    "thread_id": thread_id,
                                    "checkpoint_ns": checkpoint_ns,
                                    "checkpoint_id": checkpoint_id,
                                }
                            },
                            checkpoint={
                                **checkpoint_,
                                "channel_values": self._load_blobs(
                                    thread_id,
                                    checkpoint_ns,
                                    checkpoint_["channel_versions"],
                                ),
                            },
                            metadata=metadata,
                            parent_config=(
                                {
                                    "configurable": {
                                        "thread_id": thread_id,
                                        "checkpoint_ns": checkpoint_ns,
                                        "checkpoint_id": parent_id,
                                    }
                                }
                                if parent_id
                                else None
                            ),
                            pending_writes=[
                                (tid, ch, self.serde.loads_typed((typ, blob)))
                                for tid, ch, (typ, blob), _ in writes
                            ],
                        )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        c = checkpoint.copy()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        values: dict[str, Any] = c.pop("channel_values")  # type: ignore[misc]

        with self._lock:
            for k, v in new_versions.items():
                if k in values:
                    typ, blob = self.serde.dumps_typed(values[k])
                else:
                    typ, blob = "empty", b""
                self._conn.execute(
                    "INSERT OR REPLACE INTO blobs "
                    "(thread_id, checkpoint_ns, channel, version, type, blob) "
                    "VALUES (?,?,?,?,?,?)",
                    (thread_id, checkpoint_ns, k, str(v), typ, blob),
                )

            c_type, c_blob = self.serde.dumps_typed(c)
            m_type, m_blob = self.serde.dumps_typed(
                get_checkpoint_metadata(config, metadata)
            )
            parent = config["configurable"].get("checkpoint_id")
            self._conn.execute(
                "INSERT OR REPLACE INTO checkpoints "
                "(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, "
                "checkpoint_type, checkpoint_blob, metadata_type, metadata_blob) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint["id"],
                    parent,
                    c_type,
                    c_blob,
                    m_type,
                    m_blob,
                ),
            )
            self._conn.commit()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        with self._lock:
            for idx, (ch, v) in enumerate(writes):
                write_idx = WRITES_IDX_MAP.get(ch, idx)
                if write_idx >= 0:
                    existing = self._conn.execute(
                        "SELECT 1 FROM writes WHERE thread_id=? AND checkpoint_ns=? "
                        "AND checkpoint_id=? AND task_id=? AND idx=?",
                        (thread_id, checkpoint_ns, checkpoint_id, task_id, write_idx),
                    ).fetchone()
                    if existing is not None:
                        continue
                typ, blob = self.serde.dumps_typed(v)
                self._conn.execute(
                    "INSERT OR REPLACE INTO writes "
                    "(thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, "
                    "type, blob, task_path) VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        task_id,
                        write_idx,
                        ch,
                        typ,
                        blob,
                        task_path,
                    ),
                )
            self._conn.commit()

    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM checkpoints WHERE thread_id=?", (thread_id,)
            )
            self._conn.execute("DELETE FROM blobs WHERE thread_id=?", (thread_id,))
            self._conn.execute("DELETE FROM writes WHERE thread_id=?", (thread_id,))
            self._conn.commit()

    def get_next_version(self, current: str | None, channel: None) -> str:
        # 与 InMemorySaver 一致的单调递增字符串版本
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"


def build_checkpointer(settings) -> BaseCheckpointSaver:
    """按配置返回 checkpointer：sqlite（持久化）或 memory（测试/单进程）。"""
    serde = build_serde()
    if settings.checkpoint_backend == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver(serde=serde)
    db = settings.checkpoint_db
    db.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(str(db), serde=serde)
