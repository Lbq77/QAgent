"""测试全局配置：默认用内存 checkpointer，避免测试写脏仓库里的 data/。

SQLite checkpointer 的持久化/恢复行为在 test_checkpoint.py 里用 tmp_path 单独验证。
"""
import os

# 在任何 Settings 实例化之前生效（pydantic-settings 读 QAGENT_ 前缀环境变量）。
os.environ["QAGENT_CHECKPOINT_BACKEND"] = "memory"
