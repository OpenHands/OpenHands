"""
算力中心运维 CLI Agent 模块

基于 CodeActAgent 扩展，完全兼容原有能力。

部署步骤:
=========
1. 复制整个 computing_center_agent_dev 目录到 openhands/agenthub/

2. 修改 openhands/agenthub/__init__.py，添加导入:
   from openhands.agenthub import computing_center_agent

3. 配置 config.toml:
   [core]
   default_agent = "ComputingCenterAgent"

   [agent.ComputingCenterAgent]
   cluster_type = "slurm"
   enable_gpu_monitor = true

4. 运行:
   python -m openhands.core.main -t "查看集群状态"
"""

from openhands.controller.agent import Agent

# 导入 Agent 类
from computing_center_agent_dev.agent.computing_center_agent import (
    ComputingCenterAgent,
    ComputingCenterAgentConfig,
)

# 注册 Agent
Agent.register('ComputingCenterAgent', ComputingCenterAgent)

__all__ = [
    'ComputingCenterAgent',
    'ComputingCenterAgentConfig',
]
