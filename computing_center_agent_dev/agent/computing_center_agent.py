"""
算力中心运维 CLI Agent - 基于 CodeActAgent 扩展

这是一个专门为高性能计算集群运维设计的 AI Agent。
**完全继承 CodeActAgent 的所有能力**，并在此基础上添加算力中心特有的工具。

继承的能力:
- bash 命令执行
- Python/IPython 代码执行
- 文件编辑 (str_replace_editor)
- 浏览器操作
- 任务规划和跟踪
- 思考工具

新增能力:
- 集群状态监控
- 作业管理 (Slurm/PBS/K8s)
- GPU 监控
- 故障诊断
- 日志分析

作者: OpenHands
版本: 1.0.0
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pydantic import Field

if TYPE_CHECKING:
    from litellm import ChatCompletionToolParam

# 核心导入 - 继承 CodeActAgent
from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.llm.llm_registry import LLMRegistry
from openhands.utils.prompt import PromptManager

# ============================================================================
# 配置类定义
# ============================================================================

class ComputingCenterAgentConfig(AgentConfig):
    """
    算力中心 Agent 配置类

    继承自 AgentConfig，包含所有 CodeActAgent 的配置项，
    并添加算力中心特有的配置。

    继承的配置项:
    - cli_mode: bool - CLI 模式
    - enable_browsing: bool - 浏览器功能
    - enable_jupyter: bool - Jupyter 功能
    - enable_cmd: bool - bash 命令功能
    - enable_think: bool - 思考工具
    - enable_finish: bool - 完成工具
    - enable_editor: bool - 编辑器工具
    - enable_plan_mode: bool - 规划模式
    - ... (更多配置见 AgentConfig)

    新增配置项:
    - cluster_type: 集群类型
    - enable_gpu_monitor: GPU 监控
    - enable_job_manager: 作业管理
    - enable_diagnostic: 诊断工具
    - enable_log_analyzer: 日志分析
    """

    # ========== 集群基础配置 ==========
    cluster_type: str = Field(
        default="slurm",
        description="集群调度系统类型: slurm, pbs, k8s, custom"
    )

    cluster_name: str = Field(
        default="default",
        description="集群名称，用于多集群场景"
    )

    # ========== 算力中心工具开关 ==========
    enable_cluster_monitor: bool = Field(
        default=True,
        description="是否启用集群监控工具"
    )

    enable_gpu_monitor: bool = Field(
        default=True,
        description="是否启用 GPU 监控工具"
    )

    enable_job_manager: bool = Field(
        default=True,
        description="是否启用作业管理工具"
    )

    enable_diagnostic: bool = Field(
        default=True,
        description="是否启用诊断工具"
    )

    enable_log_analyzer: bool = Field(
        default=True,
        description="是否启用日志分析工具"
    )

    enable_resource_manager: bool = Field(
        default=True,
        description="是否启用资源管理工具"
    )

    # ========== 集群连接配置 ==========
    ssh_config_path: str | None = Field(
        default=None,
        description="SSH 配置文件路径，用于连接远程集群"
    )

    head_node: str | None = Field(
        default=None,
        description="集群头节点地址"
    )

    # ========== 显示和限制配置 ==========
    default_partition: str = Field(
        default="default",
        description="默认作业分区名称"
    )

    max_jobs_display: int = Field(
        default=50,
        description="作业列表最大显示数量"
    )

    max_nodes_display: int = Field(
        default=100,
        description="节点列表最大显示数量"
    )

    # ========== GPU 配置 ==========
    gpu_vendor: str = Field(
        default="nvidia",
        description="GPU 厂商: nvidia, amd, intel"
    )

    # ========== 告警阈值配置 ==========
    alert_gpu_util_low: int = Field(
        default=30,
        description="GPU 利用率低于此值告警 (%)"
    )

    alert_gpu_memory_high: int = Field(
        default=90,
        description="GPU 内存使用高于此值告警 (%)"
    )

    alert_cpu_high: int = Field(
        default=95,
        description="CPU 使用率高于此值告警 (%)"
    )

    alert_disk_high: int = Field(
        default=90,
        description="磁盘使用率高于此值告警 (%)"
    )


# ============================================================================
# Agent 主类
# ============================================================================

class ComputingCenterAgent(CodeActAgent):
    """
    算力中心运维 CLI Agent

    **继承自 CodeActAgent**，完全兼容原有 CLI Agent 的所有能力，
    并扩展了算力中心运维专用工具。

    继承的能力 (来自 CodeActAgent):
    ├── bash 命令执行 (execute_bash)
    ├── Python/IPython 代码执行 (execute_ipython_cell)
    ├── 文件编辑 (str_replace_editor)
    ├── 浏览器操作 (browser)
    ├── 思考工具 (think)
    ├── 完成工具 (finish)
    └── 任务跟踪 (task_tracker, 规划模式)

    新增能力:
    ├── cluster_monitor - 集群状态监控
    ├── job_manager - 作业管理 (Slurm/PBS/K8s)
    ├── gpu_monitor - GPU 监控 (NVIDIA/AMD)
    ├── diagnostic - 故障诊断
    ├── log_analyzer - 日志分析
    └── resource_manager - 资源管理

    使用方法:
    1. 配置 config.toml:
        [core]
        default_agent = "ComputingCenterAgent"

        [agent.ComputingCenterAgent]
        cluster_type = "slurm"
        enable_gpu_monitor = true

    2. 运行:
        python -m openhands.core.main -t "查看集群状态"
    """

    VERSION = '1.0.0'

    # 指定使用的配置模型类
    config_model = ComputingCenterAgentConfig

    def __init__(
        self,
        config: ComputingCenterAgentConfig,
        llm_registry: LLMRegistry
    ) -> None:
        """
        初始化算力中心 Agent

        调用父类 (CodeActAgent) 的初始化方法，然后扩展工具列表。

        Args:
            config: Agent 配置对象
            llm_registry: LLM 注册表
        """
        # 调用 CodeActAgent 的初始化
        # 这会自动加载所有 CodeActAgent 的工具
        super().__init__(config, llm_registry)

        # 扩展工具列表 - 添加算力中心专用工具
        computing_tools = self._get_computing_tools()
        self.tools.extend(computing_tools)

        logger.info(
            f'ComputingCenterAgent v{self.VERSION} initialized\n'
            f'  - Cluster type: {config.cluster_type}\n'
            f'  - Base tools (CodeActAgent): {len(self.tools) - len(computing_tools)}\n'
            f'  - Computing tools: {len(computing_tools)}\n'
            f'  - Total tools: {len(self.tools)}'
        )

    @property
    def prompt_manager(self) -> PromptManager:
        """
        获取提示词管理器

        优先使用算力中心专用的提示词模板，
        如果不存在则回退到 CodeActAgent 的默认模板。

        Returns:
            PromptManager: 提示词管理器实例
        """
        if self._prompt_manager is None:
            # 算力中心专用提示词目录
            computing_prompt_dir = os.path.join(
                os.path.dirname(__file__),
                '..',
                'prompts'
            )

            # 检查是否存在专用提示词
            if os.path.exists(computing_prompt_dir):
                self._prompt_manager = PromptManager(
                    prompt_dir=computing_prompt_dir,
                    system_prompt_filename=self.config.resolved_system_prompt_filename,
                )
                logger.debug('Using ComputingCenterAgent prompts')
            else:
                # 回退到 CodeActAgent 的提示词
                return super().prompt_manager

        return self._prompt_manager

    def _get_computing_tools(self) -> list['ChatCompletionToolParam']:
        """
        获取算力中心专用工具

        根据配置加载相应的工具。

        Returns:
            list: 算力中心专用工具列表
        """
        tools = []

        # 导入工具创建函数
        # 注意: 这些导入放在方法内部，避免循环导入问题
        from computing_center_agent_dev.tools.cluster_monitor import (
            create_cluster_monitor_tool
        )
        from computing_center_agent_dev.tools.job_manager import (
            create_job_manager_tool
        )
        from computing_center_agent_dev.tools.gpu_monitor import (
            create_gpu_monitor_tool
        )
        from computing_center_agent_dev.tools.diagnostic import (
            create_diagnostic_tool
        )
        from computing_center_agent_dev.tools.log_analyzer import (
            create_log_analyzer_tool
        )
        from computing_center_agent_dev.tools.resource_manager import (
            create_resource_manager_tool
        )

        # 集群监控工具
        if self.config.enable_cluster_monitor:
            tools.append(create_cluster_monitor_tool(
                cluster_type=self.config.cluster_type
            ))
            logger.debug('Loaded tool: cluster_monitor')

        # 作业管理工具
        if self.config.enable_job_manager:
            tools.append(create_job_manager_tool(
                cluster_type=self.config.cluster_type,
                default_partition=self.config.default_partition
            ))
            logger.debug('Loaded tool: job_manager')

        # GPU 监控工具
        if self.config.enable_gpu_monitor:
            tools.append(create_gpu_monitor_tool(
                gpu_vendor=self.config.gpu_vendor
            ))
            logger.debug('Loaded tool: gpu_monitor')

        # 诊断工具
        if self.config.enable_diagnostic:
            tools.append(create_diagnostic_tool())
            logger.debug('Loaded tool: diagnostic')

        # 日志分析工具
        if self.config.enable_log_analyzer:
            tools.append(create_log_analyzer_tool())
            logger.debug('Loaded tool: log_analyzer')

        # 资源管理工具
        if self.config.enable_resource_manager:
            tools.append(create_resource_manager_tool(
                cluster_type=self.config.cluster_type
            ))
            logger.debug('Loaded tool: resource_manager')

        return tools

    def get_cluster_info(self) -> dict:
        """
        获取当前集群配置信息

        便捷方法，用于获取当前配置的集群信息。

        Returns:
            dict: 集群配置信息
        """
        return {
            'cluster_type': self.config.cluster_type,
            'cluster_name': self.config.cluster_name,
            'head_node': self.config.head_node,
            'default_partition': self.config.default_partition,
            'gpu_vendor': self.config.gpu_vendor,
        }
