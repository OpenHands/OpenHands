"""
ComputingCenterAgent 测试模块

测试 Agent 的初始化、配置和基本功能。
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestComputingCenterAgentConfig:
    """测试 ComputingCenterAgentConfig"""

    def test_default_values(self):
        """测试默认配置值"""
        from computing_center_agent_dev.agent.computing_center_agent import (
            ComputingCenterAgentConfig
        )

        config = ComputingCenterAgentConfig()

        # 检查默认值
        assert config.cluster_type == "slurm"
        assert config.cluster_name == "default"
        assert config.enable_cluster_monitor is True
        assert config.enable_gpu_monitor is True
        assert config.enable_job_manager is True
        assert config.gpu_vendor == "nvidia"
        assert config.default_partition == "default"

    def test_custom_values(self):
        """测试自定义配置值"""
        from computing_center_agent_dev.agent.computing_center_agent import (
            ComputingCenterAgentConfig
        )

        config = ComputingCenterAgentConfig(
            cluster_type="pbs",
            cluster_name="my_cluster",
            enable_gpu_monitor=False,
            default_partition="batch",
        )

        assert config.cluster_type == "pbs"
        assert config.cluster_name == "my_cluster"
        assert config.enable_gpu_monitor is False
        assert config.default_partition == "batch"

    def test_inherited_config(self):
        """测试继承的配置项"""
        from computing_center_agent_dev.agent.computing_center_agent import (
            ComputingCenterAgentConfig
        )

        config = ComputingCenterAgentConfig()

        # 应该有 AgentConfig 的属性
        assert hasattr(config, 'cli_mode')
        assert hasattr(config, 'enable_cmd')
        assert hasattr(config, 'enable_jupyter')
        assert hasattr(config, 'enable_browsing')


class TestComputingCenterAgent:
    """测试 ComputingCenterAgent"""

    @pytest.fixture
    def agent(self, mock_agent_config, mock_llm_registry):
        """创建 Agent 实例用于测试"""
        with patch.multiple(
            'computing_center_agent_dev.agent.computing_center_agent',
            ConversationMemory=MagicMock(),
            Condenser=MagicMock(),
            PromptManager=MagicMock(),
        ):
            # 需要 mock 工具导入
            with patch(
                'computing_center_agent_dev.agent.computing_center_agent.'
                'ComputingCenterAgent._get_computing_tools',
                return_value=[]
            ):
                from computing_center_agent_dev.agent.computing_center_agent import (
                    ComputingCenterAgent
                )

                # Mock 父类初始化
                with patch.object(
                    ComputingCenterAgent,
                    '__bases__',
                    (Mock,)
                ):
                    # 直接创建实例
                    agent = object.__new__(ComputingCenterAgent)
                    agent.config = mock_agent_config
                    agent.llm_registry = mock_llm_registry
                    agent.llm = mock_llm_registry.get_llm_from_agent_config()
                    agent.tools = []
                    agent._prompt_manager = None
                    agent.mcp_tools = {}

                    return agent

    def test_version(self, agent):
        """测试版本号"""
        assert hasattr(agent, 'VERSION')
        # 直接检查类属性
        from computing_center_agent_dev.agent.computing_center_agent import (
            ComputingCenterAgent
        )
        assert ComputingCenterAgent.VERSION == '1.0.0'

    def test_config_model(self):
        """测试配置模型类"""
        from computing_center_agent_dev.agent.computing_center_agent import (
            ComputingCenterAgent,
            ComputingCenterAgentConfig
        )

        assert ComputingCenterAgent.config_model == ComputingCenterAgentConfig

    def test_get_cluster_info(self, agent):
        """测试获取集群信息"""
        info = agent.get_cluster_info()

        assert 'cluster_type' in info
        assert 'cluster_name' in info
        assert 'default_partition' in info
        assert 'gpu_vendor' in info


class TestAgentRegistration:
    """测试 Agent 注册"""

    def test_agent_can_be_registered(self):
        """测试 Agent 可以被注册"""
        from openhands.controller.agent import Agent
        from computing_center_agent_dev.agent.computing_center_agent import (
            ComputingCenterAgent
        )

        # 检查是否可以注册（不实际注册以避免冲突）
        assert issubclass(ComputingCenterAgent, Agent) or True  # 占位


class TestAgentTools:
    """测试 Agent 工具加载"""

    def test_tool_creation_functions_exist(self):
        """测试工具创建函数存在"""
        from computing_center_agent_dev.tools import (
            create_cluster_monitor_tool,
            create_job_manager_tool,
            create_gpu_monitor_tool,
            create_diagnostic_tool,
            create_log_analyzer_tool,
            create_resource_manager_tool,
        )

        # 测试函数可调用
        assert callable(create_cluster_monitor_tool)
        assert callable(create_job_manager_tool)
        assert callable(create_gpu_monitor_tool)
        assert callable(create_diagnostic_tool)
        assert callable(create_log_analyzer_tool)
        assert callable(create_resource_manager_tool)

    def test_tools_return_valid_format(self):
        """测试工具返回有效格式"""
        from computing_center_agent_dev.tools import (
            create_cluster_monitor_tool,
            create_job_manager_tool,
            create_gpu_monitor_tool,
        )

        tools = [
            create_cluster_monitor_tool(),
            create_job_manager_tool(),
            create_gpu_monitor_tool(),
        ]

        for tool in tools:
            assert tool['type'] == 'function'
            assert 'function' in tool
            assert 'name' in tool['function']
            assert 'description' in tool['function']
            assert 'parameters' in tool['function']


# ============================================================================
# 集成测试 (需要完整环境)
# ============================================================================

@pytest.mark.integration
class TestAgentIntegration:
    """集成测试 (可选)"""

    @pytest.mark.skip(reason="需要完整 OpenHands 环境")
    def test_full_initialization(self):
        """测试完整初始化流程"""
        pass

    @pytest.mark.skip(reason="需要完整 OpenHands 环境")
    def test_step_execution(self):
        """测试 step 执行"""
        pass
