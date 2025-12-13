"""
测试配置和共享 Fixtures

这个文件包含所有测试共享的 fixtures 和配置。
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Any


# ============================================================================
# Mock 配置 Fixtures
# ============================================================================

@pytest.fixture
def mock_agent_config() -> Mock:
    """创建模拟的 Agent 配置"""
    config = Mock()

    # 继承的配置 (CodeActAgent)
    config.cli_mode = False
    config.enable_browsing = False
    config.enable_jupyter = True
    config.enable_cmd = True
    config.enable_think = True
    config.enable_finish = True
    config.enable_editor = True
    config.enable_plan_mode = True
    config.resolved_system_prompt_filename = "system_prompt.j2"
    config.condenser = Mock()
    config.runtime = "docker"

    # 算力中心配置
    config.cluster_type = "slurm"
    config.cluster_name = "test_cluster"
    config.enable_cluster_monitor = True
    config.enable_gpu_monitor = True
    config.enable_job_manager = True
    config.enable_diagnostic = True
    config.enable_log_analyzer = True
    config.enable_resource_manager = True
    config.default_partition = "gpu"
    config.max_jobs_display = 50
    config.gpu_vendor = "nvidia"
    config.ssh_config_path = None
    config.head_node = None

    # 告警阈值
    config.alert_gpu_util_low = 30
    config.alert_gpu_memory_high = 90
    config.alert_cpu_high = 95
    config.alert_disk_high = 90

    return config


@pytest.fixture
def mock_llm_registry() -> MagicMock:
    """创建模拟的 LLM 注册表"""
    registry = MagicMock()

    # 模拟 LLM 实例
    mock_llm = MagicMock()
    mock_llm.config = Mock()
    mock_llm.config.model = "gpt-4o"
    mock_llm.config.max_message_chars = 10000
    mock_llm.vision_is_active.return_value = False
    mock_llm.is_caching_prompt_active.return_value = False

    registry.get_llm_from_agent_config.return_value = mock_llm
    registry.get_router.return_value = mock_llm

    return registry


@pytest.fixture
def mock_llm(mock_llm_registry) -> MagicMock:
    """获取模拟的 LLM 实例"""
    return mock_llm_registry.get_llm_from_agent_config()


# ============================================================================
# Mock 状态 Fixtures
# ============================================================================

@pytest.fixture
def mock_state() -> Mock:
    """创建模拟的 Agent 状态"""
    state = Mock()
    state.history = []
    state.get_last_user_message.return_value = None
    state.to_llm_metadata.return_value = {}
    return state


@pytest.fixture
def mock_message_action() -> Mock:
    """创建模拟的消息动作"""
    action = Mock()
    action.content = "Test message"
    action.source = "user"
    return action


# ============================================================================
# 工具测试 Fixtures
# ============================================================================

@pytest.fixture
def slurm_config() -> dict:
    """Slurm 集群配置"""
    return {
        'cluster_type': 'slurm',
        'default_partition': 'gpu',
    }


@pytest.fixture
def pbs_config() -> dict:
    """PBS 集群配置"""
    return {
        'cluster_type': 'pbs',
        'default_partition': 'batch',
    }


@pytest.fixture
def k8s_config() -> dict:
    """Kubernetes 集群配置"""
    return {
        'cluster_type': 'k8s',
        'default_partition': 'default',
    }


# ============================================================================
# 命令输出 Fixtures
# ============================================================================

@pytest.fixture
def sample_sinfo_output() -> str:
    """示例 sinfo 输出"""
    return """PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
cpu*         up   infinite     10   idle node[01-10]
gpu          up 7-00:00:00      4   idle gpu[01-04]
gpu          up 7-00:00:00      2  alloc gpu[05-06]
"""


@pytest.fixture
def sample_squeue_output() -> str:
    """示例 squeue 输出"""
    return """JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
12345       gpu   train    user1  R    2:30:45      1 gpu01
12346       gpu  infer    user2 PD       0:00      2 (Resources)
"""


@pytest.fixture
def sample_nvidia_smi_output() -> str:
    """示例 nvidia-smi 输出"""
    return """index, name, utilization.gpu [%], memory.used [MiB], memory.total [MiB]
0, NVIDIA A100, 85, 40000, 81920
1, NVIDIA A100, 92, 75000, 81920
"""


# ============================================================================
# 测试工具函数
# ============================================================================

def create_mock_tool_response(name: str, parameters: dict) -> dict:
    """创建模拟的工具调用响应"""
    return {
        'function': {
            'name': name,
            'parameters': parameters,
        }
    }


def create_mock_llm_response(content: str = "", tool_calls: list = None) -> Mock:
    """创建模拟的 LLM 响应"""
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = tool_calls or []
    return response
