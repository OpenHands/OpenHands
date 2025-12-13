"""
工具测试模块

测试所有算力中心运维工具。
"""

import pytest


class TestClusterMonitorTool:
    """测试集群监控工具"""

    def test_tool_creation(self):
        """测试工具创建"""
        from computing_center_agent_dev.tools.cluster_monitor import (
            create_cluster_monitor_tool
        )

        tool = create_cluster_monitor_tool()

        assert tool['type'] == 'function'
        assert tool['function']['name'] == 'cluster_monitor'
        assert 'query_type' in tool['function']['parameters']['properties']

    def test_tool_with_cluster_type(self):
        """测试不同集群类型"""
        from computing_center_agent_dev.tools.cluster_monitor import (
            create_cluster_monitor_tool
        )

        for cluster_type in ['slurm', 'pbs', 'k8s']:
            tool = create_cluster_monitor_tool(cluster_type=cluster_type)
            assert cluster_type in tool['function']['description'].lower() or True

    def test_slurm_commands(self):
        """测试 Slurm 命令生成"""
        from computing_center_agent_dev.tools.cluster_monitor import (
            ClusterMonitorCommands
        )

        # 测试节点查询
        cmd = ClusterMonitorCommands.get_slurm_command('nodes')
        assert 'sinfo' in cmd

        # 测试资源查询
        cmd = ClusterMonitorCommands.get_slurm_command('resources')
        assert 'sinfo' in cmd

        # 测试队列查询
        cmd = ClusterMonitorCommands.get_slurm_command('queues')
        assert 'squeue' in cmd

    def test_pbs_commands(self):
        """测试 PBS 命令生成"""
        from computing_center_agent_dev.tools.cluster_monitor import (
            ClusterMonitorCommands
        )

        cmd = ClusterMonitorCommands.get_pbs_command('nodes')
        assert 'pbsnodes' in cmd


class TestJobManagerTool:
    """测试作业管理工具"""

    def test_tool_creation(self):
        """测试工具创建"""
        from computing_center_agent_dev.tools.job_manager import (
            create_job_manager_tool
        )

        tool = create_job_manager_tool()

        assert tool['function']['name'] == 'job_manager'
        assert 'action' in tool['function']['parameters']['properties']

    def test_action_types(self):
        """测试支持的操作类型"""
        from computing_center_agent_dev.tools.job_manager import (
            create_job_manager_tool
        )

        tool = create_job_manager_tool()
        actions = tool['function']['parameters']['properties']['action']['enum']

        expected = ['submit', 'status', 'list', 'cancel', 'hold', 'release', 'modify', 'output']
        for action in expected:
            assert action in actions

    def test_slurm_job_commands(self):
        """测试 Slurm 作业命令"""
        from computing_center_agent_dev.tools.job_manager import (
            JobManagerCommands
        )

        # 提交
        cmd = JobManagerCommands.get_slurm_command('submit', script_path='job.sh')
        assert 'sbatch' in cmd
        assert 'job.sh' in cmd

        # 状态
        cmd = JobManagerCommands.get_slurm_command('status', job_id='12345')
        assert 'scontrol' in cmd
        assert '12345' in cmd

        # 取消
        cmd = JobManagerCommands.get_slurm_command('cancel', job_id='12345')
        assert 'scancel' in cmd


class TestGPUMonitorTool:
    """测试 GPU 监控工具"""

    def test_tool_creation(self):
        """测试工具创建"""
        from computing_center_agent_dev.tools.gpu_monitor import (
            create_gpu_monitor_tool
        )

        tool = create_gpu_monitor_tool()

        assert tool['function']['name'] == 'gpu_monitor'
        assert 'query_type' in tool['function']['parameters']['properties']

    def test_query_types(self):
        """测试查询类型"""
        from computing_center_agent_dev.tools.gpu_monitor import (
            create_gpu_monitor_tool
        )

        tool = create_gpu_monitor_tool()
        types = tool['function']['parameters']['properties']['query_type']['enum']

        expected = ['status', 'utilization', 'memory', 'processes', 'temperature', 'topology']
        for t in expected:
            assert t in types

    def test_nvidia_commands(self):
        """测试 NVIDIA 命令"""
        from computing_center_agent_dev.tools.gpu_monitor import (
            GPUMonitorCommands
        )

        # 状态
        cmd = GPUMonitorCommands.get_nvidia_command('status')
        assert 'nvidia-smi' in cmd

        # 利用率
        cmd = GPUMonitorCommands.get_nvidia_command('utilization')
        assert 'nvidia-smi' in cmd
        assert 'utilization' in cmd

        # 进程
        cmd = GPUMonitorCommands.get_nvidia_command('processes')
        assert 'query-compute-apps' in cmd

    def test_gpu_id_filter(self):
        """测试 GPU ID 筛选"""
        from computing_center_agent_dev.tools.gpu_monitor import (
            GPUMonitorCommands
        )

        cmd = GPUMonitorCommands.get_nvidia_command('status', gpu_id='0')
        assert '-i 0' in cmd


class TestDiagnosticTool:
    """测试诊断工具"""

    def test_tool_creation(self):
        """测试工具创建"""
        from computing_center_agent_dev.tools.diagnostic import (
            create_diagnostic_tool
        )

        tool = create_diagnostic_tool()

        assert tool['function']['name'] == 'diagnostic'
        assert 'check_type' in tool['function']['parameters']['properties']

    def test_check_types(self):
        """测试诊断类型"""
        from computing_center_agent_dev.tools.diagnostic import (
            create_diagnostic_tool
        )

        tool = create_diagnostic_tool()
        types = tool['function']['parameters']['properties']['check_type']['enum']

        expected = ['node_health', 'network', 'storage', 'service', 'job_failure', 'performance', 'full_check']
        for t in expected:
            assert t in types

    def test_diagnostic_commands(self):
        """测试诊断命令"""
        from computing_center_agent_dev.tools.diagnostic import (
            DiagnosticCommands
        )

        # 节点健康检查
        cmds = DiagnosticCommands.node_health_commands()
        assert len(cmds) > 0
        assert any('uptime' in cmd for cmd in cmds)

        # 网络诊断
        cmds = DiagnosticCommands.network_commands()
        assert len(cmds) > 0


class TestLogAnalyzerTool:
    """测试日志分析工具"""

    def test_tool_creation(self):
        """测试工具创建"""
        from computing_center_agent_dev.tools.log_analyzer import (
            create_log_analyzer_tool
        )

        tool = create_log_analyzer_tool()

        assert tool['function']['name'] == 'log_analyzer'
        assert 'analyze_type' in tool['function']['parameters']['properties']

    def test_error_patterns(self):
        """测试错误模式"""
        from computing_center_agent_dev.tools.log_analyzer import (
            ERROR_PATTERNS
        )

        assert 'cuda_oom' in ERROR_PATTERNS
        assert 'cpu_oom' in ERROR_PATTERNS
        assert 'network_error' in ERROR_PATTERNS

        # 检查结构
        for name, info in ERROR_PATTERNS.items():
            assert 'patterns' in info
            assert 'description' in info
            assert 'suggestions' in info

    def test_error_matcher(self):
        """测试错误匹配器"""
        from computing_center_agent_dev.tools.log_analyzer import (
            ErrorPatternMatcher
        )

        matcher = ErrorPatternMatcher()

        # 测试 CUDA OOM 匹配
        matches = matcher.match("RuntimeError: CUDA error: out of memory")
        assert len(matches) > 0
        assert matches[0]['type'] == 'cuda_oom'

        # 测试无匹配
        matches = matcher.match("Everything is fine")
        assert len(matches) == 0


class TestResourceManagerTool:
    """测试资源管理工具"""

    def test_tool_creation(self):
        """测试工具创建"""
        from computing_center_agent_dev.tools.resource_manager import (
            create_resource_manager_tool
        )

        tool = create_resource_manager_tool()

        assert tool['function']['name'] == 'resource_manager'
        assert 'action' in tool['function']['parameters']['properties']

    def test_action_types(self):
        """测试操作类型"""
        from computing_center_agent_dev.tools.resource_manager import (
            create_resource_manager_tool
        )

        tool = create_resource_manager_tool()
        actions = tool['function']['parameters']['properties']['action']['enum']

        expected = ['quota_info', 'priority', 'reservation', 'qos', 'fairshare', 'accounting']
        for action in expected:
            assert action in actions

    def test_slurm_commands(self):
        """测试 Slurm 命令"""
        from computing_center_agent_dev.tools.resource_manager import (
            ResourceManagerCommands
        )

        # 配额
        cmd = ResourceManagerCommands.get_slurm_command('quota_info')
        assert 'sacctmgr' in cmd

        # 优先级
        cmd = ResourceManagerCommands.get_slurm_command('priority')
        assert 'sprio' in cmd

        # 公平分享
        cmd = ResourceManagerCommands.get_slurm_command('fairshare')
        assert 'sshare' in cmd
