"""
故障诊断工具

提供系统和集群的故障诊断功能。

主要功能:
- 节点健康检查
- 网络连通性测试
- 存储系统检查
- 服务状态检查
- 性能瓶颈分析
"""

from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

# ============================================================================
# 工具描述
# ============================================================================

DIAGNOSTIC_DESCRIPTION = """诊断算力集群的各类故障和问题。

### 功能概述
提供全面的故障诊断能力，帮助快速定位和解决集群问题。

### 诊断类型

**node_health (节点健康)**
- 检查节点是否可达
- 检查关键服务状态
- 检查资源是否耗尽

**network (网络诊断)**
- 节点间网络连通性
- InfiniBand/高速网络状态
- DNS 解析检查

**storage (存储诊断)**
- 文件系统挂载状态
- 磁盘空间检查
- I/O 性能测试
- NFS/Lustre/GPFS 状态

**service (服务诊断)**
- 调度器服务状态
- 监控服务状态
- 数据库连接检查

**job_failure (作业故障)**
- 分析作业失败原因
- 检查资源限制
- 查看相关日志

**performance (性能分析)**
- CPU 性能测试
- 内存带宽测试
- GPU 性能测试
- 网络带宽测试

**full_check (完整检查)**
- 运行所有诊断项目
- 生成诊断报告

### 示例用法
1. 检查节点健康: check_type="node_health", target="node01"
2. 诊断作业失败: check_type="job_failure", job_id="12345"
3. 完整系统检查: check_type="full_check"
"""

# ============================================================================
# 参数定义
# ============================================================================

DIAGNOSTIC_PARAMETERS = {
    'type': 'object',
    'properties': {
        'check_type': {
            'type': 'string',
            'enum': ['node_health', 'network', 'storage', 'service', 'job_failure', 'performance', 'full_check'],
            'description': '诊断类型'
        },
        'target': {
            'type': 'string',
            'description': '(可选) 诊断目标，如节点名称'
        },
        'job_id': {
            'type': 'string',
            'description': '(可选) 作业 ID (job_failure 诊断需要)'
        },
        'verbose': {
            'type': 'boolean',
            'description': '(可选) 是否输出详细信息'
        },
        'timeout': {
            'type': 'integer',
            'description': '(可选) 诊断超时时间 (秒)'
        },
    },
    'required': ['check_type'],
}

# ============================================================================
# 工具创建函数
# ============================================================================

def create_diagnostic_tool(
    use_short_description: bool = False
) -> ChatCompletionToolParam:
    """
    创建诊断工具

    Returns:
        ChatCompletionToolParam: 工具定义
    """
    return ChatCompletionToolParam(
        type='function',
        function=ChatCompletionToolParamFunctionChunk(
            name='diagnostic',
            description=DIAGNOSTIC_DESCRIPTION if not use_short_description else '诊断集群故障和问题',
            parameters=DIAGNOSTIC_PARAMETERS,
        ),
    )


# ============================================================================
# 诊断命令集
# ============================================================================

class DiagnosticCommands:
    """诊断命令集合"""

    @staticmethod
    def node_health_commands(node: str = "") -> list[str]:
        """节点健康检查命令"""
        target = f"ssh {node}" if node else ""
        return [
            f'{target} uptime',
            f'{target} free -h',
            f'{target} df -h',
            f'{target} systemctl is-active slurmd 2>/dev/null || echo "slurmd not found"',
            f'{target} nvidia-smi -L 2>/dev/null || echo "No NVIDIA GPU"',
        ]

    @staticmethod
    def network_commands(target: str = "") -> list[str]:
        """网络诊断命令"""
        return [
            f'ping -c 3 {target}' if target else 'hostname -I',
            'ip link show',
            'ibstat 2>/dev/null || echo "No InfiniBand"',
            'cat /etc/resolv.conf',
        ]

    @staticmethod
    def storage_commands() -> list[str]:
        """存储诊断命令"""
        return [
            'df -h',
            'mount | grep -E "nfs|lustre|gpfs"',
            'lsblk',
            'cat /proc/mounts | head -20',
        ]

    @staticmethod
    def job_failure_commands(job_id: str, cluster_type: str = "slurm") -> list[str]:
        """作业故障诊断命令"""
        if cluster_type == "slurm":
            return [
                f'sacct -j {job_id} --format=JobID,State,ExitCode,DerivedExitCode,Comment',
                f'scontrol show job {job_id}',
                f'seff {job_id} 2>/dev/null || echo "seff not available"',
            ]
        return [f'qstat -f {job_id}']


# ============================================================================
# 诊断结果解析器
# ============================================================================

class DiagnosticResultParser:
    """诊断结果解析器"""

    @staticmethod
    def parse_node_status(output: str) -> dict:
        """解析节点状态"""
        result = {
            'status': 'unknown',
            'issues': [],
            'metrics': {}
        }

        # 检查负载
        if 'load average' in output:
            # 解析负载
            pass

        # 检查内存
        if 'Mem:' in output:
            # 解析内存使用
            pass

        return result

    @staticmethod
    def parse_job_failure(output: str) -> dict:
        """解析作业失败原因"""
        failure_reasons = {
            'OUT_OF_MEMORY': '内存不足，建议增加内存请求或优化程序',
            'TIMEOUT': '作业超时，建议增加时间限制或优化程序',
            'FAILED': '作业执行失败，请检查错误日志',
            'CANCELLED': '作业被取消',
            'NODE_FAIL': '节点故障，建议重新提交作业',
        }

        result = {
            'reason': 'unknown',
            'suggestion': '',
            'details': output
        }

        for code, desc in failure_reasons.items():
            if code in output:
                result['reason'] = code
                result['suggestion'] = desc
                break

        return result
