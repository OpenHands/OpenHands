"""
集群监控工具

提供集群状态查询功能，支持多种调度系统。

支持的调度系统:
- Slurm: sinfo, squeue, scontrol
- PBS: pbsnodes, qstat
- Kubernetes: kubectl get nodes/pods
"""

from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

# ============================================================================
# 工具描述
# ============================================================================

CLUSTER_MONITOR_DESCRIPTION = """查询算力集群的状态信息。

### 功能概述
监控高性能计算集群的实时状态，包括节点、资源、队列等信息。

### 查询类型

**nodes (节点状态)**
- 查看所有计算节点的状态
- 显示节点是否在线、空闲、繁忙或维护中
- 支持按状态筛选 (idle, allocated, down, drain)

**resources (资源使用)**
- 查看 CPU、内存、GPU 的整体使用情况
- 显示已分配/总计资源
- 计算使用率百分比

**queues (队列状态)**
- 查看作业队列/分区信息
- 显示各队列的作业数量
- 显示队列的资源限制

**partitions (分区详情)**
- 查看分区配置详情
- 显示分区的节点列表
- 显示分区的时间限制和优先级

**summary (状态摘要)**
- 整体集群健康状态概览
- 关键指标汇总
- 告警信息

### 支持的调度系统
- **Slurm**: 使用 sinfo, squeue, scontrol 命令
- **PBS/Torque**: 使用 pbsnodes, qstat 命令
- **Kubernetes**: 使用 kubectl 命令

### 示例用法
1. 查看所有节点: query_type="nodes"
2. 查看空闲节点: query_type="nodes", filter="idle"
3. 查看 GPU 队列资源: query_type="resources", partition="gpu"
4. 获取集群摘要: query_type="summary"
"""

# ============================================================================
# 参数定义
# ============================================================================

CLUSTER_MONITOR_PARAMETERS = {
    'type': 'object',
    'properties': {
        'query_type': {
            'type': 'string',
            'enum': ['nodes', 'resources', 'queues', 'partitions', 'summary'],
            'description': '查询类型: nodes-节点状态, resources-资源使用, queues-队列状态, partitions-分区详情, summary-状态摘要'
        },
        'node_name': {
            'type': 'string',
            'description': '(可选) 指定节点名称，查看单个节点的详细信息'
        },
        'partition': {
            'type': 'string',
            'description': '(可选) 指定分区/队列名称进行筛选'
        },
        'filter': {
            'type': 'string',
            'enum': ['all', 'idle', 'allocated', 'down', 'drain', 'mixed'],
            'description': '(可选) 按节点状态筛选，默认为 all'
        },
        'format': {
            'type': 'string',
            'enum': ['table', 'json', 'brief'],
            'description': '(可选) 输出格式，默认为 table'
        },
    },
    'required': ['query_type'],
}

# ============================================================================
# 工具创建函数
# ============================================================================

def create_cluster_monitor_tool(
    cluster_type: str = "slurm",
    use_short_description: bool = False
) -> ChatCompletionToolParam:
    """
    创建集群监控工具

    Args:
        cluster_type: 集群类型 (slurm, pbs, k8s)
        use_short_description: 是否使用简短描述

    Returns:
        ChatCompletionToolParam: 工具定义
    """
    description = CLUSTER_MONITOR_DESCRIPTION

    # 根据集群类型添加特定说明
    cluster_specific = {
        'slurm': '\n\n**当前集群类型: Slurm**\n使用 sinfo, squeue, scontrol 等命令',
        'pbs': '\n\n**当前集群类型: PBS/Torque**\n使用 pbsnodes, qstat 等命令',
        'k8s': '\n\n**当前集群类型: Kubernetes**\n使用 kubectl 命令',
    }

    description += cluster_specific.get(cluster_type, '')

    return ChatCompletionToolParam(
        type='function',
        function=ChatCompletionToolParamFunctionChunk(
            name='cluster_monitor',
            description=description if not use_short_description else '查询集群状态信息',
            parameters=CLUSTER_MONITOR_PARAMETERS,
        ),
    )


# ============================================================================
# 命令生成器 (供 Agent 运行时使用)
# ============================================================================

class ClusterMonitorCommands:
    """
    集群监控命令生成器

    根据查询类型和集群系统生成相应的命令。
    """

    @staticmethod
    def get_slurm_command(query_type: str, **kwargs) -> str:
        """生成 Slurm 命令"""
        commands = {
            'nodes': 'sinfo -N -l',
            'resources': 'sinfo -o "%P %a %l %D %t %c %m %G"',
            'queues': 'squeue -a -o "%.10i %.9P %.20j %.8u %.8T %.10M %.9l %.6D %R"',
            'partitions': 'scontrol show partition',
            'summary': 'sinfo -s && echo "---" && squeue -h | wc -l',
        }

        cmd = commands.get(query_type, 'sinfo')

        # 添加筛选条件
        if kwargs.get('node_name'):
            cmd += f" -n {kwargs['node_name']}"
        if kwargs.get('partition'):
            cmd += f" -p {kwargs['partition']}"
        if kwargs.get('filter') and kwargs['filter'] != 'all':
            state_map = {
                'idle': 'idle',
                'allocated': 'allocated',
                'down': 'down',
                'drain': 'drain',
                'mixed': 'mixed',
            }
            cmd += f" -t {state_map.get(kwargs['filter'], '')}"

        return cmd

    @staticmethod
    def get_pbs_command(query_type: str, **kwargs) -> str:
        """生成 PBS 命令"""
        commands = {
            'nodes': 'pbsnodes -a',
            'resources': 'pbsnodes -a -F json',
            'queues': 'qstat -Q',
            'partitions': 'qstat -Qf',
            'summary': 'qstat -B',
        }
        return commands.get(query_type, 'pbsnodes -a')

    @staticmethod
    def get_k8s_command(query_type: str, **kwargs) -> str:
        """生成 Kubernetes 命令"""
        commands = {
            'nodes': 'kubectl get nodes -o wide',
            'resources': 'kubectl top nodes',
            'queues': 'kubectl get pods --all-namespaces',
            'partitions': 'kubectl get namespaces',
            'summary': 'kubectl cluster-info && kubectl get nodes',
        }
        return commands.get(query_type, 'kubectl get nodes')
