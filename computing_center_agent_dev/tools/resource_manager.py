"""
资源管理工具

提供集群资源的配置和管理功能。

主要功能:
- 配额管理
- 优先级调整
- 资源预留
- QoS 配置
"""

from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

# ============================================================================
# 工具描述
# ============================================================================

RESOURCE_MANAGER_DESCRIPTION = """管理算力集群的资源配置。

### 功能概述
提供集群资源的配置和管理功能，包括配额、优先级、预留等。

### 操作类型

**quota_info (配额信息)**
- 查看用户/组的资源配额
- 查看已使用资源量
- 查看剩余配额

**quota_set (设置配额)**
- 设置用户资源配额
- 设置组资源配额
- 需要管理员权限

**priority (优先级管理)**
- 查看作业优先级
- 调整作业优先级
- 查看优先级因子

**reservation (资源预留)**
- 查看现有预留
- 创建资源预留
- 取消资源预留

**qos (QoS 配置)**
- 查看 QoS 配置
- 设置 QoS 策略

**fairshare (公平分享)**
- 查看公平分享状态
- 查看用户/组使用历史

**accounting (资源统计)**
- 查看资源使用统计
- 按用户/组/项目统计
- 生成使用报告

### 示例用法
1. 查看我的配额: action="quota_info"
2. 查看作业优先级: action="priority", job_id="12345"
3. 创建资源预留: action="reservation", nodes="node[01-04]", duration="4h"
"""

# ============================================================================
# 参数定义
# ============================================================================

RESOURCE_MANAGER_PARAMETERS = {
    'type': 'object',
    'properties': {
        'action': {
            'type': 'string',
            'enum': ['quota_info', 'quota_set', 'priority', 'reservation', 'qos', 'fairshare', 'accounting'],
            'description': '操作类型'
        },
        'user': {
            'type': 'string',
            'description': '(可选) 用户名'
        },
        'group': {
            'type': 'string',
            'description': '(可选) 用户组名'
        },
        'job_id': {
            'type': 'string',
            'description': '(可选) 作业 ID'
        },
        'nodes': {
            'type': 'string',
            'description': '(可选) 节点列表，如 "node[01-04]"'
        },
        'duration': {
            'type': 'string',
            'description': '(可选) 时长，如 "4h", "1d"'
        },
        'start_time': {
            'type': 'string',
            'description': '(可选) 开始时间'
        },
        'partition': {
            'type': 'string',
            'description': '(可选) 分区名称'
        },
        'time_range': {
            'type': 'string',
            'description': '(可选) 统计时间范围'
        },
    },
    'required': ['action'],
}

# ============================================================================
# 工具创建函数
# ============================================================================

def create_resource_manager_tool(
    cluster_type: str = "slurm",
    use_short_description: bool = False
) -> ChatCompletionToolParam:
    """
    创建资源管理工具

    Args:
        cluster_type: 集群类型
        use_short_description: 是否使用简短描述

    Returns:
        ChatCompletionToolParam: 工具定义
    """
    description = RESOURCE_MANAGER_DESCRIPTION

    if cluster_type == "slurm":
        description += '\n\n**当前集群: Slurm**\n使用 sacctmgr, sprio, scontrol 等命令'

    return ChatCompletionToolParam(
        type='function',
        function=ChatCompletionToolParamFunctionChunk(
            name='resource_manager',
            description=description if not use_short_description else '管理集群资源配置',
            parameters=RESOURCE_MANAGER_PARAMETERS,
        ),
    )


# ============================================================================
# 命令生成器
# ============================================================================

class ResourceManagerCommands:
    """资源管理命令生成器"""

    @staticmethod
    def get_slurm_command(action: str, **kwargs) -> str:
        """生成 Slurm 资源管理命令"""
        user = kwargs.get('user', '$USER')
        job_id = kwargs.get('job_id', '')
        time_range = kwargs.get('time_range', 'month')

        commands = {
            'quota_info': f'sacctmgr show assoc user={user} format=User,Account,GrpTRESMins,MaxTRESMins',
            'priority': f'sprio -j {job_id}' if job_id else 'sprio -l',
            'reservation': 'scontrol show reservation',
            'qos': 'sacctmgr show qos format=Name,Priority,MaxWall,MaxTRESPU',
            'fairshare': f'sshare -u {user} -a',
            'accounting': f'sreport user top start={time_range} end=now -t hourper',
        }

        return commands.get(action, 'sacctmgr show assoc')

    @staticmethod
    def get_pbs_command(action: str, **kwargs) -> str:
        """生成 PBS 资源管理命令"""
        commands = {
            'quota_info': 'qstat -Q',
            'priority': 'qstat -s',
            'reservation': 'pbs_rsub -l',
            'accounting': 'tracejob',
        }
        return commands.get(action, 'qstat -Q')


# ============================================================================
# 资源统计工具
# ============================================================================

class ResourceStatistics:
    """资源使用统计"""

    @staticmethod
    def get_user_usage_command(user: str, start_date: str, end_date: str) -> str:
        """获取用户资源使用统计命令"""
        return f'sacct -u {user} -S {start_date} -E {end_date} --format=JobID,JobName,Elapsed,NCPUS,NNodes,MaxRSS,State'

    @staticmethod
    def get_group_usage_command(group: str, start_date: str, end_date: str) -> str:
        """获取组资源使用统计命令"""
        return f'sreport cluster AccountUtilizationByUser account={group} start={start_date} end={end_date}'

    @staticmethod
    def get_gpu_usage_command(start_date: str, end_date: str) -> str:
        """获取 GPU 使用统计命令"""
        return f'sacct -S {start_date} -E {end_date} --format=JobID,User,Partition,AllocGRES,Elapsed,State | grep gpu'


# ============================================================================
# 资源预留管理
# ============================================================================

class ReservationManager:
    """资源预留管理"""

    @staticmethod
    def create_reservation_command(
        name: str,
        nodes: str,
        start_time: str,
        duration: str,
        users: str = ""
    ) -> str:
        """创建资源预留命令"""
        cmd = f'scontrol create reservation ReservationName={name} '
        cmd += f'Nodes={nodes} StartTime={start_time} Duration={duration}'
        if users:
            cmd += f' Users={users}'
        return cmd

    @staticmethod
    def delete_reservation_command(name: str) -> str:
        """删除资源预留命令"""
        return f'scontrol delete reservation {name}'

    @staticmethod
    def show_reservation_command(name: str = "") -> str:
        """显示资源预留命令"""
        if name:
            return f'scontrol show reservation {name}'
        return 'scontrol show reservation'
