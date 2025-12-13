"""
作业管理工具

提供作业的提交、查询、取消等管理功能。

支持的调度系统:
- Slurm: sbatch, squeue, scancel, scontrol
- PBS: qsub, qstat, qdel
- Kubernetes: kubectl apply/get/delete
"""

from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

# ============================================================================
# 工具描述
# ============================================================================

JOB_MANAGER_DESCRIPTION = """管理算力集群上的计算作业。

### 功能概述
提供完整的作业生命周期管理，包括提交、查询、修改和取消作业。

### 操作类型

**submit (提交作业)**
- 提交新的计算作业到集群
- 支持指定分区、资源需求
- 返回作业 ID

**status (查询状态)**
- 查看指定作业的详细状态
- 显示运行时间、资源使用
- 显示作业输出位置

**list (列出作业)**
- 列出当前用户的所有作业
- 支持按状态筛选
- 支持查看所有用户的作业

**cancel (取消作业)**
- 取消指定的作业
- 支持批量取消
- 返回取消结果

**hold (暂停作业)**
- 暂停排队中的作业
- 作业不会被调度执行

**release (释放作业)**
- 释放被暂停的作业
- 作业重新进入调度队列

**modify (修改作业)**
- 修改等待中作业的参数
- 如时间限制、优先级等

**output (查看输出)**
- 查看作业的标准输出
- 查看作业的错误输出

### 作业状态说明
- **PENDING (PD)**: 等待资源中
- **RUNNING (R)**: 正在运行
- **COMPLETED (CD)**: 已完成
- **FAILED (F)**: 运行失败
- **CANCELLED (CA)**: 已取消
- **TIMEOUT (TO)**: 超时终止

### 示例用法
1. 提交作业: action="submit", script_path="/path/to/job.sh"
2. 查看作业状态: action="status", job_id="12345"
3. 列出我的作业: action="list"
4. 取消作业: action="cancel", job_id="12345"
"""

# ============================================================================
# 参数定义
# ============================================================================

JOB_MANAGER_PARAMETERS = {
    'type': 'object',
    'properties': {
        'action': {
            'type': 'string',
            'enum': ['submit', 'status', 'list', 'cancel', 'hold', 'release', 'modify', 'output'],
            'description': '操作类型'
        },
        'job_id': {
            'type': 'string',
            'description': '作业 ID (status/cancel/hold/release/modify/output 操作需要)'
        },
        'script_path': {
            'type': 'string',
            'description': '作业脚本路径 (submit 操作需要)'
        },
        'partition': {
            'type': 'string',
            'description': '(可选) 指定分区/队列'
        },
        'user': {
            'type': 'string',
            'description': '(可选) 按用户筛选作业'
        },
        'state': {
            'type': 'string',
            'enum': ['all', 'pending', 'running', 'completed', 'failed'],
            'description': '(可选) 按状态筛选作业'
        },
        'output_type': {
            'type': 'string',
            'enum': ['stdout', 'stderr', 'both'],
            'description': '(可选) 输出类型，默认为 stdout'
        },
        'limit': {
            'type': 'integer',
            'description': '(可选) 限制显示的作业数量'
        },
    },
    'required': ['action'],
}

# ============================================================================
# 工具创建函数
# ============================================================================

def create_job_manager_tool(
    cluster_type: str = "slurm",
    default_partition: str = "default",
    use_short_description: bool = False
) -> ChatCompletionToolParam:
    """
    创建作业管理工具

    Args:
        cluster_type: 集群类型 (slurm, pbs, k8s)
        default_partition: 默认分区名称
        use_short_description: 是否使用简短描述

    Returns:
        ChatCompletionToolParam: 工具定义
    """
    description = JOB_MANAGER_DESCRIPTION

    # 添加集群特定信息
    cluster_info = {
        'slurm': f'\n\n**当前集群: Slurm**\n默认分区: {default_partition}\n命令: sbatch, squeue, scancel',
        'pbs': f'\n\n**当前集群: PBS/Torque**\n默认队列: {default_partition}\n命令: qsub, qstat, qdel',
        'k8s': '\n\n**当前集群: Kubernetes**\n命令: kubectl apply, kubectl get, kubectl delete',
    }

    description += cluster_info.get(cluster_type, '')

    return ChatCompletionToolParam(
        type='function',
        function=ChatCompletionToolParamFunctionChunk(
            name='job_manager',
            description=description if not use_short_description else '管理计算作业 (提交/查询/取消)',
            parameters=JOB_MANAGER_PARAMETERS,
        ),
    )


# ============================================================================
# 命令生成器
# ============================================================================

class JobManagerCommands:
    """作业管理命令生成器"""

    @staticmethod
    def get_slurm_command(action: str, **kwargs) -> str:
        """生成 Slurm 作业管理命令"""
        job_id = kwargs.get('job_id', '')
        script_path = kwargs.get('script_path', '')
        partition = kwargs.get('partition', '')
        user = kwargs.get('user', '$USER')

        commands = {
            'submit': f'sbatch {"-p " + partition if partition else ""} {script_path}',
            'status': f'scontrol show job {job_id}',
            'list': f'squeue -u {user} -o "%.10i %.9P %.30j %.8u %.8T %.10M %.10l %.6D %R"',
            'cancel': f'scancel {job_id}',
            'hold': f'scontrol hold {job_id}',
            'release': f'scontrol release {job_id}',
            'output': f'scontrol show job {job_id} | grep -E "StdOut|StdErr"',
        }

        cmd = commands.get(action, f'squeue -u {user}')

        # 状态筛选
        if kwargs.get('state') and kwargs['state'] != 'all':
            state_map = {
                'pending': 'PD',
                'running': 'R',
                'completed': 'CD',
                'failed': 'F',
            }
            cmd += f' -t {state_map.get(kwargs["state"], "")}'

        return cmd

    @staticmethod
    def get_pbs_command(action: str, **kwargs) -> str:
        """生成 PBS 作业管理命令"""
        job_id = kwargs.get('job_id', '')
        script_path = kwargs.get('script_path', '')

        commands = {
            'submit': f'qsub {script_path}',
            'status': f'qstat -f {job_id}',
            'list': 'qstat -u $USER',
            'cancel': f'qdel {job_id}',
            'hold': f'qhold {job_id}',
            'release': f'qrls {job_id}',
        }

        return commands.get(action, 'qstat')

    @staticmethod
    def get_k8s_command(action: str, **kwargs) -> str:
        """生成 Kubernetes 作业管理命令"""
        job_id = kwargs.get('job_id', '')
        script_path = kwargs.get('script_path', '')
        namespace = kwargs.get('namespace', 'default')

        commands = {
            'submit': f'kubectl apply -f {script_path} -n {namespace}',
            'status': f'kubectl describe job {job_id} -n {namespace}',
            'list': f'kubectl get jobs -n {namespace}',
            'cancel': f'kubectl delete job {job_id} -n {namespace}',
        }

        return commands.get(action, 'kubectl get jobs')


# ============================================================================
# 作业脚本模板
# ============================================================================

JOB_SCRIPT_TEMPLATES = {
    'slurm': '''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks}
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres=gpu:{gpus}
#SBATCH --time={time_limit}
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# 加载环境模块
# module load cuda/11.8
# module load python/3.10

# 激活虚拟环境
# source /path/to/venv/bin/activate

# 运行程序
{command}
''',

    'pbs': '''#!/bin/bash
#PBS -N {job_name}
#PBS -q {partition}
#PBS -l nodes={nodes}:ppn={cpus}:gpus={gpus}
#PBS -l walltime={time_limit}
#PBS -o {job_name}.out
#PBS -e {job_name}.err

cd $PBS_O_WORKDIR

# 运行程序
{command}
''',
}
