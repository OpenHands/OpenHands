"""
GPU 监控工具

提供 GPU 状态监控功能，支持 NVIDIA 和 AMD GPU。

主要功能:
- GPU 使用率监控
- 显存使用监控
- GPU 进程查看
- 温度和功耗监控
"""

from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

# ============================================================================
# 工具描述
# ============================================================================

GPU_MONITOR_DESCRIPTION = """监控 GPU 设备的状态和使用情况。

### 功能概述
实时监控算力集群中的 GPU 设备，提供详细的性能和状态信息。

### 查询类型

**status (设备状态)**
- 查看所有 GPU 设备的状态
- 显示 GPU 型号、驱动版本
- 显示当前运行的 CUDA 版本

**utilization (使用率)**
- GPU 计算利用率 (%)
- 显存利用率 (%)
- 编码器/解码器利用率

**memory (显存信息)**
- 已用显存 / 总显存
- 显存使用率
- 各进程的显存占用

**processes (GPU 进程)**
- 查看占用 GPU 的进程
- 显示进程 PID、用户、显存占用
- 显示运行的命令

**temperature (温度功耗)**
- GPU 核心温度
- 功耗消耗
- 风扇转速

**topology (拓扑信息)**
- GPU 间的连接拓扑
- NVLink/PCIe 连接信息
- P2P 通信能力

### 支持的 GPU
- **NVIDIA**: 使用 nvidia-smi 命令
- **AMD**: 使用 rocm-smi 命令
- **Intel**: 使用 xpu-smi 命令

### 示例用法
1. 查看所有 GPU 状态: query_type="status"
2. 查看 GPU 0 的使用率: query_type="utilization", gpu_id="0"
3. 查看 GPU 进程: query_type="processes"
4. 持续监控: query_type="utilization", watch=true
"""

# ============================================================================
# 参数定义
# ============================================================================

GPU_MONITOR_PARAMETERS = {
    'type': 'object',
    'properties': {
        'query_type': {
            'type': 'string',
            'enum': ['status', 'utilization', 'memory', 'processes', 'temperature', 'topology'],
            'description': '查询类型'
        },
        'gpu_id': {
            'type': 'string',
            'description': '(可选) 指定 GPU ID，如 "0" 或 "0,1,2"'
        },
        'node': {
            'type': 'string',
            'description': '(可选) 指定节点名称，查看远程节点的 GPU'
        },
        'format': {
            'type': 'string',
            'enum': ['table', 'json', 'csv'],
            'description': '(可选) 输出格式，默认为 table'
        },
        'watch': {
            'type': 'boolean',
            'description': '(可选) 是否持续监控 (每秒刷新)'
        },
    },
    'required': ['query_type'],
}

# ============================================================================
# 工具创建函数
# ============================================================================

def create_gpu_monitor_tool(
    gpu_vendor: str = "nvidia",
    use_short_description: bool = False
) -> ChatCompletionToolParam:
    """
    创建 GPU 监控工具

    Args:
        gpu_vendor: GPU 厂商 (nvidia, amd, intel)
        use_short_description: 是否使用简短描述

    Returns:
        ChatCompletionToolParam: 工具定义
    """
    description = GPU_MONITOR_DESCRIPTION

    # 添加厂商特定信息
    vendor_info = {
        'nvidia': '\n\n**当前 GPU 类型: NVIDIA**\n使用 nvidia-smi 命令\n支持 CUDA 和 cuDNN',
        'amd': '\n\n**当前 GPU 类型: AMD**\n使用 rocm-smi 命令\n支持 ROCm 和 HIP',
        'intel': '\n\n**当前 GPU 类型: Intel**\n使用 xpu-smi 命令\n支持 oneAPI',
    }

    description += vendor_info.get(gpu_vendor, '')

    return ChatCompletionToolParam(
        type='function',
        function=ChatCompletionToolParamFunctionChunk(
            name='gpu_monitor',
            description=description if not use_short_description else '监控 GPU 设备状态',
            parameters=GPU_MONITOR_PARAMETERS,
        ),
    )


# ============================================================================
# 命令生成器
# ============================================================================

class GPUMonitorCommands:
    """GPU 监控命令生成器"""

    @staticmethod
    def get_nvidia_command(query_type: str, **kwargs) -> str:
        """生成 NVIDIA GPU 监控命令"""
        gpu_id = kwargs.get('gpu_id', '')
        gpu_filter = f'-i {gpu_id}' if gpu_id else ''
        watch = kwargs.get('watch', False)

        commands = {
            'status': f'nvidia-smi {gpu_filter}',
            'utilization': f'nvidia-smi {gpu_filter} --query-gpu=index,name,utilization.gpu,utilization.memory --format=csv,noheader,nounits',
            'memory': f'nvidia-smi {gpu_filter} --query-gpu=index,memory.used,memory.total,memory.free --format=csv,noheader',
            'processes': f'nvidia-smi {gpu_filter} --query-compute-apps=pid,process_name,used_memory --format=csv,noheader',
            'temperature': f'nvidia-smi {gpu_filter} --query-gpu=index,temperature.gpu,power.draw,fan.speed --format=csv,noheader',
            'topology': 'nvidia-smi topo -m',
        }

        cmd = commands.get(query_type, 'nvidia-smi')

        if watch and query_type in ['utilization', 'memory', 'temperature']:
            cmd = f'nvidia-smi {gpu_filter} -l 1'

        return cmd

    @staticmethod
    def get_amd_command(query_type: str, **kwargs) -> str:
        """生成 AMD GPU 监控命令"""
        commands = {
            'status': 'rocm-smi',
            'utilization': 'rocm-smi --showuse',
            'memory': 'rocm-smi --showmeminfo vram',
            'processes': 'rocm-smi --showpids',
            'temperature': 'rocm-smi --showtemp',
            'topology': 'rocm-smi --showtopo',
        }
        return commands.get(query_type, 'rocm-smi')

    @staticmethod
    def get_intel_command(query_type: str, **kwargs) -> str:
        """生成 Intel GPU 监控命令"""
        commands = {
            'status': 'xpu-smi discovery',
            'utilization': 'xpu-smi stats',
            'memory': 'xpu-smi stats -d 0',
            'processes': 'xpu-smi ps',
            'temperature': 'xpu-smi health',
            'topology': 'xpu-smi topology',
        }
        return commands.get(query_type, 'xpu-smi discovery')


# ============================================================================
# 告警阈值检查
# ============================================================================

class GPUAlertChecker:
    """GPU 告警检查器"""

    def __init__(
        self,
        util_low: int = 30,
        memory_high: int = 90,
        temp_high: int = 85
    ):
        self.util_low = util_low
        self.memory_high = memory_high
        self.temp_high = temp_high

    def check_utilization(self, util: float) -> dict | None:
        """检查 GPU 利用率"""
        if util < self.util_low:
            return {
                'level': 'warning',
                'message': f'GPU 利用率过低 ({util}% < {self.util_low}%)',
                'suggestion': '检查作业是否正常运行，或考虑调整资源分配'
            }
        return None

    def check_memory(self, used: int, total: int) -> dict | None:
        """检查显存使用"""
        usage_pct = (used / total) * 100 if total > 0 else 0
        if usage_pct > self.memory_high:
            return {
                'level': 'warning',
                'message': f'显存使用率过高 ({usage_pct:.1f}% > {self.memory_high}%)',
                'suggestion': '考虑减小 batch size 或使用梯度检查点'
            }
        return None

    def check_temperature(self, temp: float) -> dict | None:
        """检查温度"""
        if temp > self.temp_high:
            return {
                'level': 'critical',
                'message': f'GPU 温度过高 ({temp}°C > {self.temp_high}°C)',
                'suggestion': '检查散热系统，必要时降低负载'
            }
        return None
