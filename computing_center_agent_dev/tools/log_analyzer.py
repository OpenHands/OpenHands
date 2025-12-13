"""
日志分析工具

提供作业日志和系统日志的智能分析功能。

主要功能:
- 作业输出日志分析
- 系统日志分析
- 错误模式识别
- 性能日志分析
"""

from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

# ============================================================================
# 工具描述
# ============================================================================

LOG_ANALYZER_DESCRIPTION = """分析算力集群的各类日志。

### 功能概述
智能分析作业日志和系统日志，自动识别错误和异常模式。

### 分析类型

**job_output (作业输出)**
- 分析作业的标准输出
- 分析作业的错误输出
- 提取关键信息和错误

**system_log (系统日志)**
- 分析 syslog/messages
- 分析调度器日志
- 分析内核日志

**error_pattern (错误模式)**
- 识别常见错误模式
- CUDA 错误分析
- OOM 错误分析
- 网络错误分析

**performance_log (性能日志)**
- 分析训练日志
- 提取损失曲线
- 分析吞吐量

**custom (自定义分析)**
- 按关键字搜索
- 按时间范围筛选
- 正则表达式匹配

### 常见错误识别
- **CUDA out of memory**: 显存不足
- **Segmentation fault**: 内存访问错误
- **Connection refused**: 网络连接问题
- **Permission denied**: 权限问题
- **No space left on device**: 磁盘空间不足

### 示例用法
1. 分析作业输出: analyze_type="job_output", job_id="12345"
2. 搜索 CUDA 错误: analyze_type="error_pattern", pattern="CUDA"
3. 分析系统日志: analyze_type="system_log", time_range="1h"
"""

# ============================================================================
# 参数定义
# ============================================================================

LOG_ANALYZER_PARAMETERS = {
    'type': 'object',
    'properties': {
        'analyze_type': {
            'type': 'string',
            'enum': ['job_output', 'system_log', 'error_pattern', 'performance_log', 'custom'],
            'description': '分析类型'
        },
        'job_id': {
            'type': 'string',
            'description': '(可选) 作业 ID'
        },
        'log_path': {
            'type': 'string',
            'description': '(可选) 日志文件路径'
        },
        'pattern': {
            'type': 'string',
            'description': '(可选) 搜索模式/关键字'
        },
        'time_range': {
            'type': 'string',
            'description': '(可选) 时间范围，如 "1h", "30m", "1d"'
        },
        'lines': {
            'type': 'integer',
            'description': '(可选) 显示行数限制，默认 100'
        },
        'include_context': {
            'type': 'boolean',
            'description': '(可选) 是否包含上下文行'
        },
    },
    'required': ['analyze_type'],
}

# ============================================================================
# 工具创建函数
# ============================================================================

def create_log_analyzer_tool(
    use_short_description: bool = False
) -> ChatCompletionToolParam:
    """
    创建日志分析工具

    Returns:
        ChatCompletionToolParam: 工具定义
    """
    return ChatCompletionToolParam(
        type='function',
        function=ChatCompletionToolParamFunctionChunk(
            name='log_analyzer',
            description=LOG_ANALYZER_DESCRIPTION if not use_short_description else '分析作业和系统日志',
            parameters=LOG_ANALYZER_PARAMETERS,
        ),
    )


# ============================================================================
# 日志分析命令
# ============================================================================

class LogAnalyzerCommands:
    """日志分析命令生成器"""

    @staticmethod
    def get_job_log_commands(job_id: str, cluster_type: str = "slurm") -> list[str]:
        """获取作业日志命令"""
        if cluster_type == "slurm":
            return [
                f'scontrol show job {job_id} | grep -E "StdOut|StdErr"',
                f'cat slurm-{job_id}.out 2>/dev/null || echo "Output file not found"',
                f'cat slurm-{job_id}.err 2>/dev/null || echo "Error file not found"',
            ]
        return [f'qstat -f {job_id}']

    @staticmethod
    def get_system_log_commands(time_range: str = "1h") -> list[str]:
        """获取系统日志命令"""
        return [
            f'journalctl --since "{time_range} ago" -p err --no-pager | tail -50',
            'dmesg | tail -30',
        ]

    @staticmethod
    def get_error_search_command(pattern: str, log_path: str = "") -> str:
        """获取错误搜索命令"""
        if log_path:
            return f'grep -i "{pattern}" {log_path} | tail -50'
        return f'grep -ri "{pattern}" /var/log/ 2>/dev/null | tail -30'


# ============================================================================
# 错误模式库
# ============================================================================

ERROR_PATTERNS = {
    'cuda_oom': {
        'patterns': [
            'CUDA out of memory',
            'RuntimeError: CUDA error: out of memory',
            'torch.cuda.OutOfMemoryError',
        ],
        'description': 'GPU 显存不足',
        'suggestions': [
            '减小 batch size',
            '使用梯度检查点 (gradient checkpointing)',
            '使用混合精度训练',
            '清理 GPU 内存缓存',
        ]
    },
    'cpu_oom': {
        'patterns': [
            'Out of memory',
            'Cannot allocate memory',
            'MemoryError',
            'Killed',
        ],
        'description': 'CPU 内存不足',
        'suggestions': [
            '增加内存请求',
            '减少数据预加载数量',
            '使用内存映射文件',
        ]
    },
    'network_error': {
        'patterns': [
            'Connection refused',
            'Connection timed out',
            'Network is unreachable',
            'NCCL error',
        ],
        'description': '网络连接问题',
        'suggestions': [
            '检查网络配置',
            '检查防火墙设置',
            '检查分布式训练配置',
        ]
    },
    'permission_error': {
        'patterns': [
            'Permission denied',
            'Operation not permitted',
            'Access denied',
        ],
        'description': '权限不足',
        'suggestions': [
            '检查文件/目录权限',
            '确认用户组设置',
            '联系管理员授权',
        ]
    },
    'disk_error': {
        'patterns': [
            'No space left on device',
            'Disk quota exceeded',
            'Read-only file system',
        ],
        'description': '磁盘空间问题',
        'suggestions': [
            '清理临时文件',
            '检查磁盘配额',
            '联系管理员扩容',
        ]
    },
    'gpu_error': {
        'patterns': [
            'CUDA driver version is insufficient',
            'no CUDA-capable device',
            'GPU has fallen off the bus',
            'Xid error',
        ],
        'description': 'GPU 设备错误',
        'suggestions': [
            '检查 CUDA 驱动版本',
            '检查 GPU 设备状态',
            '联系管理员检查硬件',
        ]
    },
}


class ErrorPatternMatcher:
    """错误模式匹配器"""

    def __init__(self):
        self.patterns = ERROR_PATTERNS

    def match(self, log_content: str) -> list[dict]:
        """匹配日志中的错误模式"""
        matches = []
        log_lower = log_content.lower()

        for error_type, info in self.patterns.items():
            for pattern in info['patterns']:
                if pattern.lower() in log_lower:
                    matches.append({
                        'type': error_type,
                        'pattern': pattern,
                        'description': info['description'],
                        'suggestions': info['suggestions'],
                    })
                    break  # 每种类型只匹配一次

        return matches
