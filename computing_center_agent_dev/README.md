# 算力中心运维 CLI Agent

基于 OpenHands 框架的高性能计算集群运维智能助手。

## 特性

- **完全兼容**: 继承 CodeActAgent 全部能力
- **专业工具**: 6 个算力中心专用运维工具
- **多集群支持**: Slurm/PBS/Kubernetes
- **智能诊断**: 自动故障排查和日志分析

## 快速开始

### 1. 安装

```bash
# 复制到 OpenHands
cp -r computing_center_agent_dev /path/to/openhands/agenthub/

# 在 openhands/agenthub/__init__.py 中添加
from openhands.agenthub import computing_center_agent
```

### 2. 配置

```bash
cp config.example.toml config.toml
# 编辑 config.toml 设置 API Key 和集群参数
```

### 3. 运行

```bash
python -m openhands.core.main -t "查看集群状态"
```

## 工具列表

| 工具 | 功能 |
|------|------|
| `cluster_monitor` | 集群节点和资源监控 |
| `job_manager` | 作业提交、查询、取消 |
| `gpu_monitor` | GPU 状态和进程监控 |
| `diagnostic` | 节点和网络故障诊断 |
| `log_analyzer` | 作业日志智能分析 |
| `resource_manager` | 配额和资源管理 |

## 使用示例

```bash
# 查看集群状态
python -m openhands.core.main -t "显示所有节点状态"

# 提交作业
python -m openhands.core.main -t "提交 train.sh 到 gpu 分区"

# GPU 监控
python -m openhands.core.main -t "查看所有 GPU 使用情况"

# 故障诊断
python -m openhands.core.main -t "作业 12345 为什么失败了"
```

## 配置说明

```toml
[agent.ComputingCenterAgent]
cluster_type = "slurm"           # 集群类型
default_partition = "gpu"        # 默认分区
enable_gpu_monitor = true        # 启用 GPU 监控
gpu_vendor = "nvidia"            # GPU 厂商
```

## 目录结构

```
computing_center_agent_dev/
├── agent/          # Agent 核心
├── tools/          # 运维工具
├── prompts/        # 提示词模板
├── microagents/    # 知识库
├── tests/          # 测试
└── .claude/        # 专家团队
```

## 文档

- [详细开发文档](.CLAUDE.md)
- [配置示例](config.example.toml)

## 许可证

MIT License
