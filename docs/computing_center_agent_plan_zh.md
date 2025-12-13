# OpenHands 算力中心运维 CLI Agent 开发规划与学习教程

## 目录

1. [项目概述](#1-项目概述)
2. [OpenHands 架构详解](#2-openhands-架构详解)
3. [核心组件学习指南](#3-核心组件学习指南)
4. [算力中心运维 Agent 开发规划](#4-算力中心运维-agent-开发规划)
5. [具体实现步骤](#5-具体实现步骤)
6. [最佳实践与注意事项](#6-最佳实践与注意事项)

---

## 1. 项目概述

### 1.1 OpenHands 是什么

OpenHands 是一个功能强大的多智能体框架，支持：
- **多种 Agent 类型**：CodeActAgent、BrowsingAgent、ReadOnlyAgent 等
- **多种运行时环境**：Docker、本地、CLI、Kubernetes、远程
- **灵活的工具系统**：bash、Python、浏览器、文件编辑等
- **微智能体系统**：可扩展的领域知识注入机制
- **MCP 协议支持**：Model Context Protocol 工具扩展

### 1.2 仓库核心目录结构

```
OpenHandsDev/
├── openhands/                    # 主 Python 包
│   ├── agenthub/                # Agent 实现 (6种预置Agent)
│   │   ├── codeact_agent/       # 主力 Agent（多工具支持）
│   │   ├── browsing_agent/      # 浏览器专用 Agent
│   │   ├── readonly_agent/      # 只读 Agent
│   │   ├── loc_agent/           # 代码定位 Agent
│   │   ├── visualbrowsing_agent/# 可视化浏览 Agent
│   │   └── dummy_agent/         # 测试用 Agent
│   ├── controller/              # Agent 控制器和状态管理
│   ├── core/                    # 核心配置、设置、日志
│   │   ├── config/              # 配置类定义
│   │   ├── main.py              # CLI 入口点
│   │   └── setup.py             # Agent/Runtime 创建
│   ├── events/                  # 事件系统（Action/Observation）
│   ├── microagent/              # 微智能体系统
│   ├── runtime/                 # 运行时实现
│   │   └── impl/
│   │       ├── docker/          # Docker 容器运行时
│   │       ├── local/           # 本地运行时
│   │       ├── cli/             # CLI 运行时
│   │       └── kubernetes/      # K8s 运行时
│   ├── memory/                  # 记忆管理
│   ├── llm/                     # LLM 集成（LiteLLM）
│   └── mcp/                     # MCP 协议支持
├── config.template.toml         # 配置文件模板
└── skills/                      # 技能/微智能体模板
```

---

## 2. OpenHands 架构详解

### 2.1 核心架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户输入 (CLI/Web)                        │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Controller (控制器)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Agent      │  │    State     │  │   EventStream        │  │
│  │  (智能体)     │  │   (状态)     │  │   (事件流)            │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │    LLM       │ │   Memory     │ │  Microagent  │
            │  (大模型)     │ │   (记忆)     │ │  (微智能体)   │
            └──────────────┘ └──────────────┘ └──────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Runtime (运行时)                          │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐  │
│  │   Docker   │ │   Local    │ │    CLI     │ │    K8s     │  │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Agent 基类定义

文件位置：`openhands/controller/agent.py`

```python
class Agent(ABC):
    """Agent 抽象基类"""

    _registry: dict[str, type['Agent']] = {}  # Agent 注册表
    sandbox_plugins: list[PluginRequirement] = []  # 沙箱插件
    config_model: type[AgentConfig] = AgentConfig  # 配置模型

    def __init__(self, config: AgentConfig, llm_registry: LLMRegistry):
        self.llm = llm_registry.get_llm_from_agent_config('agent', config)
        self.config = config
        self.tools: list = []  # 工具列表
        self.mcp_tools: dict = {}  # MCP 工具

    @abstractmethod
    def step(self, state: 'State') -> 'Action':
        """执行一步推理，必须由子类实现"""
        pass

    @classmethod
    def register(cls, name: str, agent_cls: type['Agent']) -> None:
        """注册 Agent 到全局注册表"""
        cls._registry[name] = agent_cls

    @classmethod
    def get_cls(cls, name: str) -> type['Agent']:
        """根据名称获取 Agent 类"""
        return cls._registry[name]
```

### 2.3 事件系统

OpenHands 使用 **Action-Observation** 模式：

| Action (动作) | 说明 | 对应 Observation |
|---------------|------|------------------|
| `CmdRunAction` | 执行 bash 命令 | `CmdOutputObservation` |
| `IPythonRunCellAction` | 执行 Python 代码 | `IPythonRunCellObservation` |
| `FileReadAction` | 读取文件 | `FileReadObservation` |
| `FileWriteAction` | 写入文件 | `FileWriteObservation` |
| `BrowseURLAction` | 访问网页 | `BrowserOutputObservation` |
| `MessageAction` | 发送消息 | - |
| `AgentFinishAction` | 完成任务 | - |

### 2.4 工具定义格式

文件位置：`openhands/agenthub/codeact_agent/tools/bash.py`

```python
from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

def create_my_tool() -> ChatCompletionToolParam:
    return ChatCompletionToolParam(
        type='function',
        function=ChatCompletionToolParamFunctionChunk(
            name='my_tool_name',
            description='工具描述',
            parameters={
                'type': 'object',
                'properties': {
                    'param1': {
                        'type': 'string',
                        'description': '参数1描述',
                    },
                },
                'required': ['param1'],
            },
        ),
    )
```

---

## 3. 核心组件学习指南

### 3.1 CodeActAgent 分析

文件位置：`openhands/agenthub/codeact_agent/codeact_agent.py`

**核心特点：**
- 版本：2.2
- 支持多种工具：bash、Python、文件编辑、浏览器等
- 使用 Function Calling 模式与 LLM 交互

**关键代码结构：**

```python
class CodeActAgent(Agent):
    VERSION = '2.2'

    # 沙箱插件（Jupyter、AgentSkills）
    sandbox_plugins: list[PluginRequirement] = [
        AgentSkillsRequirement(),
        JupyterRequirement(),
    ]

    def __init__(self, config: AgentConfig, llm_registry: LLMRegistry):
        super().__init__(config, llm_registry)
        self.tools = self._get_tools()  # 获取工具列表
        self.conversation_memory = ConversationMemory(...)
        self.condenser = Condenser.from_config(...)

    def _get_tools(self) -> list:
        """根据配置加载工具"""
        tools = []
        if self.config.enable_cmd:
            tools.append(create_cmd_run_tool())
        if self.config.enable_jupyter:
            tools.append(IPythonTool)
        # ... 更多工具
        return tools

    def step(self, state: State) -> Action:
        """核心推理步骤"""
        # 1. 获取历史消息
        messages = self._get_messages(condensed_history, initial_user_message)

        # 2. 调用 LLM
        response = self.llm.completion(messages=messages, tools=self.tools)

        # 3. 解析响应为 Action
        actions = self.response_to_actions(response)
        return actions[0]
```

### 3.2 配置系统

**主配置文件：** `config.template.toml`

```toml
# 核心配置
[core]
default_agent = "CodeActAgent"
runtime = "docker"
max_iterations = 500

# LLM 配置
[llm]
model = "gpt-4o"
api_key = ""

# Agent 配置
[agent]
enable_browsing = true
enable_jupyter = true
enable_cmd = true
enable_think = true

# 自定义 Agent 配置
[agent.MyCustomAgent]
classpath = "my_package.my_agent.MyAgent"
```

**AgentConfig 类：** `openhands/core/config/agent_config.py`

```python
class AgentConfig(BaseModel):
    cli_mode: bool = False           # CLI 模式
    enable_browsing: bool = True     # 启用浏览器
    enable_jupyter: bool = True      # 启用 Jupyter
    enable_cmd: bool = True          # 启用 bash
    enable_think: bool = True        # 启用思考工具
    enable_finish: bool = True       # 启用完成工具
    enable_plan_mode: bool = True    # 启用规划模式
    disabled_microagents: list[str] = []  # 禁用的微智能体
    condenser: CondenserConfig = ...  # 上下文压缩配置
```

### 3.3 微智能体系统

**文件位置：** `openhands/microagent/microagent.py`

**微智能体类型：**

| 类型 | 说明 | 触发方式 |
|------|------|----------|
| `KNOWLEDGE` | 知识型 | 关键词触发 |
| `REPO_KNOWLEDGE` | 仓库知识型 | 始终激活 |
| `TASK` | 任务型 | `/name` 格式触发 |

**微智能体文件格式：**

```markdown
---
name: my_agent
type: knowledge
version: 1.0.0
agent: CodeActAgent
triggers:
  - keyword1
  - keyword2
---

# 微智能体内容

这里是微智能体的指令和知识内容...
```

**微智能体存放位置：**
- 项目级：`.openhands/microagents/`
- 全局级：`skills/`

### 3.4 运行时系统

**CLI 运行时：** `openhands/runtime/impl/cli/cli_runtime.py`

特点：
- 不需要 Docker
- 直接在本地执行命令
- 适合轻量级场景

**Docker 运行时：** `openhands/runtime/impl/docker/`

特点：
- 隔离的执行环境
- 支持完整的 Jupyter
- 适合生产环境

---

## 4. 算力中心运维 Agent 开发规划

### 4.1 需求分析

算力中心运维 CLI Agent 需要具备以下能力：

| 功能模块 | 具体能力 |
|----------|----------|
| **集群监控** | 节点状态查询、GPU使用率、内存/CPU监控 |
| **作业管理** | Slurm/PBS作业提交、查询、取消 |
| **资源调度** | GPU分配、队列管理、优先级调整 |
| **故障诊断** | 日志分析、错误排查、自动修复 |
| **性能优化** | 资源利用率分析、调度优化建议 |
| **安全管理** | 用户权限、访问控制、安全审计 |

### 4.2 整体架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                  ComputingCenterAgent                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ ClusterMonitor  │ │ JobManager     │ │ ResourceScheduler│   │
│  │ Tool            │ │ Tool           │ │ Tool             │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ DiagnosticTool  │ │ PerformanceTool│ │ SecurityTool    │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                      Microagents                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │ slurm.md   │ │ nvidia.md  │ │ k8s.md      │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 开发阶段规划

#### 第一阶段：基础框架搭建

**目标：** 创建基本的 Agent 结构和核心工具

**任务清单：**
1. 创建 `ComputingCenterAgent` 类
2. 实现基础配置 `ComputingCenterAgentConfig`
3. 创建集群状态查询工具
4. 创建作业管理工具
5. 编写系统提示词模板

**预计文件结构：**
```
openhands/agenthub/computing_center_agent/
├── __init__.py
├── computing_center_agent.py
├── config.py
├── tools/
│   ├── __init__.py
│   ├── cluster_monitor.py
│   ├── job_manager.py
│   ├── resource_scheduler.py
│   └── diagnostic.py
├── prompts/
│   ├── system_prompt.j2
│   └── examples/
└── README.md
```

#### 第二阶段：工具完善

**目标：** 实现完整的运维工具集

**任务清单：**
1. GPU 监控工具（nvidia-smi 集成）
2. Slurm/PBS 作业调度工具
3. 日志分析工具
4. 性能报告生成工具
5. 告警处理工具

#### 第三阶段：微智能体扩展

**目标：** 创建领域知识微智能体

**任务清单：**
1. Slurm 命令知识微智能体
2. NVIDIA GPU 运维微智能体
3. 网络诊断微智能体
4. 存储管理微智能体

#### 第四阶段：集成与优化

**目标：** 系统集成和性能优化

**任务清单：**
1. MCP 服务器集成
2. 上下文管理优化
3. 多集群支持
4. 安全策略实现
5. 测试与文档

---

## 5. 具体实现步骤

### 5.1 创建 Agent 基础结构

**步骤 1：创建 Agent 目录**

```bash
mkdir -p openhands/agenthub/computing_center_agent/tools
mkdir -p openhands/agenthub/computing_center_agent/prompts
```

**步骤 2：创建 Agent 主文件**

文件：`openhands/agenthub/computing_center_agent/computing_center_agent.py`

```python
from typing import TYPE_CHECKING

from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.llm.llm_registry import LLMRegistry

if TYPE_CHECKING:
    from openhands.events.action import Action

from openhands.agenthub.computing_center_agent.tools import (
    create_cluster_monitor_tool,
    create_job_manager_tool,
    create_gpu_monitor_tool,
)

class ComputingCenterAgentConfig(AgentConfig):
    """算力中心 Agent 专用配置"""
    cluster_type: str = "slurm"  # slurm, pbs, k8s
    enable_gpu_monitor: bool = True
    enable_job_manager: bool = True
    ssh_config_path: str | None = None


class ComputingCenterAgent(Agent):
    """算力中心运维 CLI Agent"""

    VERSION = '1.0'
    config_model = ComputingCenterAgentConfig

    def __init__(self, config: ComputingCenterAgentConfig, llm_registry: LLMRegistry):
        super().__init__(config, llm_registry)
        self.tools = self._get_tools()

    def _get_tools(self) -> list:
        tools = []
        tools.append(create_cluster_monitor_tool())
        if self.config.enable_job_manager:
            tools.append(create_job_manager_tool())
        if self.config.enable_gpu_monitor:
            tools.append(create_gpu_monitor_tool())
        return tools

    def step(self, state: State) -> 'Action':
        # 实现推理逻辑
        pass
```

**步骤 3：注册 Agent**

文件：`openhands/agenthub/computing_center_agent/__init__.py`

```python
from openhands.controller.agent import Agent
from openhands.agenthub.computing_center_agent.computing_center_agent import (
    ComputingCenterAgent,
)

Agent.register('ComputingCenterAgent', ComputingCenterAgent)
```

**步骤 4：在 agenthub 中导入**

修改 `openhands/agenthub/__init__.py`：

```python
from openhands.agenthub import (
    # ... 其他导入
    computing_center_agent,  # 添加这行
)
```

### 5.2 实现核心工具

**文件：** `openhands/agenthub/computing_center_agent/tools/cluster_monitor.py`

```python
from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

CLUSTER_MONITOR_DESCRIPTION = """查询算力集群的状态信息。

### 功能
- 查看集群节点状态
- 查看资源使用情况（CPU、内存、GPU）
- 查看队列状态
- 查看作业统计

### 参数
- query_type: 查询类型
  - nodes: 节点状态
  - resources: 资源使用
  - queues: 队列状态
  - jobs: 作业统计
- node_name: (可选) 特定节点名称
"""

def create_cluster_monitor_tool() -> ChatCompletionToolParam:
    return ChatCompletionToolParam(
        type='function',
        function=ChatCompletionToolParamFunctionChunk(
            name='cluster_monitor',
            description=CLUSTER_MONITOR_DESCRIPTION,
            parameters={
                'type': 'object',
                'properties': {
                    'query_type': {
                        'type': 'string',
                        'enum': ['nodes', 'resources', 'queues', 'jobs'],
                        'description': '查询类型',
                    },
                    'node_name': {
                        'type': 'string',
                        'description': '(可选) 特定节点名称',
                    },
                },
                'required': ['query_type'],
            },
        ),
    )
```

**文件：** `openhands/agenthub/computing_center_agent/tools/job_manager.py`

```python
from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

JOB_MANAGER_DESCRIPTION = """管理算力集群上的作业。

### 功能
- 提交作业
- 查询作业状态
- 取消作业
- 查看作业输出

### 支持的调度器
- Slurm (squeue, sbatch, scancel)
- PBS (qstat, qsub, qdel)
"""

def create_job_manager_tool() -> ChatCompletionToolParam:
    return ChatCompletionToolParam(
        type='function',
        function=ChatCompletionToolParamFunctionChunk(
            name='job_manager',
            description=JOB_MANAGER_DESCRIPTION,
            parameters={
                'type': 'object',
                'properties': {
                    'action': {
                        'type': 'string',
                        'enum': ['submit', 'status', 'cancel', 'output', 'list'],
                        'description': '操作类型',
                    },
                    'job_id': {
                        'type': 'string',
                        'description': '作业ID（status/cancel/output时需要）',
                    },
                    'script_path': {
                        'type': 'string',
                        'description': '作业脚本路径（submit时需要）',
                    },
                    'user': {
                        'type': 'string',
                        'description': '(可选) 按用户筛选作业',
                    },
                },
                'required': ['action'],
            },
        ),
    )
```

### 5.3 创建系统提示词

**文件：** `openhands/agenthub/computing_center_agent/prompts/system_prompt.j2`

```jinja2
# 算力中心运维助手

你是一个专业的算力中心运维 AI 助手。你的职责是帮助用户管理和监控高性能计算集群。

## 核心能力

1. **集群监控**: 查看节点状态、资源使用率、队列信息
2. **作业管理**: 提交、查询、取消计算作业
3. **GPU 监控**: 监控 NVIDIA GPU 使用情况
4. **故障诊断**: 分析日志、定位问题
5. **性能优化**: 提供资源调度建议

## 可用工具

{% for tool in tools %}
### {{ tool.function.name }}
{{ tool.function.description }}

{% endfor %}

## 工作原则

1. **安全第一**: 在执行任何可能影响系统的操作前，先确认用户意图
2. **信息准确**: 基于实际查询结果回答，不要猜测
3. **简洁高效**: 用最少的命令完成任务
4. **主动建议**: 发现问题时主动提供解决方案

## 常用命令参考

### Slurm 命令
- `sinfo`: 查看集群/分区状态
- `squeue`: 查看作业队列
- `sbatch`: 提交作业
- `scancel`: 取消作业
- `sacct`: 查看历史作业

### NVIDIA GPU
- `nvidia-smi`: GPU 状态
- `nvidia-smi -L`: 列出 GPU
- `nvidia-smi --query-gpu=...`: 自定义查询

## 当前时间
{{ current_time }}
```

### 5.4 创建微智能体

**文件：** `.openhands/microagents/slurm.md`

```markdown
---
name: slurm
type: knowledge
version: 1.0.0
agent: ComputingCenterAgent
triggers:
  - slurm
  - squeue
  - sbatch
  - sinfo
  - scancel
  - 作业调度
  - job scheduler
---

# Slurm 作业调度系统知识

## 常用命令

### 查看集群状态
```bash
# 查看节点状态
sinfo -N -l

# 查看特定分区
sinfo -p gpu

# 查看节点详情
scontrol show node <node_name>
```

### 作业管理
```bash
# 提交作业
sbatch job.sh

# 查看作业队列
squeue -u $USER

# 取消作业
scancel <job_id>

# 查看作业详情
scontrol show job <job_id>
```

### 作业脚本模板
```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load cuda/11.8
python train.py
```

## 故障排查

### 常见问题
1. 作业长时间 Pending
   - 检查资源请求是否超出限制
   - 查看队列优先级

2. 作业失败
   - 查看 .err 文件
   - 检查资源使用情况
```

**文件：** `.openhands/microagents/nvidia_gpu.md`

```markdown
---
name: nvidia_gpu
type: knowledge
version: 1.0.0
agent: ComputingCenterAgent
triggers:
  - nvidia
  - gpu
  - cuda
  - 显卡
  - GPU使用率
---

# NVIDIA GPU 运维知识

## 状态监控

### 基础命令
```bash
# 查看 GPU 状态
nvidia-smi

# 持续监控
nvidia-smi -l 1

# JSON 格式输出
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### 进程管理
```bash
# 查看 GPU 进程
nvidia-smi pmon

# 查看特定 GPU 进程
fuser -v /dev/nvidia*
```

## 常见问题

### GPU 内存泄漏
```bash
# 找到占用进程
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# 终止进程
kill -9 <pid>
```

### CUDA 版本问题
```bash
# 查看 CUDA 版本
nvcc --version

# 查看驱动支持的最高 CUDA 版本
nvidia-smi
```
```

### 5.5 配置文件

**文件：** `config.toml` (添加)

```toml
# 算力中心 Agent 配置
[agent.ComputingCenterAgent]
cluster_type = "slurm"
enable_gpu_monitor = true
enable_job_manager = true

# 使用特定的 LLM 配置
llm_config = "computing"

[llm.computing]
model = "gpt-4o"
temperature = 0.1
```

---

## 6. 最佳实践与注意事项

### 6.1 开发最佳实践

1. **继承而非重写**
   - 尽量继承 `CodeActAgent` 并扩展
   - 复用现有的工具和提示词管理

2. **工具设计原则**
   - 每个工具专注单一功能
   - 提供清晰的参数描述
   - 包含使用示例

3. **配置灵活性**
   - 使用 Pydantic 模型定义配置
   - 支持环境变量覆盖
   - 提供合理的默认值

4. **安全考虑**
   - 实现命令白名单
   - 敏感操作需要确认
   - 日志记录所有操作

### 6.2 测试策略

```python
# tests/unit/agenthub/test_computing_center_agent.py
import pytest
from openhands.agenthub.computing_center_agent import ComputingCenterAgent

class TestComputingCenterAgent:
    def test_agent_registration(self):
        from openhands.controller.agent import Agent
        agent_cls = Agent.get_cls('ComputingCenterAgent')
        assert agent_cls == ComputingCenterAgent

    def test_tools_loaded(self, mock_config, mock_llm_registry):
        agent = ComputingCenterAgent(mock_config, mock_llm_registry)
        tool_names = [t['function']['name'] for t in agent.tools]
        assert 'cluster_monitor' in tool_names
        assert 'job_manager' in tool_names
```

### 6.3 常见问题与解决

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Agent 未注册 | `__init__.py` 未导入 | 确保在 agenthub 的 `__init__.py` 中导入 |
| 工具不生效 | 配置未启用 | 检查 AgentConfig 中的 enable 标志 |
| 上下文过长 | 历史过多 | 配置合适的 Condenser |
| LLM 调用失败 | API 配置错误 | 检查 `config.toml` 中的 LLM 配置 |

### 6.4 性能优化建议

1. **上下文管理**
   ```toml
   [agent.ComputingCenterAgent.condenser]
   type = "llm"
   max_size = 50
   ```

2. **工具响应截断**
   ```python
   # 限制输出长度
   max_message_chars = 5000
   ```

3. **缓存机制**
   - 缓存集群状态查询结果
   - 设置合理的缓存过期时间

---

## 附录 A：关键文件速查表

| 文件 | 说明 |
|------|------|
| `openhands/controller/agent.py` | Agent 基类定义 |
| `openhands/core/main.py` | CLI 入口点 |
| `openhands/core/config/agent_config.py` | Agent 配置类 |
| `openhands/agenthub/codeact_agent/codeact_agent.py` | 主力 Agent 实现 |
| `openhands/events/action/` | Action 定义 |
| `openhands/events/observation/` | Observation 定义 |
| `openhands/microagent/microagent.py` | 微智能体系统 |
| `openhands/runtime/impl/cli/cli_runtime.py` | CLI 运行时 |
| `config.template.toml` | 配置文件模板 |

## 附录 B：CLI 使用方法

```bash
# 基本使用
python -m openhands.core.main -t "查看集群状态"

# 指定配置文件
python -m openhands.core.main --config-file config.toml -t "提交作业"

# 指定 Agent
python -m openhands.core.main --agent-config ComputingCenterAgent -t "GPU使用率"

# 从文件读取任务
python -m openhands.core.main -f task.txt

# 指定 LLM 配置
python -m openhands.core.main -l computing -t "查看队列"
```

## 附录 C：学习路径建议

1. **第一周：基础理解**
   - 阅读 `openhands/controller/agent.py`
   - 阅读 `openhands/agenthub/codeact_agent/codeact_agent.py`
   - 运行示例任务，观察事件流

2. **第二周：工具开发**
   - 学习工具定义格式
   - 阅读现有工具实现
   - 开发第一个自定义工具

3. **第三周：Agent 开发**
   - 创建自定义 Agent 结构
   - 实现 `step()` 方法
   - 集成自定义工具

4. **第四周：微智能体与优化**
   - 创建领域知识微智能体
   - 配置上下文管理
   - 性能测试与优化

---

**文档版本**: 1.0
**最后更新**: 2024年
**作者**: OpenHands 学习者
