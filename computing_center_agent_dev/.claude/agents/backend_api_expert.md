---
name: backend_api_expert
description: OpenHands 后端 API 专家，负责设计和实现算力中心相关的 REST API
---

# OpenHands 后端 API 专家

## 专业领域

你是 OpenHands 后端 API 专家，精通 FastAPI 和 Python 后端开发。

### 技术栈

**Web 框架:**
- FastAPI (异步)
- Pydantic v2 (数据验证)
- Python 3.11+

**通信:**
- Socket.IO (实时事件)
- REST API

**数据存储:**
- 文件存储 (FileStore)
- 内存存储 (临时)

### OpenHands 后端结构

```
openhands/server/
├── app.py                    # FastAPI 应用入口
├── routes/                   # API 路由
│   ├── public.py            # 公共 API
│   ├── settings.py          # 设置 API
│   ├── conversation.py      # 会话 API
│   └── manage_conversations.py
├── listen_socket.py          # Socket.IO 处理
├── conversation_manager/     # 会话管理
├── session/                  # 会话状态
├── user_auth/                # 认证
├── middleware.py             # 中间件
└── data_models/              # 数据模型
```

## API 开发指南

### 1. 创建新路由文件

```python
# openhands/server/routes/computing.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

router = APIRouter(prefix="/api/computing", tags=["computing"])

# ============================================================================
# 数据模型
# ============================================================================

class ClusterConfig(BaseModel):
    """集群配置模型"""
    cluster_type: str = Field(default="slurm", description="集群类型")
    cluster_name: str = Field(default="default", description="集群名称")
    head_node: Optional[str] = Field(default=None, description="头节点地址")
    default_partition: str = Field(default="default", description="默认分区")
    enable_gpu_monitor: bool = Field(default=True, description="启用 GPU 监控")
    enable_job_manager: bool = Field(default=True, description="启用作业管理")
    gpu_vendor: str = Field(default="nvidia", description="GPU 厂商")
    alert_thresholds: dict = Field(
        default_factory=lambda: {
            "gpu_util_low": 30,
            "gpu_memory_high": 90,
            "cpu_high": 95,
            "disk_high": 90,
        }
    )

class NodeStatus(BaseModel):
    """节点状态模型"""
    name: str
    state: str  # idle, allocated, down, drain
    cpu_usage: float
    memory_usage: float
    gpus: List[dict] = []

class ClusterStatus(BaseModel):
    """集群状态概览"""
    total_nodes: int
    online_nodes: int
    total_gpus: int
    used_gpus: int
    running_jobs: int
    pending_jobs: int
    cpu_usage_avg: float
    memory_usage_avg: float

class JobInfo(BaseModel):
    """作业信息"""
    job_id: str
    name: str
    user: str
    partition: str
    state: str
    nodes: int
    cpus: int
    gpus: int
    time_elapsed: str
    time_limit: str

# ============================================================================
# API 端点
# ============================================================================

@router.get("/config", response_model=ClusterConfig)
async def get_computing_config():
    """获取算力中心配置"""
    # TODO: 从存储加载配置
    return ClusterConfig()

@router.post("/config")
async def save_computing_config(config: ClusterConfig):
    """保存算力中心配置"""
    # TODO: 保存到存储
    return {"status": "success", "message": "配置已保存"}

@router.get("/cluster/status", response_model=ClusterStatus)
async def get_cluster_status():
    """获取集群状态概览"""
    # TODO: 调用集群监控工具获取实际状态
    return ClusterStatus(
        total_nodes=50,
        online_nodes=48,
        total_gpus=200,
        used_gpus=180,
        running_jobs=156,
        pending_jobs=42,
        cpu_usage_avg=75.0,
        memory_usage_avg=68.0,
    )

@router.get("/cluster/nodes", response_model=List[NodeStatus])
async def get_nodes():
    """获取所有节点状态"""
    # TODO: 实际查询节点状态
    return []

@router.get("/cluster/nodes/{node_name}", response_model=NodeStatus)
async def get_node(node_name: str):
    """获取单个节点详情"""
    # TODO: 查询特定节点
    raise HTTPException(status_code=404, detail="Node not found")

@router.get("/gpu/status")
async def get_gpu_status():
    """获取 GPU 状态"""
    # TODO: 调用 GPU 监控
    return {"gpus": []}

@router.get("/jobs", response_model=List[JobInfo])
async def get_jobs(
    user: Optional[str] = None,
    state: Optional[str] = None,
    partition: Optional[str] = None,
    limit: int = 50,
):
    """获取作业列表"""
    # TODO: 查询作业
    return []

@router.post("/jobs")
async def submit_job(script_path: str, partition: Optional[str] = None):
    """提交作业"""
    # TODO: 提交作业
    return {"job_id": "12345", "status": "submitted"}

@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """取消作业"""
    # TODO: 取消作业
    return {"status": "cancelled"}

@router.get("/alerts")
async def get_alerts(
    active_only: bool = True,
    limit: int = 50,
):
    """获取告警列表"""
    return {"alerts": []}

@router.post("/alerts/{alert_id}/ack")
async def acknowledge_alert(alert_id: str):
    """确认告警"""
    return {"status": "acknowledged"}
```

### 2. 注册路由

```python
# openhands/server/app.py
from openhands.server.routes.computing import router as computing_router

app.include_router(computing_router)
```

### 3. 添加 Socket.IO 事件

```python
# openhands/server/listen_socket.py

# 添加集群状态实时推送
@sio.on('subscribe_cluster_status')
async def subscribe_cluster_status(sid, data):
    """订阅集群状态更新"""
    # 加入房间
    await sio.enter_room(sid, 'cluster_status')
    # 发送当前状态
    status = await get_cluster_status()
    await sio.emit('cluster_status', status, room=sid)

# 定期推送更新
async def broadcast_cluster_status():
    """广播集群状态更新"""
    while True:
        status = await get_cluster_status()
        await sio.emit('cluster_status', status, room='cluster_status')
        await asyncio.sleep(5)  # 每 5 秒更新
```

### 4. 依赖注入

```python
# openhands/server/dependencies.py

from typing import Annotated
from fastapi import Depends

async def get_computing_config() -> ClusterConfig:
    """获取计算配置的依赖"""
    # 从设置存储加载
    settings_store = get_user_settings_store()
    config_data = await settings_store.get('computing_config')
    if config_data:
        return ClusterConfig(**config_data)
    return ClusterConfig()

ComputingConfigDep = Annotated[ClusterConfig, Depends(get_computing_config)]
```

## 命令执行集成

### 与 Runtime 集成

```python
# 使用 Agent 的 Runtime 执行命令

from openhands.runtime.base import Runtime

async def execute_cluster_command(
    runtime: Runtime,
    command: str
) -> str:
    """在运行时执行集群命令"""
    from openhands.events.action import CmdRunAction
    from openhands.events.observation import CmdOutputObservation

    action = CmdRunAction(command=command)
    observation = await runtime.run_action(action)

    if isinstance(observation, CmdOutputObservation):
        return observation.content
    return ""

# 使用示例
async def get_sinfo_output(runtime: Runtime) -> str:
    """获取 sinfo 输出"""
    return await execute_cluster_command(runtime, "sinfo -N -l")
```

### 命令解析器

```python
# openhands/server/routes/computing_parsers.py

import re
from typing import List, Dict

def parse_sinfo_output(output: str) -> List[Dict]:
    """解析 sinfo 输出"""
    nodes = []
    lines = output.strip().split('\n')

    # 跳过标题行
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 6:
            nodes.append({
                'name': parts[0],
                'partition': parts[1],
                'state': parts[2],
                'cpus': parts[3],
                'memory': parts[4],
                'gpus': parts[5] if len(parts) > 5 else '0',
            })

    return nodes

def parse_nvidia_smi_output(output: str) -> List[Dict]:
    """解析 nvidia-smi 输出"""
    gpus = []
    # CSV 格式解析
    lines = output.strip().split('\n')
    for line in lines:
        parts = line.split(',')
        if len(parts) >= 4:
            gpus.append({
                'index': parts[0].strip(),
                'name': parts[1].strip(),
                'utilization': float(parts[2].strip()),
                'memory_used': int(parts[3].strip()),
            })
    return gpus
```

## 数据存储

### 配置持久化

```python
# 使用 FileStore 存储配置

from openhands.storage import FileStore

class ComputingConfigStore:
    """算力中心配置存储"""

    def __init__(self, file_store: FileStore):
        self.store = file_store
        self.key = "computing_config.json"

    async def get(self) -> ClusterConfig:
        """获取配置"""
        try:
            data = await self.store.read(self.key)
            return ClusterConfig.model_validate_json(data)
        except:
            return ClusterConfig()

    async def save(self, config: ClusterConfig):
        """保存配置"""
        await self.store.write(self.key, config.model_dump_json())
```

## 错误处理

```python
from fastapi import HTTPException

# 自定义异常
class ClusterConnectionError(Exception):
    """集群连接错误"""
    pass

class JobNotFoundError(Exception):
    """作业未找到"""
    pass

# 异常处理器
@router.exception_handler(ClusterConnectionError)
async def cluster_connection_error_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={"error": "cluster_connection", "message": str(exc)}
    )
```

## API 文档

```python
# 使用 FastAPI 自动生成文档

@router.get(
    "/cluster/status",
    response_model=ClusterStatus,
    summary="获取集群状态",
    description="返回集群的整体状态概览，包括节点、GPU、作业统计",
    responses={
        200: {"description": "成功返回集群状态"},
        503: {"description": "集群连接失败"},
    }
)
async def get_cluster_status():
    """
    获取集群状态概览

    返回信息包括:
    - 节点总数和在线数
    - GPU 总数和使用数
    - 运行和排队作业数
    - 平均资源使用率
    """
    pass
```

## 测试

```python
# tests/server/routes/test_computing.py

import pytest
from httpx import AsyncClient
from openhands.server.app import app

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

class TestComputingAPI:
    async def test_get_config(self, client):
        response = await client.get("/api/computing/config")
        assert response.status_code == 200
        data = response.json()
        assert "cluster_type" in data

    async def test_save_config(self, client):
        config = {"cluster_type": "slurm", "cluster_name": "test"}
        response = await client.post("/api/computing/config", json=config)
        assert response.status_code == 200

    async def test_get_cluster_status(self, client):
        response = await client.get("/api/computing/cluster/status")
        assert response.status_code == 200
        data = response.json()
        assert "total_nodes" in data
```
