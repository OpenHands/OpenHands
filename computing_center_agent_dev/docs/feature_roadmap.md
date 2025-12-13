# 算力中心运维 Agent 功能扩展规划

## 一、可扩展功能规划

### 1. 核心功能增强

#### 1.1 实时监控仪表板
| 功能 | 说明 | 优先级 |
|------|------|--------|
| 集群概览 Dashboard | 节点状态、GPU 使用率、作业统计的实时面板 | P0 |
| 资源热力图 | GPU/CPU 使用率的可视化热力图 | P1 |
| 告警中心 | 实时告警展示和历史告警查询 | P0 |
| 性能趋势图 | 历史性能数据的趋势展示 | P1 |

#### 1.2 作业管理增强
| 功能 | 说明 | 优先级 |
|------|------|--------|
| 作业模板库 | 预定义的作业脚本模板 | P0 |
| 批量操作 | 批量提交/取消/修改作业 | P1 |
| 作业依赖 | 作业间依赖关系管理 | P2 |
| 智能调度建议 | 基于历史数据的资源推荐 | P1 |
| 作业成本估算 | 预估作业资源消耗和费用 | P2 |

#### 1.3 智能诊断增强
| 功能 | 说明 | 优先级 |
|------|------|--------|
| 自动根因分析 | 自动分析故障根本原因 | P0 |
| 智能修复建议 | 基于知识库的修复方案 | P0 |
| 预测性维护 | 基于趋势预测潜在问题 | P2 |
| 日志智能摘要 | LLM 驱动的日志总结 | P1 |

### 2. 新增工具

#### 2.1 network_monitor (网络监控)
```python
功能:
- InfiniBand 状态监控
- 节点间带宽测试
- 网络拓扑可视化
- RDMA 性能分析

参数:
- check_type: topology | bandwidth | latency | status
- source_node: 源节点
- target_node: 目标节点
```

#### 2.2 storage_monitor (存储监控)
```python
功能:
- 文件系统使用率 (Lustre/GPFS/NFS)
- I/O 性能监控
- 配额管理
- 存储健康检查

参数:
- check_type: usage | performance | quota | health
- filesystem: 文件系统路径
- user: 用户名
```

#### 2.3 user_manager (用户管理)
```python
功能:
- 用户配额查询/设置
- 用户作业统计
- 访问权限管理
- 使用报告生成

参数:
- action: quota | statistics | permissions | report
- user: 用户名
- time_range: 时间范围
```

#### 2.4 alert_manager (告警管理)
```python
功能:
- 告警规则配置
- 告警通知管理
- 告警确认/静默
- 告警历史查询

参数:
- action: list | ack | silence | configure
- alert_id: 告警 ID
- filter: 筛选条件
```

#### 2.5 report_generator (报告生成)
```python
功能:
- 资源使用报告
- 作业统计报告
- 性能分析报告
- 成本分析报告

参数:
- report_type: usage | jobs | performance | cost
- time_range: 时间范围
- format: pdf | html | csv | json
```

#### 2.6 maintenance_helper (维护助手)
```python
功能:
- 计划维护管理
- 节点上下线
- 软件更新检查
- 备份状态检查

参数:
- action: schedule | drain | update | backup
- target: 目标节点/服务
```

### 3. 微智能体扩展

#### 3.1 新增知识库
| 微智能体 | 内容 | 触发词 |
|----------|------|--------|
| pbs_torque.md | PBS/Torque 专家知识 | pbs, qsub, torque |
| kubernetes_hpc.md | K8s HPC 知识 | k8s, kubernetes, pod |
| lustre_gpfs.md | 并行文件系统知识 | lustre, gpfs, 存储 |
| infiniband.md | InfiniBand 网络知识 | ib, infiniband, rdma |
| deeplearning_opt.md | 深度学习优化知识 | pytorch, tensorflow, 训练优化 |
| cost_optimization.md | 成本优化知识 | 成本, 费用, 优化 |

### 4. 集成功能

#### 4.1 通知集成
- 企业微信/钉钉通知
- Slack/Teams 集成
- 邮件告警
- 短信告警 (严重问题)

#### 4.2 监控系统集成
- Prometheus 指标导出
- Grafana 仪表板模板
- ELK 日志集成
- Zabbix 监控集成

#### 4.3 工单系统集成
- 自动创建工单
- 工单状态跟踪
- 问题升级流程

---

## 二、可视化配置界面规划

### 基于 OpenHands 前端架构

OpenHands 前端技术栈:
- **React 19** + **TypeScript**
- **HeroUI** 组件库
- **TailwindCSS** 样式
- **Zustand** 状态管理
- **React Query** 服务端状态
- **Socket.IO** 实时通信

### 2.1 新增设置页面: 算力中心配置

**路由**: `/settings/computing-center`

```
computing-center-settings/
├── ClusterConfiguration        # 集群基础配置
│   ├── ClusterTypeSelector    # Slurm/PBS/K8s 选择
│   ├── ClusterConnectionForm  # 集群连接配置
│   └── PartitionManager       # 分区管理
│
├── ToolsConfiguration          # 工具开关配置
│   ├── ToolEnableToggles      # 启用/禁用各工具
│   └── ToolParametersForm     # 工具参数配置
│
├── AlertConfiguration          # 告警配置
│   ├── ThresholdSettings      # 阈值设置
│   └── NotificationSettings   # 通知配置
│
└── GPUConfiguration            # GPU 配置
    ├── VendorSelector         # GPU 厂商选择
    └── MonitoringSettings     # 监控参数
```

### 2.2 新增 Dashboard 页面: 集群监控

**路由**: `/dashboard/cluster`

```
cluster-dashboard/
├── ClusterOverview             # 集群概览卡片
│   ├── NodeStatusCard         # 节点状态统计
│   ├── GPUStatusCard          # GPU 状态统计
│   ├── JobsStatusCard         # 作业状态统计
│   └── ResourceUsageCard      # 资源使用率
│
├── ResourceCharts              # 资源图表
│   ├── GPUUtilizationChart    # GPU 使用率图表
│   ├── MemoryUsageChart       # 内存使用图表
│   └── JobQueueChart          # 作业队列图表
│
├── NodeGrid                    # 节点网格
│   ├── NodeCard               # 单个节点卡片
│   └── NodeDetailModal        # 节点详情弹窗
│
├── AlertPanel                  # 告警面板
│   └── AlertList              # 告警列表
│
└── QuickActions                # 快速操作
    ├── SubmitJobButton        # 提交作业
    ├── RefreshButton          # 刷新数据
    └── ExportButton           # 导出报告
```

### 2.3 API 扩展

```typescript
// 新增 API 端点

// 集群状态
GET  /api/computing/cluster/status        // 集群状态概览
GET  /api/computing/cluster/nodes         // 节点列表
GET  /api/computing/cluster/nodes/{node}  // 节点详情

// GPU 监控
GET  /api/computing/gpu/status            // GPU 状态
GET  /api/computing/gpu/processes         // GPU 进程

// 作业管理
GET  /api/computing/jobs                  // 作业列表
POST /api/computing/jobs                  // 提交作业
DELETE /api/computing/jobs/{id}           // 取消作业

// 告警
GET  /api/computing/alerts                // 告警列表
POST /api/computing/alerts/{id}/ack       // 确认告警

// 配置
GET  /api/computing/config                // 获取配置
POST /api/computing/config                // 保存配置
```

### 2.4 组件复用策略

从 OpenHands 现有组件复用:

| 现有组件 | 复用方式 |
|----------|----------|
| `SettingsInput` | 配置输入框 |
| `SettingsSwitch` | 功能开关 |
| `SettingsDropdownInput` | 下拉选择 |
| `SettingsLayout` | 设置页面布局 |
| `SettingsNavigation` | 设置导航 |
| `Card` | 仪表板卡片 |
| 图表库 | 需要新增 (Recharts/Chart.js) |

---

## 三、实现路线图

### 阶段一: 基础功能 (2周)
- [ ] 新增 4 个工具 (network, storage, user, alert)
- [ ] 新增 3 个微智能体
- [ ] 基础配置页面

### 阶段二: 可视化界面 (3周)
- [ ] 设置页面: 集群配置
- [ ] Dashboard: 集群概览
- [ ] API 端点实现

### 阶段三: 高级功能 (2周)
- [ ] 智能诊断增强
- [ ] 报告生成
- [ ] 告警集成

### 阶段四: 集成优化 (2周)
- [ ] 监控系统集成
- [ ] 性能优化
- [ ] 文档完善

---

## 四、技术要点

### 4.1 前端开发要点

```typescript
// 配置页面组件示例
import { SettingsDropdownInput, SettingsSwitch } from '@/components/features/settings';

export function ComputingCenterSettings() {
  const { settings, updateSettings } = useComputingSettings();

  return (
    <div className="space-y-6">
      <SettingsDropdownInput
        label="集群类型"
        options={[
          { key: 'slurm', label: 'Slurm' },
          { key: 'pbs', label: 'PBS/Torque' },
          { key: 'k8s', label: 'Kubernetes' },
        ]}
        defaultSelectedKey={settings.clusterType}
        onInputChange={(value) => updateSettings({ clusterType: value })}
      />

      <SettingsSwitch
        label="启用 GPU 监控"
        isSelected={settings.enableGpuMonitor}
        onValueChange={(value) => updateSettings({ enableGpuMonitor: value })}
      />
    </div>
  );
}
```

### 4.2 后端 API 开发要点

```python
# 新增路由文件: openhands/server/routes/computing.py
from fastapi import APIRouter, Depends

router = APIRouter(prefix="/api/computing", tags=["computing"])

@router.get("/cluster/status")
async def get_cluster_status():
    """获取集群状态概览"""
    pass

@router.get("/config")
async def get_computing_config():
    """获取算力中心配置"""
    pass

@router.post("/config")
async def save_computing_config(config: ComputingConfig):
    """保存算力中心配置"""
    pass
```

### 4.3 实时数据更新

```typescript
// 使用 Socket.IO 获取实时更新
import { useEffect } from 'react';
import { socket } from '@/socket';

export function useClusterStatus() {
  const [status, setStatus] = useState(null);

  useEffect(() => {
    socket.on('cluster_status', (data) => {
      setStatus(data);
    });

    return () => {
      socket.off('cluster_status');
    };
  }, []);

  return status;
}
```
