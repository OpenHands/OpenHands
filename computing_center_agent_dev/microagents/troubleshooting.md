---
name: troubleshooting
type: knowledge
version: 1.0.0
agent: ComputingCenterAgent
triggers:
  - 故障
  - 错误
  - 失败
  - 问题
  - error
  - failed
  - trouble
  - 排查
  - 诊断
---

# HPC 集群故障排查指南

## 快速诊断流程

```
用户报告问题
    │
    ▼
┌─────────────────┐
│ 1. 确认问题现象 │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 2. 收集信息    │
│ - 作业ID       │
│ - 节点名       │
│ - 错误信息     │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 3. 分类问题    │
├─────────────────┤
│ □ 作业问题     │
│ □ 节点问题     │
│ □ 网络问题     │
│ □ 存储问题     │
│ □ GPU 问题     │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 4. 深入诊断    │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 5. 解决/上报   │
└─────────────────┘
```

## 作业问题排查

### 作业一直排队 (Pending)

**检查步骤:**
```bash
# 1. 查看排队原因
squeue -j <job_id> -o "%i %j %r"

# 2. 检查资源请求
scontrol show job <job_id> | grep -E "NumNodes|NumCPUs|Gres|TimeLimit"

# 3. 检查分区状态
sinfo -p <partition>

# 4. 检查用户限制
sacctmgr show assoc user=$USER
```

**常见原因和解决:**
| 原因码 | 含义 | 解决方案 |
|--------|------|----------|
| Resources | 等待资源 | 减少资源请求或等待 |
| Priority | 优先级低 | 等待或联系管理员 |
| QOSMaxJobsPerUserLimit | 超过作业限制 | 等待已有作业完成 |
| ReqNodeNotAvail | 节点不可用 | 换分区或等待维护完成 |

### 作业失败

**检查步骤:**
```bash
# 1. 查看退出状态
sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed,MaxRSS

# 2. 查看错误日志
cat slurm-<job_id>.err

# 3. 查看作业详情
scontrol show job <job_id>
```

**退出码解释:**
| 退出码 | 信号 | 含义 | 解决方案 |
|--------|------|------|----------|
| 0 | - | 成功 | - |
| 1 | - | 一般错误 | 检查程序逻辑 |
| 2 | - | 误用命令 | 检查命令参数 |
| 126 | - | 权限问题 | chmod +x 脚本 |
| 127 | - | 命令未找到 | 检查 PATH |
| 137 | SIGKILL | OOM 或超时 | 增加内存/时间 |
| 139 | SIGSEGV | 段错误 | 调试程序 |
| 143 | SIGTERM | 被终止 | 检查是否被 scancel |

## 节点问题排查

### 节点不可用 (down/drain)

**检查步骤:**
```bash
# 1. 查看节点状态
sinfo -N -n <node>

# 2. 查看详细原因
scontrol show node <node> | grep -E "State|Reason"

# 3. SSH 检查连通性
ssh <node> hostname

# 4. 检查系统日志 (需要 root)
ssh <node> journalctl -p err --since "1 hour ago"
```

### 节点性能异常

```bash
# 检查负载
ssh <node> uptime

# 检查内存
ssh <node> free -h

# 检查磁盘
ssh <node> df -h

# 检查进程
ssh <node> top -bn1 | head -20
```

## GPU 问题排查

### GPU 不可见

```bash
# 检查设备
nvidia-smi -L

# 检查驱动
lsmod | grep nvidia

# 检查 CUDA
nvcc --version

# 检查环境变量
echo $CUDA_VISIBLE_DEVICES
```

### GPU 错误 (Xid)

```bash
# 查看内核日志中的 GPU 错误
dmesg | grep -i nvidia
dmesg | grep -i xid

# 常见 Xid 错误
# Xid 13: 显存错误
# Xid 31: GPU 挂起
# Xid 43: GPU 掉电
# Xid 79: GPU 挂起恢复失败
```

## 网络问题排查

### 节点间连通性

```bash
# Ping 测试
ping -c 3 <node>

# SSH 测试
ssh -v <node> hostname

# 端口测试
nc -zv <node> 22
```

### InfiniBand 问题

```bash
# 查看 IB 状态
ibstat

# 检查 IB 连接
iblinkinfo

# 测试 IB 带宽
ib_write_bw <server>
```

### 分布式训练网络问题

```bash
# 测试 NCCL
export NCCL_DEBUG=INFO
python -c "import torch.distributed as dist; dist.init_process_group('nccl')"

# 检查防火墙
iptables -L
```

## 存储问题排查

### 文件系统检查

```bash
# 检查挂载
mount | grep -E "nfs|lustre|gpfs"

# 检查磁盘空间
df -h

# 检查 inode
df -i

# 检查配额
quota -s
```

### I/O 性能测试

```bash
# 写速度测试
dd if=/dev/zero of=testfile bs=1G count=1 oflag=direct

# 读速度测试
dd if=testfile of=/dev/null bs=1G count=1 iflag=direct
```

## 常见错误信息解析

| 错误信息 | 可能原因 | 解决方案 |
|----------|----------|----------|
| `CUDA out of memory` | 显存不足 | 减小 batch size |
| `Connection refused` | 服务未启动/端口被封 | 检查服务和防火墙 |
| `Permission denied` | 权限不足 | 检查文件权限 |
| `No space left on device` | 磁盘满 | 清理文件 |
| `Segmentation fault` | 内存访问错误 | 调试程序 |
| `Bus error` | 内存对齐问题 | 检查共享内存配置 |

## 问题上报模板

```
## 问题描述
[简要描述问题现象]

## 环境信息
- 用户名:
- 节点:
- 作业ID:
- 时间:

## 重现步骤
1.
2.
3.

## 错误信息
[粘贴错误日志]

## 已尝试的解决方法
1.
2.

## 附加信息
[其他相关信息]
```
