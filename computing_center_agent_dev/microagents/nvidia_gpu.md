---
name: nvidia_gpu
type: knowledge
version: 1.0.0
agent: ComputingCenterAgent
triggers:
  - nvidia
  - gpu
  - cuda
  - cudnn
  - 显卡
  - 显存
  - GPU使用率
  - nvidia-smi
  - nccl
---

# NVIDIA GPU 运维专家知识

## 概述

NVIDIA GPU 是 AI/HPC 计算的主力硬件，需要正确的驱动、CUDA 和工具链。

## nvidia-smi 命令

### 基础查看

```bash
# 标准状态显示
nvidia-smi

# 持续监控 (每秒刷新)
nvidia-smi -l 1

# 只显示 GPU 信息
nvidia-smi -L
```

### 自定义查询

```bash
# 查询使用率和显存
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv

# 查询温度和功耗
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,power.limit --format=csv

# 查询 GPU 进程
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# JSON 格式输出
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits
```

### 进程管理

```bash
# 查看 GPU 进程详情
nvidia-smi pmon -c 1

# 查看设备上的进程
fuser -v /dev/nvidia*

# 终止占用 GPU 的进程
kill -9 <pid>
```

### GPU 拓扑

```bash
# 查看 GPU 连接拓扑
nvidia-smi topo -m

# 查看 NVLink 状态
nvidia-smi nvlink -s
```

## CUDA 管理

### 版本检查

```bash
# 检查驱动版本 (从 nvidia-smi 输出)
nvidia-smi --query-gpu=driver_version --format=csv,noheader

# 检查 CUDA 版本
nvcc --version

# 检查 cuDNN 版本
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

### CUDA 环境变量

```bash
# 指定使用的 GPU
export CUDA_VISIBLE_DEVICES=0,1

# CUDA 路径
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## 常见问题解决

### CUDA out of memory

**诊断:**
```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

**解决方案:**
1. 减小 batch size
2. 使用梯度检查点
3. 使用混合精度 (FP16/BF16)
4. 清理 GPU 缓存: `torch.cuda.empty_cache()`
5. 使用模型并行

### GPU 利用率低

**诊断:**
```bash
nvidia-smi -l 1  # 观察利用率波动
```

**常见原因:**
1. 数据加载瓶颈 → 增加 num_workers
2. CPU 预处理慢 → 使用 GPU 预处理
3. 小 batch size → 增大 batch size
4. 频繁同步 → 减少 CPU-GPU 数据传输

### NCCL 错误

**诊断:**
```bash
# 检查 NCCL 版本
python -c "import torch; print(torch.cuda.nccl.version())"

# 设置调试
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

**常见解决:**
1. 确保所有节点 NCCL 版本一致
2. 检查 InfiniBand 或以太网配置
3. 设置 `NCCL_SOCKET_IFNAME` 指定网络接口

### 驱动/CUDA 版本不匹配

```bash
# 检查兼容性
nvidia-smi  # 显示驱动支持的最高 CUDA 版本
nvcc --version  # 显示安装的 CUDA 版本

# CUDA 版本必须 <= 驱动支持的版本
```

## 性能优化

### PyTorch 优化

```python
# 启用 cuDNN 自动调优
torch.backends.cudnn.benchmark = True

# 混合精度训练
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 多 GPU 训练

```python
# DataParallel (简单但效率低)
model = nn.DataParallel(model)

# DistributedDataParallel (推荐)
model = nn.parallel.DistributedDataParallel(model)
```

## 监控脚本

```bash
#!/bin/bash
# GPU 监控脚本

while true; do
    clear
    echo "=== GPU 状态 $(date) ==="
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
    echo ""
    echo "=== GPU 进程 ==="
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    sleep 5
done
```
