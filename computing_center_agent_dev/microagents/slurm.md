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
  - scontrol
  - sacct
  - 作业调度
  - job scheduler
  - 分区
  - partition
---

# Slurm 作业调度系统专家知识

## 概述

Slurm (Simple Linux Utility for Resource Management) 是最流行的 HPC 作业调度系统。

## 核心命令

### 集群信息 (sinfo)

```bash
# 查看节点状态
sinfo -N -l

# 查看分区信息
sinfo -s

# 查看特定分区
sinfo -p gpu

# 自定义输出格式
sinfo -o "%P %a %l %D %t %c %m %G"
# %P: 分区名, %a: 可用性, %l: 时间限制, %D: 节点数
# %t: 状态, %c: CPU数, %m: 内存, %G: GPU
```

### 作业管理 (squeue/sbatch/scancel)

```bash
# 查看我的作业
squeue -u $USER

# 详细作业信息
squeue -l -u $USER

# 自定义输出
squeue -o "%.10i %.9P %.20j %.8u %.8T %.10M %.10l %.6D %R"

# 提交作业
sbatch job.sh

# 提交到指定分区
sbatch -p gpu job.sh

# 取消作业
scancel <job_id>

# 取消所有我的作业
scancel -u $USER
```

### 作业控制 (scontrol)

```bash
# 查看作业详情
scontrol show job <job_id>

# 暂停作业
scontrol hold <job_id>

# 释放作业
scontrol release <job_id>

# 修改作业时间限制
scontrol update job <job_id> TimeLimit=48:00:00

# 查看节点详情
scontrol show node <node_name>
```

### 历史作业 (sacct)

```bash
# 查看历史作业
sacct -u $USER

# 指定时间范围
sacct -u $USER --starttime=2024-01-01 --endtime=2024-01-31

# 详细格式
sacct -j <job_id> --format=JobID,JobName,Elapsed,State,ExitCode,MaxRSS

# 查看作业效率
seff <job_id>
```

## 作业脚本模板

### 基础模板

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

echo "Job started at $(date)"
echo "Running on node: $SLURMD_NODENAME"

# 你的程序
./my_program

echo "Job finished at $(date)"
```

### GPU 作业模板

```bash
#!/bin/bash
#SBATCH --job-name=gpu_train
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

module load cuda/11.8
source ~/venv/bin/activate

python train.py
```

### 多节点 GPU 作业

```bash
#!/bin/bash
#SBATCH --job-name=multi_gpu
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00

module load cuda/11.8 nccl

# 获取主节点
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

srun python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    train_distributed.py
```

## 常见问题解决

### 作业一直 Pending

**原因检查:**
```bash
squeue -j <job_id> -o "%i %r"
# 查看 REASON 字段
```

**常见原因:**
- `Resources`: 等待资源
- `Priority`: 优先级低
- `QOSMaxJobsPerUserLimit`: 超过用户作业限制
- `ReqNodeNotAvail`: 请求的节点不可用

### 作业失败排查

```bash
# 查看退出状态
sacct -j <job_id> --format=JobID,State,ExitCode,DerivedExitCode

# 常见退出码
# 0: 成功
# 1: 一般错误
# 137: 被 OOM Killer 终止
# 143: 被 SIGTERM 终止
```

### 资源请求建议

- CPU 密集型: 多 CPU，少内存
- 内存密集型: 少 CPU，多内存
- GPU 训练: GPU + 足够 CPU + 大内存
