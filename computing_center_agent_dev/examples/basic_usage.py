"""
ComputingCenterAgent 基础使用示例

这个文件展示如何使用算力中心运维 Agent。

运行方式:
    1. 确保已正确配置 config.toml
    2. 运行: python -m openhands.core.main -t "查看集群状态"

或者直接运行本脚本 (需要在 OpenHands 环境中):
    python examples/basic_usage.py
"""

import asyncio
from typing import Optional


def example_cli_commands():
    """CLI 命令示例"""
    print("""
    ============================================================
    算力中心运维 Agent CLI 使用示例
    ============================================================

    1. 基础查询
    -----------------------------------------------------------
    # 查看集群状态
    python -m openhands.core.main -t "显示所有节点状态"

    # 查看 GPU 使用情况
    python -m openhands.core.main -t "查看所有 GPU 的使用率"

    # 查看作业队列
    python -m openhands.core.main -t "显示我的作业列表"


    2. 作业管理
    -----------------------------------------------------------
    # 提交作业
    python -m openhands.core.main -t "提交 train.sh 到 gpu 分区"

    # 查看作业状态
    python -m openhands.core.main -t "作业 12345 的状态是什么"

    # 取消作业
    python -m openhands.core.main -t "取消作业 12345"


    3. 故障诊断
    -----------------------------------------------------------
    # 诊断作业失败
    python -m openhands.core.main -t "作业 12345 为什么失败了"

    # 检查节点状态
    python -m openhands.core.main -t "node01 节点是否正常"

    # 分析错误日志
    python -m openhands.core.main -t "分析 slurm-12345.err 中的错误"


    4. 复杂任务
    -----------------------------------------------------------
    # 集群健康检查
    python -m openhands.core.main -t "执行完整的集群健康检查"

    # 性能分析
    python -m openhands.core.main -t "分析 GPU 集群的性能瓶颈"

    # 资源统计
    python -m openhands.core.main -t "统计本月 GPU 使用情况"


    5. 使用配置文件
    -----------------------------------------------------------
    # 指定配置文件
    python -m openhands.core.main --config-file config.toml -t "查看集群状态"

    # 指定 Agent
    python -m openhands.core.main --agent-config ComputingCenterAgent -t "查看 GPU"


    ============================================================
    """)


def example_tool_usage():
    """工具使用示例"""
    print("""
    ============================================================
    工具直接调用示例
    ============================================================

    以下是 Agent 可用工具的使用方式:

    1. cluster_monitor - 集群监控
    -----------------------------------------------------------
    参数:
        - query_type: nodes | resources | queues | partitions | summary
        - node_name: (可选) 指定节点
        - partition: (可选) 指定分区
        - filter: (可选) all | idle | allocated | down

    示例:
        查看所有节点: query_type="nodes"
        查看空闲节点: query_type="nodes", filter="idle"
        查看 GPU 分区: query_type="partitions", partition="gpu"


    2. job_manager - 作业管理
    -----------------------------------------------------------
    参数:
        - action: submit | status | list | cancel | hold | release
        - job_id: 作业 ID
        - script_path: 脚本路径 (submit)
        - partition: (可选) 分区

    示例:
        提交作业: action="submit", script_path="job.sh"
        查看状态: action="status", job_id="12345"
        取消作业: action="cancel", job_id="12345"


    3. gpu_monitor - GPU 监控
    -----------------------------------------------------------
    参数:
        - query_type: status | utilization | memory | processes | temperature
        - gpu_id: (可选) GPU ID
        - node: (可选) 节点名称

    示例:
        查看所有 GPU: query_type="status"
        查看 GPU 0: query_type="utilization", gpu_id="0"
        查看 GPU 进程: query_type="processes"


    4. diagnostic - 故障诊断
    -----------------------------------------------------------
    参数:
        - check_type: node_health | network | storage | job_failure | full_check
        - target: (可选) 目标节点
        - job_id: (可选) 作业 ID

    示例:
        节点检查: check_type="node_health", target="node01"
        作业诊断: check_type="job_failure", job_id="12345"


    5. log_analyzer - 日志分析
    -----------------------------------------------------------
    参数:
        - analyze_type: job_output | system_log | error_pattern
        - job_id: (可选) 作业 ID
        - pattern: (可选) 搜索模式

    示例:
        分析作业日志: analyze_type="job_output", job_id="12345"
        搜索 CUDA 错误: analyze_type="error_pattern", pattern="CUDA"


    6. resource_manager - 资源管理
    -----------------------------------------------------------
    参数:
        - action: quota_info | priority | reservation | fairshare | accounting
        - user: (可选) 用户名
        - job_id: (可选) 作业 ID

    示例:
        查看配额: action="quota_info"
        查看优先级: action="priority", job_id="12345"


    ============================================================
    """)


def example_config():
    """配置示例"""
    print("""
    ============================================================
    配置文件示例 (config.toml)
    ============================================================

    [core]
    default_agent = "ComputingCenterAgent"
    runtime = "docker"

    [llm]
    model = "gpt-4o"
    api_key = "your-api-key"

    [agent.ComputingCenterAgent]
    # 集群配置
    cluster_type = "slurm"
    default_partition = "gpu"

    # 功能开关
    enable_gpu_monitor = true
    enable_job_manager = true
    enable_diagnostic = true

    # GPU 配置
    gpu_vendor = "nvidia"

    # 告警阈值
    alert_gpu_util_low = 30
    alert_gpu_memory_high = 90

    ============================================================
    """)


async def programmatic_example():
    """
    编程方式使用示例

    注意: 这个示例需要完整的 OpenHands 环境才能运行
    """
    try:
        from openhands.core.config import OpenHandsConfig, parse_arguments
        from openhands.core.main import run_controller
        from openhands.events.action import MessageAction

        # 配置
        config = OpenHandsConfig()
        config.default_agent = "ComputingCenterAgent"

        # 创建任务
        task = MessageAction(content="查看集群状态")

        # 运行
        state = await run_controller(
            config=config,
            initial_user_action=task,
        )

        print(f"最终状态: {state}")

    except ImportError as e:
        print(f"需要 OpenHands 环境: {e}")
    except Exception as e:
        print(f"运行错误: {e}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print(" 算力中心运维 CLI Agent 使用指南")
    print("=" * 60)

    print("\n选择要查看的示例:")
    print("1. CLI 命令示例")
    print("2. 工具使用示例")
    print("3. 配置文件示例")
    print("4. 全部显示")

    try:
        choice = input("\n请输入选项 (1-4): ").strip()
    except EOFError:
        choice = "4"

    if choice == "1":
        example_cli_commands()
    elif choice == "2":
        example_tool_usage()
    elif choice == "3":
        example_config()
    else:
        example_cli_commands()
        example_tool_usage()
        example_config()


if __name__ == "__main__":
    main()
