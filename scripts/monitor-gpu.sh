#!/bin/bash
# Script לניטור GPU
# נוצר: 2025-12-30

set -euo pipefail

# בדוק אם nvidia-smi זמין
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi לא נמצא. ודא ש-NVIDIA drivers מותקנים."
    exit 1
fi

# אם יש פרמטר -w, הרץ watch
if [[ "${1:-}" == "-w" ]] || [[ "${1:-}" == "--watch" ]]; then
    watch -n 1 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv'
else
    echo "=========================================="
    echo "GPU Status"
    echo "=========================================="
    echo ""
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv
    echo ""
    echo "=========================================="
    echo "Processes using GPU:"
    echo "=========================================="
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
fi

