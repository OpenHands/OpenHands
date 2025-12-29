#!/bin/bash
# סקריפט ניטור שוטף של השרת
# נוצר: 2025-12-27

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERVAL=${1:-30}  # ברירת מחדל: 30 שניות

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

monitor_loop() {
    while true; do
        clear
        echo "=========================================="
        echo "ניטור שוטף - $(date)"
        echo "=========================================="
        echo ""
        
        "$SCRIPT_DIR/show-server-status.sh"
        
        echo ""
        echo "רענון בעוד $INTERVAL שניות... (Ctrl+C לעצירה)"
        sleep "$INTERVAL"
    done
}

# טיפול ב-Ctrl+C
trap 'echo ""; log "ניטור נעצר"; exit 0' INT

log "מתחיל ניטור שוטף (כל $INTERVAL שניות)"
monitor_loop






