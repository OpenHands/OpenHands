#!/bin/bash
# סקריפט הפעלה מחדש של OpenHands
# נוצר: 2025-12-27

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "מפעיל מחדש OpenHands..."

# עצירה
"$SCRIPT_DIR/stop-openhands.sh"

# המתנה קצרה
sleep 3

# הפעלה
"$SCRIPT_DIR/start-openhands.sh"






