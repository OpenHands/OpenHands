#!/bin/bash
# סקריפט עצירה של OpenHands
# נוצר: 2025-12-27

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_success() {
    echo "[SUCCESS] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

stop_openhands() {
    log "עוצר OpenHands..."
    
    cd "$PROJECT_DIR" || {
        echo "לא ניתן לעבור לתיקיית הפרויקט"
        exit 1
    }
    
    if docker compose version &> /dev/null; then
        docker compose down
    else
        docker-compose down
    fi
    
    log_success "OpenHands נעצר"
}

# אופציונלי: עצירה עם מחיקת volumes
if [[ "${1:-}" == "--volumes" ]] || [[ "${1:-}" == "-v" ]]; then
    log "עוצר ומסיר volumes..."
    cd "$PROJECT_DIR" || exit 1
    if docker compose version &> /dev/null; then
        docker compose down -v
    else
        docker-compose down -v
    fi
    log_success "Volumes הוסרו"
else
    stop_openhands
fi






