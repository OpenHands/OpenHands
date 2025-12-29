#!/bin/bash
# סקריפט ניטור של OpenHands
# נוצר: 2025-12-27

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

show_status() {
    echo "=========================================="
    echo "סטטוס OpenHands"
    echo "=========================================="
    echo ""
    
    cd "$PROJECT_DIR" || exit 1
    
    if docker compose version &> /dev/null; then
        docker compose ps
    else
        docker-compose ps
    fi
    
    echo ""
    echo "פורטים פעילים:"
    netstat -tuln 2>/dev/null | grep -E "3000|4000|8000|8080|11434" || \
    ss -tuln 2>/dev/null | grep -E "3000|4000|8000|8080|11434" || \
    echo "לא ניתן לבדוק פורטים"
    
    echo ""
    echo "לוגים אחרונים:"
    if docker compose version &> /dev/null; then
        docker compose logs --tail=10
    else
        docker-compose logs --tail=10
    fi
}

# אם יש פרמטר -f, הרץ עם follow
if [[ "${1:-}" == "-f" ]] || [[ "${1:-}" == "--follow" ]]; then
    cd "$PROJECT_DIR" || exit 1
    if docker compose version &> /dev/null; then
        docker compose logs -f
    else
        docker-compose logs -f
    fi
else
    show_status
fi






