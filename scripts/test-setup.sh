#!/bin/bash
# סקריפט בדיקת התקנה
# נוצר: 2025-12-27

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_success() {
    echo "✅ $1"
}

log_error() {
    echo "❌ $1"
}

test_services() {
    log "בודק שירותים..."
    
    # בדיקת Docker containers
    log "בודק Docker containers..."
    if docker ps --format '{{.Names}}' | grep -q "openhands"; then
        log_success "Docker containers רצים"
        docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "openhands|NAMES"
    else
        log_error "Docker containers לא רצים"
    fi
    echo ""
    
    # בדיקת פורטים
    log "בודק פורטים..."
    PORTS=(3000 4000 8000 8080 11434)
    for port in "${PORTS[@]}"; do
        if netstat -tuln 2>/dev/null | grep -q ":${port} " || ss -tuln 2>/dev/null | grep -q ":${port} "; then
            log_success "פורט $port פעיל"
        else
            log_error "פורט $port לא פעיל"
        fi
    done
    echo ""
    
    # בדיקת HTTP endpoints
    log "בודק HTTP endpoints..."
    
    # OpenHands UI
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        log_success "OpenHands UI זמין"
    else
        log_error "OpenHands UI לא זמין"
    fi
    
    # Ollama
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        log_success "Ollama API זמין"
        MODELS=$(curl -s http://localhost:11434/api/tags | jq -r '.models[].name' 2>/dev/null | head -3)
        if [[ -n "$MODELS" ]]; then
            log "מודלים זמינים: $MODELS"
        fi
    else
        log_error "Ollama API לא זמין"
    fi
    
    # Security Analyzer
    if curl -s http://localhost:8000 > /dev/null 2>&1; then
        log_success "Security Analyzer זמין"
    else
        log_error "Security Analyzer לא זמין"
    fi
    
    # Code Server
    if curl -s http://localhost:8080 > /dev/null 2>&1; then
        log_success "Code Server זמין"
    else
        log_error "Code Server לא זמין"
    fi
    echo ""
}

test_websocket() {
    log "בודק WebSocket..."
    
    # בדיקת לוגים
    if docker ps --format '{{.Names}}' | grep -q "openhands-app"; then
        WS_ACCEPTED=$(docker logs openhands-app 2>&1 | grep -c "WebSocket.*accepted" || echo "0")
        WS_REJECTED=$(docker logs openhands-app 2>&1 | grep -c "WebSocket.*403\|connection rejected" || echo "0")
        
        if [[ $WS_ACCEPTED -gt 0 ]]; then
            log_success "WebSocket connections התקבלו: $WS_ACCEPTED"
        fi
        
        if [[ $WS_REJECTED -gt 0 ]]; then
            log_error "WebSocket connections נדחו: $WS_REJECTED"
        fi
    else
        log_error "openhands-app לא רץ"
    fi
    echo ""
}

test_runtime() {
    log "בודק Runtime containers..."
    
    RUNTIME_COUNT=$(docker ps --format '{{.Names}}' | grep -c "runtime" || echo "0")
    if [[ $RUNTIME_COUNT -gt 0 ]]; then
        log_success "Runtime containers פעילים: $RUNTIME_COUNT"
    else
        log "אין Runtime containers פעילים (זה תקין אם אין משימות פעילות)"
    fi
    echo ""
}

test_server_access() {
    log "בודק גישה לשרת..."
    
    if docker ps --format '{{.Names}}' | grep -q "openhands-app"; then
        # בדיקת volumes
        if docker exec openhands-app test -d /host/proc 2>/dev/null; then
            log_success "גישה ל-/host/proc"
        else
            log_error "אין גישה ל-/host/proc"
        fi
        
        if docker exec openhands-app test -S /var/run/docker.sock 2>/dev/null; then
            log_success "גישה ל-Docker socket"
        else
            log_error "אין גישה ל-Docker socket"
        fi
    else
        log_error "openhands-app לא רץ"
    fi
    echo ""
}

main() {
    echo "=========================================="
    echo "בדיקת התקנת OpenHands"
    echo "=========================================="
    echo ""
    
    test_services
    test_websocket
    test_runtime
    test_server_access
    
    echo "=========================================="
    echo "בדיקה הושלמה"
    echo "=========================================="
}

main "$@"






