#!/bin/bash
# סקריפט הפעלה מקצועי של OpenHands
# נוצר: 2025-12-27

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_FILE="$PROJECT_DIR/logs/startup.log"

# יצירת תיקיית לוגים
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo "[SUCCESS] $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# בדיקת דרישות
check_requirements() {
    log "בודק דרישות מערכת..."
    
    # בדיקת Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker לא מותקן"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker לא רץ או אין הרשאות"
        exit 1
    fi
    log_success "Docker זמין"
    
    # בדיקת docker-compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "docker-compose לא מותקן"
        exit 1
    fi
    log_success "docker-compose זמין"
    
    # בדיקת קובץ docker-compose.yml
    if [[ ! -f "$PROJECT_DIR/docker-compose.yml" ]]; then
        log_error "קובץ docker-compose.yml לא נמצא"
        exit 1
    fi
    log_success "קובץ docker-compose.yml נמצא"
}

# עצירת קונטיינרים ישנים
stop_old_containers() {
    log "בודק קונטיינרים ישנים..."
    
    OLD_CONTAINERS=("openhands-invariant-server" "openhands-app" "openhands-sglang")
    
    for container in "${OLD_CONTAINERS[@]}"; do
        if docker ps -a --format '{{.Names}}' | grep -q "^${container}$"; then
            log "עוצר קונטיינר ישן: $container"
            docker stop "$container" 2>/dev/null || true
            docker rm "$container" 2>/dev/null || true
        fi
    done
}

# הפעלת OpenHands
start_openhands() {
    log "מפעיל OpenHands..."
    
    cd "$PROJECT_DIR" || {
        log_error "לא ניתן לעבור לתיקיית הפרויקט"
        exit 1
    }
    
    # טעינת משתני סביבה
    if [[ -f .env ]]; then
        log "טוען משתני סביבה מ-.env"
        set -a
        source .env
        set +a
    fi
    
    # הפעלה
    log "מריץ docker-compose up -d..."
    if docker compose version &> /dev/null; then
        docker compose up -d
    else
        docker-compose up -d
    fi
    
    log_success "docker-compose הופעל"
}

# המתנה לשירותים
wait_for_services() {
    log "ממתין לשירותים להתחיל..."
    
    SERVICES=("openhands-sglang" "openhands-app" "openhands-invariant-server")
    
    for service in "${SERVICES[@]}"; do
        log "ממתין ל-$service..."
        local max_attempts=30
        local attempt=0
        
        while [[ $attempt -lt $max_attempts ]]; do
            if docker ps --format '{{.Names}}' | grep -q "^${service}$"; then
                STATUS=$(docker inspect "$service" --format '{{.State.Status}}' 2>/dev/null)
                if [[ "$STATUS" == "running" ]]; then
                    log_success "$service רץ"
                    break
                fi
            fi
            sleep 2
            attempt=$((attempt + 1))
        done
        
        if [[ $attempt -eq $max_attempts ]]; then
            log_error "$service לא התחיל תוך 60 שניות"
        fi
    done
}

# בדיקת health
check_health() {
    log "בודק health של שירותים..."
    
    sleep 10
    
    # בדיקת SGLang
    if curl -s http://localhost:11435/health &> /dev/null; then
        log_success "SGLang עובד"
    else
        log_error "SGLang לא עובד"
    fi
    
    # בדיקת OpenHands
    if curl -s http://localhost:3000 &> /dev/null; then
        log_success "OpenHands UI עובד"
    else
        log_error "OpenHands UI לא עובד"
    fi
    
    # בדיקת Security Analyzer
    if curl -s http://localhost:8000 &> /dev/null; then
        log_success "Security Analyzer עובד"
    else
        log_error "Security Analyzer לא עובד"
    fi
}

# הצגת מידע
show_info() {
    echo ""
    echo "=========================================="
    echo "OpenHands הופעל בהצלחה!"
    echo "=========================================="
    echo ""
    echo "שירותים:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "openhands|NAMES"
    echo ""
    echo "גישה:"
    echo "  🌐 OpenHands UI: http://localhost:3002"
    echo "  🚀 SGLang API: http://localhost:11435"
    echo "  🔒 Security Analyzer: http://localhost:8000"
    echo "  💻 Code Server: http://localhost:8081"
    echo ""
    echo "לוגים:"
    echo "  📋 docker-compose logs -f"
    echo "  📋 $LOG_FILE"
    echo ""
}

# פונקציה ראשית
main() {
    echo "=========================================="
    echo "הפעלת OpenHands - סקריפט מקצועי"
    echo "=========================================="
    echo ""
    
    check_requirements
    echo ""
    
    stop_old_containers
    echo ""
    
    start_openhands
    echo ""
    
    wait_for_services
    echo ""
    
    check_health
    echo ""
    
    show_info
    
    log_success "הפעלה הושלמה!"
}

# הרצה
main "$@"




