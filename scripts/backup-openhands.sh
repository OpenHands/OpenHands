#!/bin/bash
# Script לגיבוי אוטומטי של OpenHands
# נוצר: 2025-12-30

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-/mnt/nvme/backups/openhands}"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS="${RETENTION_DAYS:-7}"

# יצירת תיקיית גיבוי אם לא קיימת
mkdir -p "$BACKUP_DIR"

echo "=========================================="
echo "גיבוי OpenHands - $DATE"
echo "=========================================="
echo ""

# 1. גיבוי OpenHands State
echo "📦 גיבוי OpenHands State..."
if docker ps | grep -q "openhands-app"; then
    docker exec openhands-app tar czf /tmp/openhands-state-$DATE.tar.gz /app/.openhands-state 2>/dev/null || true
    docker cp openhands-app:/tmp/openhands-state-$DATE.tar.gz "$BACKUP_DIR/" 2>/dev/null || true
    docker exec openhands-app rm -f /tmp/openhands-state-$DATE.tar.gz 2>/dev/null || true
    echo "✅ State נשמר"
else
    echo "⚠️  OpenHands לא רץ, מדלג על גיבוי state"
fi

# 2. גיבוי Workspace
echo "📦 גיבוי Workspace..."
if [ -d "$PROJECT_DIR/workspace" ]; then
    tar czf "$BACKUP_DIR/workspace-$DATE.tar.gz" -C "$PROJECT_DIR" workspace 2>/dev/null || true
    echo "✅ Workspace נשמר"
else
    echo "⚠️  תיקיית workspace לא נמצאה"
fi

# 3. גיבוי Config
echo "📦 גיבוי Configuration..."
if [ -f "$PROJECT_DIR/config.toml" ]; then
    cp "$PROJECT_DIR/config.toml" "$BACKUP_DIR/config-$DATE.toml"
    echo "✅ Config נשמר"
fi

# 4. גיבוי docker-compose.yml
if [ -f "$PROJECT_DIR/docker-compose.yml" ]; then
    cp "$PROJECT_DIR/docker-compose.yml" "$BACKUP_DIR/docker-compose-$DATE.yml"
    echo "✅ docker-compose.yml נשמר"
fi

# 5. גיבוי .env (אם קיים)
if [ -f "$PROJECT_DIR/.env" ]; then
    cp "$PROJECT_DIR/.env" "$BACKUP_DIR/env-$DATE.env"
    echo "✅ .env נשמר"
fi

# 6. ניקוי גיבויים ישנים
echo ""
echo "🧹 ניקוי גיבויים ישנים (יותר מ-$RETENTION_DAYS ימים)..."
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete 2>/dev/null || true
find "$BACKUP_DIR" -name "*.toml" -mtime +$RETENTION_DAYS -delete 2>/dev/null || true
find "$BACKUP_DIR" -name "*.yml" -mtime +$RETENTION_DAYS -delete 2>/dev/null || true
find "$BACKUP_DIR" -name "*.env" -mtime +$RETENTION_DAYS -delete 2>/dev/null || true
echo "✅ ניקוי הושלם"

# 7. סיכום
echo ""
echo "=========================================="
echo "גיבוי הושלם בהצלחה!"
echo "מיקום: $BACKUP_DIR"
echo "גודל: $(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1 || echo 'לא זמין')"
echo "=========================================="

