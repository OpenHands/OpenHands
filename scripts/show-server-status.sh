#!/bin/bash
# סקריפט להצגת סטטוס מלא של השרת
# נוצר: 2025-12-27

set -euo pipefail

echo "=========================================="
echo "📊 סטטוס מלא של השרת"
echo "=========================================="
echo ""

# תהליכים
echo "🔄 תהליכים פעילים (Top 20):"
echo "----------------------------------------"
if [ -d /host/proc ]; then
    echo "תהליכים מהשרת (דרך /host/proc):"
    ps aux 2>/dev/null | head -20 || echo "לא ניתן לגשת לתהליכים"
else
    ps aux | head -20
fi
echo ""

# Docker containers
echo "🐳 Docker Containers:"
echo "----------------------------------------"
if [ -S /var/run/docker.sock ]; then
    docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
else
    echo "לא ניתן לגשת ל-Docker socket"
fi
echo ""

# שירותי systemd
echo "⚙️  שירותי Systemd:"
echo "----------------------------------------"
if [ -d /host/run/systemd ]; then
    systemctl list-units --type=service --state=running --no-pager 2>/dev/null | head -20 || \
    echo "רשימת שירותים דרך /host/run/systemd"
else
    echo "לא ניתן לגשת ל-systemd"
fi
echo ""

# פורטים
echo "🌐 פורטים פעילים:"
echo "----------------------------------------"
if command -v netstat &> /dev/null; then
    netstat -tulnp 2>/dev/null | head -20
elif command -v ss &> /dev/null; then
    ss -tulnp 2>/dev/null | head -20
else
    echo "לא נמצא כלי לבדיקת פורטים"
fi
echo ""

# רשת
echo "📡 רשת:"
echo "----------------------------------------"
if [ -d /host/sys/class/net ]; then
    echo "ממשקי רשת:"
    ls -la /host/sys/class/net/ 2>/dev/null | head -10
else
    ip addr show 2>/dev/null | head -20 || echo "לא ניתן לבדוק רשת"
fi
echo ""

# דיסק
echo "💾 שימוש בדיסק:"
echo "----------------------------------------"
df -h | head -10
echo ""

# זיכרון
echo "🧠 זיכרון:"
echo "----------------------------------------"
free -h
echo ""

# CPU
echo "⚡ CPU:"
echo "----------------------------------------"
if [ -d /host/proc ]; then
    cat /host/proc/cpuinfo 2>/dev/null | grep -E "processor|model name" | head -10 || \
    cat /proc/cpuinfo | grep -E "processor|model name" | head -10
else
    cat /proc/cpuinfo | grep -E "processor|model name" | head -10
fi
echo ""

# לוגים אחרונים
echo "📋 לוגים אחרונים:"
echo "----------------------------------------"
if [ -d /host/var/log ]; then
    echo "לוגים מ-/host/var/log:"
    ls -lht /host/var/log/*.log 2>/dev/null | head -10 || echo "אין לוגים"
else
    echo "לא ניתן לגשת ללוגים"
fi
echo ""

echo "=========================================="
echo "✅ סיכום הושלם"
echo "=========================================="






