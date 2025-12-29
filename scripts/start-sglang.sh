#!/bin/bash
# סקריפט להפעלת SGLang ישירות על המערכת
# נוצר: 2025-12-27

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/home/noya/sglang.log"
PID_FILE="/home/noya/sglang.pid"

# בדיקה אם SGLang כבר רץ
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "SGLang כבר רץ (PID: $OLD_PID)"
        exit 0
    fi
fi

echo "🚀 מפעיל SGLang..."
echo "📋 לוגים: $LOG_FILE"

# הפעלת SGLang ברקע עם quantization כדי לחסוך זיכרון GPU
# אם יש בעיית זיכרון, נסה מודל קטן יותר או quantization
echo "🔄 מוודא SGLang מאזין ל-0.0.0.0 ולא רק ל-localhost"
echo "📦 משתמש במודל qwen2.5-coder:14b כפי שביקש המשתמש"
# מודל Coder-14B - מתאים לקוד
nohup python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-Coder-14B-Instruct \
    --host 0.0.0.0 \
    --port 11434 \
    --dtype bfloat16 \
    > "$LOG_FILE" 2>&1 &

SGLANG_PID=$!
echo $SGLANG_PID > "$PID_FILE"

echo "✅ SGLang הופעל (PID: $SGLANG_PID)"
echo "🌐 API: http://localhost:11435/v1"
echo "📋 לוגים: tail -f $LOG_FILE"

# המתנה קצרה לבדיקה
sleep 5
if ps -p "$SGLANG_PID" > /dev/null 2>&1; then
    echo "✅ SGLang רץ בהצלחה!"
else
    echo "❌ SGLang נכשל, בדוק לוגים: $LOG_FILE"
    exit 1
fi


