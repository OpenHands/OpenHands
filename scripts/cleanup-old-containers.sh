#!/bin/bash
# Script לניקוי containers ישנים
# נוצר: 2025-12-30

set -euo pipefail

echo "=========================================="
echo "ניקוי Containers ישנים"
echo "=========================================="
echo ""

# מחק containers שנעצרו
echo "🗑️  מחק containers שנעצרו..."
docker container prune -f

# מחק images לא בשימוש
echo "🗑️  מחק images לא בשימוש..."
docker image prune -a -f --filter "until=168h"  # 7 ימים

# מחק volumes לא בשימוש
echo "🗑️  מחק volumes לא בשימוש..."
docker volume prune -f

# מחק networks לא בשימוש
echo "🗑️  מחק networks לא בשימוש..."
docker network prune -f

# מחק build cache
echo "🗑️  מחק build cache..."
docker builder prune -f

echo ""
echo "=========================================="
echo "ניקוי הושלם!"
echo "=========================================="

# הצג מקום שפונה
echo ""
echo "מקום שפונה:"
docker system df

