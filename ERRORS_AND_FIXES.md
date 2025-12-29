# סיכום שגיאות ותיקונים - OpenHands

**תאריך:** 2025-12-27

## ✅ תיקונים שבוצעו

### 1. Ollama Health Check
**בעיה:** Health check נכשל כי curl/wget לא קיימים בקונטיינר  
**פתרון:** שינוי ל-`ollama list`  
**סטטוס:** ✅ תוקן - Ollama עכשיו healthy

### 2. פורטים תפוסים
**בעיה:** 
- פורט 3000 תפוס על ידי Cursor
- פורט 8080 תפוס על ידי Cursor

**פתרון:** 
- OpenHands UI: שונה ל-3002
- Code Server: שונה ל-8081

**סטטוס:** ✅ תוקן

### 3. depends_on condition
**בעיה:** openhands-app תלוי ב-Ollama להיות healthy, אבל health check נכשל  
**פתרון:** שינוי ל-`service_started` במקום `service_healthy`  
**סטטוס:** ✅ תוקן

## ❌ בעיות שנותרו

### 1. openhands-app - חסר tmux
**שגיאה:**
```
ValueError: tmux is not properly installed or available on the path.
```

**סיבה:** הקונטיינר של OpenHands דורש tmux אבל הוא לא מותקן  
**פתרון אפשרי:**
- צריך לבנות image מותאם עם tmux
- או להשתמש ב-image אחר
- או להתקין tmux בקונטיינר

**סטטוס:** ❌ לא תוקן - דורש שינוי ב-image

### 2. litellm-gateway - Platform Mismatch
**שגיאה:**
```
exec /bin/sh: exec format error
The requested image's platform (linux/arm64/v8) does not match the detected host platform (linux/amd64/v3)
```

**סיבה:** Image מיועד ל-arm64 אבל השרת הוא amd64  
**פתרון:** הושבת זמנית (commented out ב-docker-compose.yml)  
**סטטוס:** ⚠️ הושבת זמנית

### 3. code-server - Permission Denied
**שגיאה:**
```
EACCES: permission denied, mkdir '/home/coder/.local/share/code-server/coder-logs'
```

**סיבה:** בעיית הרשאות  
**פתרון:** הוספת `user: "1000:1000"` ב-docker-compose.yml  
**סטטוס:** ⚠️ תוקן חלקית - צריך לבדוק

## 📊 סטטוס שירותים

| שירות | סטטוס | פורט | הערות |
|-------|-------|------|-------|
| **Ollama** | ✅ Healthy | 11434 | עובד מצוין |
| **Security Analyzer** | ✅ Healthy | 8000 | עובד מצוין |
| **Code Server** | ⚠️ Starting | 8081 | בעיית הרשאות |
| **OpenHands App** | ❌ Failed | 3002 | חסר tmux |
| **LiteLLM Gateway** | ❌ Disabled | - | Platform mismatch |

## 🔧 פתרונות מומלצים

### לפתור את בעיית tmux:

**אפשרות 1:** יצירת Dockerfile מותאם
```dockerfile
FROM ghcr.io/all-hands-ai/openhands:latest
RUN apt-get update && apt-get install -y tmux && rm -rf /var/lib/apt/lists/*
```

**אפשרות 2:** התקנת tmux בקונטיינר רץ
```bash
docker exec -it openhands-app apt-get update
docker exec -it openhands-app apt-get install -y tmux
docker restart openhands-app
```

**אפשרות 3:** שימוש ב-runtime אחר (docker במקום local)

### לפתור את litellm-gateway:

**אפשרות 1:** שימוש ב-image מותאם ל-amd64
```yaml
litellm-gateway:
  image: ghcr.io/berriai/litellm:latest-amd64
  platform: linux/amd64
```

**אפשרות 2:** השארתו מושבת (לא קריטי)

## 📝 פורטים סופיים

- **OpenHands UI:** http://localhost:3002 (לא עובד - חסר tmux)
- **Ollama API:** http://localhost:11434 ✅
- **Security Analyzer:** http://localhost:8000 ✅
- **Code Server:** http://localhost:8081 ⚠️
- **LiteLLM Gateway:** מושבת

## 🎯 צעדים הבאים

1. **לתקן את בעיית tmux** - זה הקריטי ביותר
2. **לבדוק את code-server** - לוודא שהרשאות תקינות
3. **להחליט על litellm-gateway** - האם צריך אותו

## 📋 פקודות לבדיקה

```bash
# בדיקת שירותים
docker compose ps

# לוגים של openhands-app
docker compose logs openhands-app

# ניסיון להתקין tmux
docker exec -it openhands-app bash
apt-get update && apt-get install -y tmux

# בדיקת UI
curl http://localhost:3002
```






