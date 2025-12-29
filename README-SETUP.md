# מדריך התקנה והפעלה מקצועי - OpenHands

**תאריך יצירה:** 2025-12-27  
**גרסה:** Professional Setup v1.0

## תוכן עניינים

1. [דרישות מערכת](#דרישות-מערכת)
2. [התקנה](#התקנה)
3. [הפעלה](#הפעלה)
4. [שימוש](#שימוש)
5. [ניהול](#ניהול)
6. [גישה לשרת](#גישה-לשרת)
7. [פתרון בעיות](#פתרון-בעיות)
8. [תיעוד נוסף](#תיעוד-נוסף)

---

## דרישות מערכת

### חובה
- **Docker** >= 20.10
- **Docker Compose** >= 2.0 (או docker-compose >= 1.29)
- **Linux** (Ubuntu 22.04+ מומלץ)
- **RAM**: מינימום 8GB, מומלץ 16GB+
- **דיסק**: מינימום 20GB פנוי

### אופציונלי
- **GPU** עם NVIDIA drivers (למודלים גדולים)
- **Internet connection** (להורדת images ומודלים)

---

## התקנה

### 1. הכנת סביבה

```bash
cd /home/noya/OpenHands
```

### 2. בדיקת קבצים

ודא שקיימים הקבצים הבאים:
- `docker-compose.yml`
- `config.toml`
- `.env`
- `scripts/` (תיקיית סקריפטים)

### 3. הגדרת משתני סביבה

ערוך את `.env` לפי הצרכים שלך:

```bash
nano .env
```

**משתנים חשובים:**
- `CODE_SERVER_PASSWORD` - סיסמה ל-Code Server
- `LLM_MODEL` - מודל LLM (ברירת מחדל: ollama/qwen2.5:14b)
- `OLLAMA_MODELS` - נתיב למודלי Ollama

---

## הפעלה

### הפעלה מהירה

```bash
./scripts/start-openhands.sh
```

### הפעלה ידנית

```bash
docker compose up -d
```

### בדיקת סטטוס

```bash
./scripts/monitor-openhands.sh
```

או:

```bash
docker compose ps
```

---

## שימוש

### גישה לשירותים

| שירות | כתובת | תיאור |
|-------|-------|-------|
| **OpenHands UI** | http://localhost:3000 | ממשק המשתמש הראשי |
| **Ollama API** | http://localhost:11434 | API למודלי LLM |
| **Security Analyzer** | http://localhost:8000 | ניתוח אבטחה |
| **Code Server** | http://localhost:8080 | IDE מבוסס web |
| **LiteLLM Gateway** | http://localhost:4000 | Gateway ל-LLM |

### שימוש ב-OpenHands

1. פתח דפדפן וגש ל: http://localhost:3000
2. בחר מודל LLM (למשל: qwen2.5:14b)
3. בחר Agent (למשל: CodeActAgent)
4. התחל משימה

### דוגמאות משימות

```
כתוב סקריפט Python שמדפיס "Hello World"
```

```
בנה API פשוט ב-Flask עם endpoint אחד
```

```
בדוק את הלוגים של השרת ותן סיכום
```

---

## ניהול

### סקריפטי ניהול

#### הפעלה
```bash
./scripts/start-openhands.sh
```

#### עצירה
```bash
./scripts/stop-openhands.sh
```

#### עצירה עם מחיקת volumes
```bash
./scripts/stop-openhands.sh --volumes
```

#### הפעלה מחדש
```bash
./scripts/restart-openhands.sh
```

#### ניטור
```bash
# סטטוס חד-פעמי
./scripts/monitor-openhands.sh

# ניטור שוטף
./scripts/monitor-openhands.sh -f
```

### פקודות Docker Compose

```bash
# הפעלה
docker compose up -d

# עצירה
docker compose down

# לוגים
docker compose logs -f

# עדכון images
docker compose pull
docker compose up -d

# סטטוס
docker compose ps
```

---

## גישה לשרת

OpenHands מוגדר עם גישה מלאה לשרת כדי שהאגנט יוכל לראות ולנהל את כל המערכת.

### מה נגיש

- **תהליכים** - דרך `/host/proc`
- **שירותי systemd** - דרך `/host/run/systemd`
- **לוגים** - דרך `/host/var/log`
- **Docker** - דרך `/var/run/docker.sock`
- **קבצי מערכת** - דרך `/host/etc`, `/host/usr/bin`
- **רשת** - מידע על ממשקי רשת

### סקריפטים לגישה לשרת

#### הצגת סטטוס מלא
```bash
# בתוך הקונטיינר
docker exec openhands-app /workspace/scripts/show-server-status.sh

# או מהשרת
./scripts/show-server-status.sh
```

#### מידע JSON
```bash
docker exec openhands-app python3 /workspace/scripts/server-info.py
```

#### ניטור שוטף
```bash
docker exec openhands-app /workspace/scripts/server-monitor.sh
```

### שימוש באגנט

האגנט יכול לבצע:

```bash
# בדיקת תהליכים
ps aux

# בדיקת שירותים
systemctl status <service>

# בדיקת Docker
docker ps

# קריאת לוגים
cat /host/var/log/syslog

# בדיקת רשת
ip addr show
```

---

## פתרון בעיות

### בעיות נפוצות

#### 1. פורט תפוס

**בעיה:** `bind: address already in use`

**פתרון:**
```bash
# מצא תהליך על הפורט
lsof -i :3000

# עצור את התהליך
kill <PID>

# או שנה פורט ב-docker-compose.yml
```

#### 2. WebSocket לא עובד

**בעיה:** `403 Forbidden` על WebSocket connections

**פתרון:**
- בדוק לוגים: `docker compose logs openhands-app`
- ודא שהקוד תוקן (הקוד הנוכחי אמור להיות גמיש)
- הפעל מחדש: `./scripts/restart-openhands.sh`

#### 3. מודל לא נטען

**בעיה:** Ollama לא מוצא מודל

**פתרון:**
```bash
# בדוק מודלים זמינים
curl http://localhost:11434/api/tags

# טען מודל
ollama pull qwen2.5:14b
```

#### 4. אין גישה לשרת

**בעיה:** האגנט לא רואה תהליכים/שירותים

**פתרון:**
- ודא ש-privileged: true ב-docker-compose.yml
- ודא ש-volumes מוגדרים נכון
- הפעל מחדש: `./scripts/restart-openhands.sh`

### בדיקת תקינות

הרץ את סקריפט הבדיקה:

```bash
./scripts/test-setup.sh
```

### לוגים

```bash
# כל השירותים
docker compose logs -f

# שירות ספציפי
docker compose logs -f openhands-app

# אחרונים 100 שורות
docker compose logs --tail=100
```

---

## תיעוד נוסף

### קבצי קונפיגורציה

- **docker-compose.yml** - הגדרת כל השירותים
- **config.toml** - קונפיגורציית OpenHands
- **.env** - משתני סביבה

### סקריפטים

כל הסקריפטים נמצאים ב-`scripts/`:

- `start-openhands.sh` - הפעלה
- `stop-openhands.sh` - עצירה
- `restart-openhands.sh` - הפעלה מחדש
- `monitor-openhands.sh` - ניטור
- `show-server-status.sh` - סטטוס שרת
- `server-info.py` - מידע JSON
- `server-monitor.sh` - ניטור שוטף
- `test-setup.sh` - בדיקת התקנה

### מבנה תיקיות

```
OpenHands/
├── docker-compose.yml      # הגדרת Docker Compose
├── config.toml             # קונפיגורציית OpenHands
├── .env                    # משתני סביבה
├── scripts/                # סקריפטי ניהול
│   ├── start-openhands.sh
│   ├── stop-openhands.sh
│   ├── restart-openhands.sh
│   ├── monitor-openhands.sh
│   ├── show-server-status.sh
│   ├── server-info.py
│   ├── server-monitor.sh
│   └── test-setup.sh
├── workspace/              # תיקיית עבודה
└── logs/                   # לוגים
```

---

## תמיכה

### בעיות טכניות

1. בדוק לוגים: `docker compose logs -f`
2. הרץ בדיקה: `./scripts/test-setup.sh`
3. בדוק דוקומנטציה: [OpenHands Docs](https://docs.openhands.dev)

### עדכונים

```bash
# עדכן images
docker compose pull

# הפעל מחדש
./scripts/restart-openhands.sh
```

---

## רישיון

OpenHands מופץ תחת רישיון MIT.

---

**נוצר על ידי:** Auto (Cursor AI)  
**תאריך:** 2025-12-27






