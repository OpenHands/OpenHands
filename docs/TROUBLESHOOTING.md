# פתרון בעיות OpenHands

## בעיות נפוצות ופתרונות

### 1. שירותים לא עולים

#### OpenHands App לא עולה
```bash
# בדוק לוגים
docker logs openhands-app

# בדוק health check
curl http://localhost:3002/api/health

# הפעל מחדש
cd /home/noya/OpenHands
docker-compose restart openhands
```

#### SGLang לא עולה
```bash
# בדוק GPU
nvidia-smi

# בדוק לוגים
docker logs openhands-sglang

# בדוק זיכרון GPU
docker exec openhands-sglang nvidia-smi
```

### 2. בעיות חיבור

#### לא ניתן להתחבר ל-LLM
```bash
# בדוק ש-SGLang פעיל
curl http://localhost:30000/health

# בדוק network
docker network inspect openhands-network

# בדוק IP
docker exec openhands-app ping -c 3 openhands-sglang
```

#### CORS Errors
```bash
# בדוק PERMITTED_CORS_ORIGINS
docker exec openhands-app env | grep CORS

# עדכן ב-docker-compose.yml
# PERMITTED_CORS_ORIGINS=http://10.0.0.13:3002
```

### 3. בעיות זיכרון

#### Out of Memory
```bash
# בדוק שימוש זיכרון
free -h
docker stats

# הגבל זיכרון ל-SGLang
# ב-docker-compose.yml:
# mem_limit: 16g
```

#### GPU Out of Memory
```bash
# בדוק שימוש GPU
nvidia-smi

# הקטן את mem-fraction-static
# ב-docker-compose.yml:
# --mem-fraction-static 0.6
```

### 4. בעיות דיסק

#### דיסק מלא
```bash
# בדוק שימוש
df -h

# נקה Docker
docker system prune -a

# נקה גיבויים ישנים
find /mnt/nvme/backups/openhands -mtime +7 -delete
```

### 5. בעיות Agent Server

#### Agent לא מגיב
```bash
# בדוק containers
docker ps | grep agent-server

# בדוק לוגים
docker logs oh-agent-server-XXXXX

# הפעל מחדש
docker restart oh-agent-server-XXXXX
```

### 6. בעיות Browser

#### Browser לא עובד
```bash
# בדוק Playwright
docker exec openhands-app playwright install chromium

# בדוק headless mode
# ב-config.toml:
# BROWSER_HEADLESS=false
```

## כלי אבחון

### Health Check
```bash
cd /home/noya/OpenHands
./scripts/health-check.py
```

### System Status
```bash
./scripts/monitor-openhands.sh
```

### Docker Status
```bash
docker-compose ps
docker stats
```

### Network Debugging
```bash
# בדוק connectivity
docker exec openhands-app ping -c 3 openhands-sglang
docker exec openhands-app curl http://openhands-sglang:30000/health

# בדוק DNS
docker exec openhands-app nslookup openhands-sglang
```

## לוגים

### מיקום לוגים
- OpenHands: `docker logs openhands-app`
- SGLang: `docker logs openhands-sglang`
- Agent Servers: `docker logs oh-agent-server-XXXXX`
- System: `/var/log/`

### צפייה בלוגים
```bash
# כל הלוגים
docker-compose logs -f

# לוגים של שירות ספציפי
docker-compose logs -f openhands

# לוגים אחרונים
docker-compose logs --tail=100
```

## איפוס

### איפוס מלא (זהירות!)
```bash
# עצור הכל
docker-compose down

# מחק volumes (מאבד נתונים!)
docker volume rm openhands-data

# הפעל מחדש
docker-compose up -d
```

### איפוס State בלבד
```bash
docker exec openhands-app rm -rf /app/.openhands-state/*
docker-compose restart openhands
```

## תמיכה

אם הבעיה נמשכת:
1. אסוף לוגים: `docker-compose logs > logs.txt`
2. הרץ health check: `./scripts/health-check.py > health.txt`
3. בדוק system info: `./scripts/server-info.py > system.txt`

