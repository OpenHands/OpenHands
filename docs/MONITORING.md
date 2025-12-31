# מדריך ניטור OpenHands

## סקירה כללית

מערכת הניטור של OpenHands כוללת:
- **Prometheus** - איסוף ומאגר מטריקות
- **Grafana** - ויזואליזציה ודשבורדים
- **Node Exporter** - מטריקות מערכת
- **Health Checks** - בדיקות תקינות שירותים

## גישה לשירותים

### Prometheus
- **URL**: http://10.0.0.13:9090
- **תיאור**: ממשק לשאילתות מטריקות
- **דוגמה לשאילתה**: `up{job="openhands-app"}`

### Grafana
- **URL**: http://10.0.0.13:3001
- **Username**: `admin`
- **Password**: (מוגדר ב-`.env` או `admin` כברירת מחדל)
- **תיאור**: דשבורדים ויזואליים

## Health Checks

### הרצת Health Check ידני
```bash
cd /home/noya/OpenHands
./scripts/health-check.py
```

### בדיקת שירות בודד
```bash
# OpenHands
curl http://localhost:3002/api/health

# SGLang
curl http://localhost:30000/health

# Invariant Server
curl http://localhost:8000/

# Code Server
curl http://localhost:8081/healthz
```

## מטריקות חשובות

### OpenHands App
- `openhands_conversations_active` - מספר שיחות פעילות
- `openhands_requests_total` - סך הבקשות
- `openhands_requests_duration_seconds` - זמן תגובה

### System
- `node_cpu_seconds_total` - שימוש CPU
- `node_memory_MemAvailable_bytes` - זיכרון זמין
- `node_disk_io_time_seconds_total` - פעילות דיסק

### Docker
- `container_cpu_usage_seconds_total` - שימוש CPU של containers
- `container_memory_usage_bytes` - שימוש זיכרון של containers

## דשבורדים מומלצים

1. **System Overview** - סקירה כללית של המערכת
2. **OpenHands Performance** - ביצועי OpenHands
3. **LLM Metrics** - מטריקות של SGLang
4. **Container Resources** - משאבי containers

## פתרון בעיות

### Prometheus לא אוסף מטריקות
```bash
# בדוק את הקונפיגורציה
docker exec openhands-prometheus cat /etc/prometheus/prometheus.yml

# בדוק את הלוגים
docker logs openhands-prometheus
```

### Grafana לא מציגה נתונים
1. בדוק חיבור ל-Prometheus: Configuration → Data Sources
2. ודא ש-Prometheus פעיל
3. בדוק את הלוגים: `docker logs openhands-grafana`

## הגדרות מתקדמות

### הגדרת Alerts
ערוך את `monitoring/prometheus/alerts.yml` והוסף rules ל-`prometheus.yml`:
```yaml
rule_files:
  - "alerts.yml"
```

### הגדרת Retention
בדוק את `docker-compose.yml`:
```yaml
- '--storage.tsdb.retention.time=30d'
```

## תחזוקה

### ניקוי נתונים ישנים
```bash
docker exec openhands-prometheus promtool tsdb clean --older-than 30d
```

### גיבוי נתונים
```bash
docker cp openhands-prometheus:/prometheus ./backups/prometheus-$(date +%Y%m%d)
```

