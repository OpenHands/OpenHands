# OpenHands - מדריך ניטור וגיבויים

## שירותי ניטור

### Prometheus
- **URL**: http://10.0.0.13:9090
- **תפקיד**: איסוף ומאגר מטריקות
- **Retention**: 30 ימים

### Grafana
- **URL**: http://10.0.0.13:3001
- **Username**: `admin`
- **Password**: (מוגדר ב-`.env` או `admin`)
- **תפקיד**: ויזואליזציה ודשבורדים

## Scripts זמינים

### Health Check
```bash
./scripts/health-check.py
```
בודק את מצב כל השירותים במערכת.

### Backup
```bash
./scripts/backup-openhands.sh
```
יוצר גיבוי של State, Workspace, ו-Config.

### GPU Monitoring
```bash
./scripts/monitor-gpu.sh          # בדיקה חד-פעמית
./scripts/monitor-gpu.sh -w        # ניטור רציף
```

### Cleanup
```bash
./scripts/cleanup-old-containers.sh
```
מנקה containers, images, ו-volumes ישנים.

### System Monitor
```bash
./scripts/monitor-openhands.sh     # סטטוס
./scripts/monitor-openhands.sh -f  # לוגים רציפים
```

## הגדרת גיבויים אוטומטיים

### Crontab
```bash
# ערוך crontab
crontab -e

# הוסף גיבוי יומי ב-2:00
0 2 * * * /home/noya/OpenHands/scripts/backup-openhands.sh >> /var/log/openhands-backup.log 2>&1
```

## תיעוד נוסף

- `docs/MONITORING.md` - מדריך ניטור מפורט
- `docs/BACKUP.md` - מדריך גיבויים
- `docs/TROUBLESHOOTING.md` - פתרון בעיות

## הפעלת שירותי ניטור

```bash
cd /home/noya/OpenHands
docker-compose up -d prometheus grafana
```

## גישה ראשונית ל-Grafana

1. גש ל-http://10.0.0.13:3001
2. התחבר עם `admin` / `admin`
3. שנה סיסמה בהתחברות הראשונה
4. Prometheus כבר מוגדר כ-Data Source

## מטריקות חשובות

- `up{job="openhands-app"}` - האם OpenHands פעיל
- `node_cpu_seconds_total` - שימוש CPU
- `node_memory_MemAvailable_bytes` - זיכרון זמין
- `container_memory_usage_bytes` - שימוש זיכרון containers

