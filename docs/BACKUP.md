# מדריך גיבוי OpenHands

## סקירה כללית

מערכת הגיבויים שומרת:
- OpenHands State (שיחות, הגדרות)
- Workspace (פרויקטים)
- Configuration files
- Docker Compose configuration

## הרצת גיבוי

### גיבוי ידני
```bash
cd /home/noya/OpenHands
./scripts/backup-openhands.sh
```

### גיבוי אוטומטי (Cron)
הוסף ל-crontab:
```bash
# גיבוי יומי ב-2:00 בלילה
0 2 * * * /home/noya/OpenHands/scripts/backup-openhands.sh >> /var/log/openhands-backup.log 2>&1
```

## מיקום גיבויים

ברירת מחדל: `/mnt/nvme/backups/openhands`

לשינוי המיקום:
```bash
export BACKUP_DIR="/path/to/backup"
./scripts/backup-openhands.sh
```

## שמירת גיבויים

ברירת מחדל: 7 ימים

לשינוי תקופת השמירה:
```bash
export RETENTION_DAYS=14
./scripts/backup-openhands.sh
```

## שחזור מגיבוי

### שחזור State
```bash
# עצור את OpenHands
cd /home/noya/OpenHands
docker-compose stop openhands

# שחזר את ה-state
docker cp /mnt/nvme/backups/openhands/openhands-state-YYYYMMDD_HHMMSS.tar.gz openhands-app:/tmp/
docker exec openhands-app tar xzf /tmp/openhands-state-YYYYMMDD_HHMMSS.tar.gz -C /
docker exec openhands-app rm /tmp/openhands-state-YYYYMMDD_HHMMSS.tar.gz

# הפעל מחדש
docker-compose start openhands
```

### שחזור Workspace
```bash
cd /home/noya/OpenHands
tar xzf /mnt/nvme/backups/openhands/workspace-YYYYMMDD_HHMMSS.tar.gz
```

### שחזור Config
```bash
cp /mnt/nvme/backups/openhands/config-YYYYMMDD_HHMMSS.toml /home/noya/OpenHands/config.toml
```

## גיבוי ל-Cloud

### AWS S3
```bash
# התקן AWS CLI
pip install awscli

# העלה גיבוי
aws s3 cp /mnt/nvme/backups/openhands/ s3://your-bucket/openhands/ --recursive
```

### Google Cloud Storage
```bash
# התקן gsutil
# העלה גיבוי
gsutil -m cp -r /mnt/nvme/backups/openhands/ gs://your-bucket/openhands/
```

## אימות גיבויים

### בדיקת תקינות קובץ
```bash
tar tzf /mnt/nvme/backups/openhands/workspace-YYYYMMDD_HHMMSS.tar.gz
```

### בדיקת גודל
```bash
du -sh /mnt/nvme/backups/openhands/
```

## לוח זמנים מומלץ

- **גיבוי יומי**: State ו-Config
- **גיבוי שבועי**: Workspace מלא
- **גיבוי חודשי**: הכל + העתקה ל-Cloud

## פתרון בעיות

### שגיאת מקום בדיסק
```bash
# בדוק מקום פנוי
df -h /mnt/nvme

# נקה גיבויים ישנים
find /mnt/nvme/backups/openhands -mtime +7 -delete
```

### שגיאת הרשאות
```bash
# בדוק הרשאות
ls -la /mnt/nvme/backups/openhands

# שנה הרשאות
chmod 755 /mnt/nvme/backups/openhands
```

