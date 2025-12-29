# הוראות מהירות להגדרת API Key

## 🚀 איך להגדיר את ה-API Key

### שלב 1: ערוך את קובץ .env
```bash
nano /home/noya/OpenHands/.env
```

### שלב 2: מצא את השורה
```
ALUMA_API_KEY=your_aluma_api_key_here
```

### שלב 3: החלף ב-API key האמיתי שלך
```
ALUMA_API_KEY=your_real_api_key_here
```

### שלב 4: שמור וסגור
- לחץ `Ctrl+X`
- לחץ `Y` (לשמור)
- לחץ `Enter`

### שלב 5: הפעל מחדש את הקונטיינר
```bash
cd /home/noya/OpenHands
docker compose restart openhands
```

### שלב 6: בדוק שהכל עובד
```bash
docker exec openhands-app env | grep LLM_API_KEY
```

אם אתה רואה את ה-API key האמיתי שלך (לא `your_aluma_api_key_here`), זה עובד! ✅

## 📝 הערות

- אם אין לך API key, הירשם ב-Aluma: https://aluma.ai
- ה-API key נשמר בקובץ .env שלא נשמר ב-Git (בטוח)
- אחרי השינוי, הפעל מחדש את הקונטיינר




