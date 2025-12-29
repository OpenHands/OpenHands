# הגדרת API Key - OpenHands

**תאריך:** 2025-12-29

## ✅ תיקונים שבוצעו

1. **הוספת תמיכה ב-.env file:**
   - הוסף `env_file: - .env` ל-docker-compose.yml
   - ה-API key כעת נטען מ-.env file

2. **שינוי docker-compose.yml:**
   - שונה `LLM_API_KEY=your_aluma_api_key_here` ל-`LLM_API_KEY=${ALUMA_API_KEY:-your_aluma_api_key_here}`
   - זה מאפשר לטעון את ה-API key מ-.env file

3. **יצירת .env.example:**
   - נוצר קובץ `.env.example` עם דוגמה

## 🔧 איך להגדיר את ה-API Key

### שיטה 1: באמצעות .env file (מומלץ)

1. **ערוך את קובץ .env:**
   ```bash
   nano /home/noya/OpenHands/.env
   ```

2. **הוסף את ה-API key שלך:**
   ```bash
   ALUMA_API_KEY=your_real_api_key_here
   ```

3. **שמור וסגור** (Ctrl+X, Y, Enter)

4. **הפעל מחדש את הקונטיינר:**
   ```bash
   cd /home/noya/OpenHands
   docker compose restart openhands
   ```

### שיטה 2: ישירות ב-docker-compose.yml

1. **ערוך את docker-compose.yml:**
   ```bash
   nano /home/noya/OpenHands/docker-compose.yml
   ```

2. **מצא את השורה:**
   ```yaml
   - LLM_API_KEY=${ALUMA_API_KEY:-your_aluma_api_key_here}
   ```

3. **החלף ב:**
   ```yaml
   - LLM_API_KEY=your_real_api_key_here
   ```

4. **שמור וסגור**

5. **הפעל מחדש:**
   ```bash
   docker compose restart openhands
   ```

## 🔍 בדיקה

לבדוק שה-API key נטען נכון:
```bash
docker exec openhands-app env | grep LLM_API_KEY
```

אם אתה רואה את ה-API key האמיתי שלך (לא `your_aluma_api_key_here`), זה עובד!

## ⚠️ הערות חשובות

1. **אל תעלה את .env ל-Git:**
   - קובץ .env כבר ב-.gitignore
   - זה שומר על ה-API key שלך בטוח

2. **אם אין לך API key:**
   - הירשם ב-Aluma: https://aluma.ai
   - קבל API key מהדשבורד
   - הוסף אותו ל-.env

3. **אם האגנט עדיין לא עובד:**
   - בדוק את הלוגים: `docker logs openhands-app -f`
   - ודא שה-API key תקין
   - ודא שיש חיבור לאינטרנט

## 📋 פקודות שימושיות

```bash
# בדיקת API key
docker exec openhands-app env | grep LLM_API_KEY

# לוגים
docker logs openhands-app -f

# הפעלה מחדש
docker compose restart openhands

# בדיקת שירותים
docker compose ps
```




