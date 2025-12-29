# SGLang Setup - פתרון מקצועי עם NGINX

## סקירה כללית

SGLang מופעל ישירות על המערכת (לא ב-Docker) בגלל בעיית Docker overlayfs, עם NGINX reverse proxy לחשיפה מקצועית.

## תצורה

### מודל
- **מודל**: Qwen/Qwen2.5-Coder-14B-Instruct
- **פורט**: 11434 (במקום ollama)
- **מיקום**: רץ ישירות על המערכת

### NGINX Reverse Proxy
- **פורט חיצוני**: 11434
- **פורט פנימי**: 127.0.0.1:11434
- **קובץ קונפיגורציה**: `/etc/nginx/sites-available/sglang_proxy`

### הפעלה
```bash
/home/noya/OpenHands/scripts/start-sglang.sh
```

### בדיקה
```bash
# בדיקת תהליך
ps aux | grep sglang

# בדיקת API
curl http://localhost:11434/v1/models

# בדיקת לוגים
tail -f /home/noya/sglang.log
```

## OpenHands Configuration

- **LLM_BASE_URL**: `http://host.docker.internal:11434/v1`
- **LLM_MODEL**: `Qwen/Qwen2.5-Coder-14B-Instruct`

## פתרון בעיות

### אם SGLang לא מאזין על 0.0.0.0
- בדוק את הלוגים: `tail -f /home/noya/sglang.log`
- ודא שהפרמטר `--host 0.0.0.0` מופיע בסקריפט

### אם NGINX מחזיר 502
- בדוק ש-SGLang רץ: `ps aux | grep sglang`
- בדוק את לוגי NGINX: `sudo tail -f /var/log/nginx/error.log`

### אם יש OutOfMemory
- המודל 14B כבד - ייתכן שיהיה צורך במודל קטן יותר או quantization


