# SGLang Docker Setup - פתרון מלא

## ✅ מה הוגדר

1. **SGLang Container** - רץ ב-Docker עם מודל `qwen2.5-coder:14b`
2. **NGINX Reverse Proxy** - חושף את SGLang על פורט 11434 (במקום ollama)
3. **Volume Mapping** - המודלים נשמרים ב-`/mnt/nvme/ollama_models`

## 📋 תצורה

### Docker Compose
- **Service**: `sglang`
- **Image**: `lmsysorg/sglang:latest`
- **Port**: `30000` (פנימי) -> `30000` (חיצוני)
- **Model**: `Qwen/Qwen2.5-Coder-14B-Instruct`
- **Volume**: `/mnt/nvme/ollama_models:/root/.cache/huggingface`

### NGINX
- **Config**: `/etc/nginx/sites-available/sglang_proxy`
- **Port**: `11434` (חיצוני) -> `127.0.0.1:30000` (SGLang)

### OpenHands Configuration
- **LLM_BASE_URL**: `http://sglang:30000/v1` (מ-Docker network)
- **LLM_MODEL**: `Qwen/Qwen2.5-Coder-14B-Instruct`

## 🚀 הפעלה

```bash
cd /home/noya/OpenHands
docker compose up -d sglang
```

## 🔍 בדיקה

### בדיקת Container
```bash
docker ps | grep sglang
docker logs -f openhands-sglang
```

### בדיקת API
```bash
# ישירות מ-Container
curl http://localhost:30000/health
curl http://localhost:30000/v1/models

# דרך NGINX
curl http://localhost:11434/v1/models
```

### בדיקת NGINX
```bash
sudo nginx -t
sudo systemctl status nginx
sudo tail -f /var/log/nginx/error.log
```

## ⏱️ זמן טעינה

המודל `qwen2.5-coder:14b` לוקח כ-5-10 דקות להיטען בפעם הראשונה (הורדה + טעינה ל-GPU).

## 🔧 פתרון בעיות

### אם NGINX מחזיר 502
- SGLang עדיין לא מוכן - חכה כמה דקות
- בדוק: `docker logs openhands-sglang`

### אם יש OutOfMemory
- המודל 14B כבד - ייתכן שיהיה צורך במודל קטן יותר
- נסה: `Qwen/Qwen2.5-Coder-7B-Instruct`

### אם Container לא רץ
```bash
docker compose logs sglang
docker compose restart sglang
```

## 📊 ניטור

```bash
# לוגים בזמן אמת
docker logs -f openhands-sglang

# שימוש ב-GPU
nvidia-smi

# גודל המודל
du -sh /mnt/nvme/ollama_models/hub/models--Qwen--Qwen2.5-Coder-14B-Instruct
```


