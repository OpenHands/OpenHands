#!/bin/bash
set -e
export RUNTIME=local
export INSTALL_DOCKER=0

rm -f /run/.containerenv

# Ensure nginx proxy config
ln -sf /etc/nginx/sites-available/openhands /etc/nginx/sites-enabled/openhands
nginx -s stop 2>/dev/null || true
sleep 1
nginx

# Start backend in background
cd /home/workspace/OpenHands
poetry run uvicorn openhands.server.listen:app --host 127.0.0.1 --port 3000 &
BACKEND_PID=$!

# Start frontend using npx for cross-env
cd /home/workspace/OpenHands/frontend
npm run make-i18n
npx cross-env VITE_MOCK_API=false react-router dev --port 3001 --host 127.0.0.1 &
FRONTEND_PID=$!

wait -n $BACKEND_PID $FRONTEND_PID
