#!/usr/bin/env bash
# The checks behind before.txt and after.txt. Needs Canvas started with
# MARS_WEB=1 and LOCAL_BACKEND_API_KEY=<key> on 127.0.0.1:<port>, signed in to
# DigitalOcean (or the stand-in from #18714 via MARS_API_BASE_URL + MARS_TOKEN),
# with <session> connected:
#   curl -X POST -H "X-Session-API-Key: <key>" -H 'Content-Type: application/json' \
#     -d '{"args":[{"sessionId":"<session>"}]}' http://127.0.0.1:<port>/mars/rpc/openTunnel
#   bash .pr/18707-agent-login/repro.sh <port> <key> <session>
set -u
PORT=$1
KEY=$2
SESSION=$3
D=$(dirname "$0")
echo "### event socket logins"
node "$D/agent-check.mjs" "http://127.0.0.1:$PORT" "$KEY" "$SESSION"
echo "### (b) no key at all, one 32 MiB frame before any login"
PID=$(lsof -ti "tcp:$PORT" -sTCP:LISTEN | head -1)
RSS0=$(ps -o rss= -p "$PID")
node "$D/ws-raw.mjs" "$PORT" "/mars/sessions/$SESSION/sockets/events/00000000000000000000000000000000" big:32
echo "ingress RSS: $RSS0 KB -> $(ps -o rss= -p "$PID") KB"
echo "### (a) '#' in the socket path, then a valid login frame"
node "$D/ws-raw.mjs" "$PORT" "/mars/sessions/$SESSION/sockets/events/c1#x" auth "$KEY"
sleep 2
echo "Canvas still up? /mars/health -> $(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/mars/health")"
