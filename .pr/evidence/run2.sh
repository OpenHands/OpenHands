#!/bin/bash
cd "C:/Users/Kris/Documents/ChatGPT/提PR/openhands-pr-17434"
node scripts/dev-with-automation.mjs --port 18101 > .pr/evidence/launcher-run2.log 2>&1 &
LPID=$!
echo "launcher pid: $LPID" > .pr/evidence/run2-exitcode.txt
for i in $(seq 1 150); do
  if ! kill -0 $LPID 2>/dev/null; then break; fi
  # wait for all 4 services listening
  n=$(netstat -ano 2>/dev/null | grep -cE ":(3001|18000|18001|18101)\s.*LISTENING")
  if [ "$n" -ge 4 ]; then
    # all up: kill the required vite child (listener on :3001)
    VITE_PID=$(netstat -ano | grep ":3001\s.*LISTENING" | awk '{print $5}' | head -1)
    echo "all services up; killing required vite pid $VITE_PID at iteration $i" >> .pr/evidence/run2-exitcode.txt
    taskkill //PID $VITE_PID //T //F >> .pr/evidence/run2-exitcode.txt 2>&1
  fi
  sleep 2
done
wait $LPID
RC=$?
echo "launcher exit code: $RC" >> .pr/evidence/run2-exitcode.txt
echo "FINAL exit code: $RC"
