#!/bin/bash
# usage: run17899.sh <label> <checkout>
set -u
LABEL=$1; CO=$2
export PATH=/opt/node24/bin:$PATH
export CONTROL_OPENHANDS_BROWSER=/opt/pw-browsers/chromium
export CONTROL_OPENHANDS_TRUST_CA=/root/.ccr/ca-bundle.crt
export OH_VERIFY_HOME=/tmp/claude-0/verify-home
PY=/root/.cache/uv/archive-v0/20IFE14lts3pJ4Fj/bin/python
MP=19999
pkill -f "mock-llm-server.py --port $MP" 2>/dev/null
$PY -I $CO/tests/e2e/mock-llm/scripts/mock-llm-server.py --port $MP > /tmp/claude-0/mock-$LABEL.log 2>&1 &
until curl -s --noproxy '*' http://127.0.0.1:$MP/ >/dev/null; do sleep 1; done
CMD="env | grep '^GIT_' | sort; git init -q qa_t && cd qa_t && git commit --allow-empty -qm x; git log -1 --format='AUTHOR=%an <%ae> COMMITTER=%cn <%ce>'"
python3 - "$CMD" <<'PY'
import json,sys,urllib.request
urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
cmd=sys.argv[1]
body={"name":"git","turns":[{"tool_call":{"name":"terminal","arguments":{"command":cmd}}},{"tool_call":{"name":"terminal","arguments":{"command":cmd}}},{"text":"GIT_IDENTITY_CHECK_DONE"}]}
for path,data in [("/admin/trajectory/register",body),("/admin/trajectory/activate",{"name":"git"})]:
    r=urllib.request.Request("http://127.0.0.1:19999"+path,data=json.dumps(data).encode(),headers={"Content-Type":"application/json"})
    print(urllib.request.urlopen(r).read().decode())
PY
cd $CO
C="node .agents/skills/verify-openhands/scripts/control-openhands.mjs"
RUN=$($C launch --new --build always --print-run --timeout 900) || { echo "launch failed"; exit 1; }
export OH_VERIFY_RUN=$RUN; echo "RUN=$RUN"
$C onboard --skip >/dev/null
MOCK_KEY=mock-api-key-for-testing $C llm set --profile mock --model openai/mock-test-model --api-key-env MOCK_KEY --base-url http://127.0.0.1:19999/v1 | head -c 400; echo
F=bug-17899-$LABEL
$C browser goto /settings/app >/dev/null
$C browser fill 'testid=git-user-name-input' 'QA Tester'
$C browser fill 'testid=git-user-email-input' 'qa-tester@example.com'
$C browser click 'testid=submit-button'
sleep 2
$C browser screenshot --feature $F --name 1-settings-saved
$C api GET /api/settings | grep -o '"git_user_[a-z]*": *"[^"]*"'
OUT=$($C conversation start --prompt "Check the git identity" --wait --timeout 180)
echo "$OUT" | head -c 600; echo
CID=$(echo "$OUT" | grep -o '"conversation_id": *"[^"]*"\|"id": *"[^"]*"' | head -1 | grep -o '[0-9a-f-]\{32,\}')
echo CID=$CID
$C conversation events $CID --kinds ObservationEvent --full | grep -oE "GIT_[A-Z_]+=[^\\\\]*|AUTHOR=[^\\\\]*|Author identity unknown|fatal:[^\\\\]*" | sort -u
$C browser goto /conversations/$CID >/dev/null; sleep 4
$C browser click 'button[aria-label="Expand"] >> nth=0' >/dev/null; sleep 1
$C browser screenshot --feature $F --name 2-conversation
$C stop >/dev/null
pkill -f "mock-llm-server.py --port $MP"
echo DONE
