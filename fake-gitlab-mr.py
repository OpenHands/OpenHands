import os, json, time, hmac, hashlib, base64, urllib.request
secret = os.environ["SECRET"]
url = "http://localhost:8000/api/automation/v1/events/46d09c5c-7d7d-585f-8c37-a6b45a3de738/technogise-gitlab"
body = json.dumps({
  "object_kind": "merge_request",
  "user": {"username": "saroj"},
  "project": {"path_with_namespace": "feedback-app/feedback-app-backend"},
  "object_attributes": {"action": "open", "iid": 1, "title": "POC: test",
    "description": "", "source_branch": "test/agent-canvas-review-poc", "target_branch": "develop"},
}).encode()
msg_id, ts = "msg_%d" % time.time(), str(int(time.time()))
key = base64.b64decode(secret.removeprefix("whsec_"))
sig = base64.b64encode(hmac.new(key, f"{msg_id}.{ts}.".encode() + body, hashlib.sha256).digest()).decode()
req = urllib.request.Request(url, body, {"Content-Type": "application/json",
  "webhook-id": msg_id, "webhook-timestamp": ts, "webhook-signature": "v1," + sig})
print(urllib.request.urlopen(req).read().decode())
