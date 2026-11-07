"""Arrange genuine idle Agent Server data; no model request or mocked response."""

import json
import pathlib
import subprocess
import sys

run = pathlib.Path(sys.argv[1]).resolve()
cli = pathlib.Path(__file__).resolve().parents[2] / ".agents/skills/verify-openhands/scripts/control-openhands"

def control(*args):
    result = subprocess.run([str(cli), *args, "--run", str(run)], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)

fixture = control("fixture", "git-repo", "--name", "diff-lifecycle")
repo = run / "workspace" / "diff-lifecycle"
(repo / "src/calc.py").write_text("def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n")
(repo / "README.md").write_text("# Diff lifecycle fixture\n\nUpdated README for the second file check.\n")
(repo / ".agents_tmp").mkdir(exist_ok=True)
(repo / ".agents_tmp/PLAN.md").write_text("# Diff lifecycle verification\n\n- Inspect the tracked Python change.\n- Return to the same diff after viewing this plan.\n")
payload = {
    "agent": {
        "kind": "Agent",
        "llm": {"model": "openai/gpt-4o", "base_url": "http://127.0.0.1:1/v1", "api_key": "qa-not-real", "num_retries": 0, "timeout": 2},
        "tools": [],
    },
    "workspace": {"working_dir": str(repo)},
    "autotitle": False,
}
request = run / "private" / "diff-lifecycle-request.json"
request.write_text(json.dumps(payload))
try:
    conversation = control("api", "POST", "/api/conversations", "--write", "--data", "@" + str(request), "--pick", "id")
finally:
    request.unlink()
receipt = {"conversationId": conversation["value"], "workspace": str(repo), "modelExecution": "None; idle conversation with no initial message", "backend": "Real local Agent Server"}
(run / "evidence/fixture.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
