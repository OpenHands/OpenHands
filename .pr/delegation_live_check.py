"""Live check: a profile saved by Canvas delegates to general-purpose.

Runs a real agent-server and Canvas's scripted mock LLM. The profile's `tools`
come from Canvas's buildProfileToolsValue (--tools-json), saved as a profile and
launched. Exits non-zero if the outcome differs from --expect.

    python .pr/delegation_live_check.py --mock tests/e2e/mock-llm/scripts/mock-llm-server.py \
        --server-python <env with openhands-agent-server>/bin/python \
        --tools-json '[{"name":"terminal"},...]' --expect pass|refuse
"""

import argparse
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx


MOCK_PORT = 19787
SERVER_PORT = 19788
SERVER = f"http://127.0.0.1:{SERVER_PORT}"
MOCK = f"http://127.0.0.1:{MOCK_PORT}"
LLM = {"model": "openai/mock-model", "base_url": f"{MOCK}/v1", "api_key": "sk-mock"}

failures: list[str] = []


def check(label: str, ok: bool, detail: object = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + ("" if ok else f"  [{detail}]"))
    if not ok:
        failures.append(label)


def wait_for(url: str, timeout: float = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2).status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise SystemExit(f"timed out waiting for {url}")


def tool_names(body: dict) -> list[str]:
    return sorted(spec["function"]["name"] for spec in body.get("tools", []))


def tool_results(body: dict) -> list[str]:
    results = []
    for message in body["messages"]:
        if message["role"] != "tool":
            continue
        content = message["content"]
        if not isinstance(content, str):
            content = " ".join(block.get("text", "") for block in content)
        results.append(content)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", required=True, type=Path)
    parser.add_argument("--server-python", default=sys.executable)
    parser.add_argument("--tools-json", required=True)
    parser.add_argument("--expect", choices=["pass", "refuse"], required=True)
    args = parser.parse_args()

    root = Path(tempfile.mkdtemp(prefix="conv-local-e2e-"))
    workdir = root / "workspace"
    workdir.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.endswith("SESSION_API_KEY")}
    env.update(
        HOME=str(root / "home"),
        OH_PERSISTENCE_DIR=str(root / "persist"),
        OH_SECRET_KEY=secrets.token_urlsafe(32),
        OPENHANDS_SUPPRESS_BANNER="1",
        TMUX_TMPDIR=tempfile.mkdtemp(prefix="oht", dir="/tmp"),
    )
    mock = subprocess.Popen(
        [sys.executable, str(args.mock), "--port", str(MOCK_PORT)],
        env=env,
        stdout=(root / "mock.log").open("w"),
        stderr=subprocess.STDOUT,
    )
    server = subprocess.Popen(
        [args.server_python, "-m", "openhands.agent_server"]
        + ["--host", "127.0.0.1", "--port", str(SERVER_PORT)],
        env=env,
        cwd=root,
        stdout=(root / "server.log").open("w"),
        stderr=subprocess.STDOUT,
    )
    try:
        wait_for(f"{MOCK}/admin/requests")
        wait_for(f"{SERVER}/server_info")
        client = httpx.Client(base_url=SERVER, timeout=60)
        print("agent-server", client.get("/server_info").json().get("version"))
        client.post(
            "/api/profiles/mock", json={"llm": {**LLM, "usage_id": "agent"}}
        ).raise_for_status()
        client.post(
            "/api/agent-profiles/coder",
            json={
                "agent_kind": "openhands",
                "llm_profile_ref": "mock",
                "tools": json.loads(args.tools_json),
            },
        ).raise_for_status()
        profile_id = next(
            p["id"]
            for p in client.get("/api/agent-profiles").json()["profiles"]
            if p["name"] == "coder"
        )

        turns = [
            {
                "tool_call": {
                    "name": "task",
                    "arguments": {
                        "prompt": "Plan the work.",
                        "subagent_type": "general-purpose",
                    },
                }
            },
            {
                "tool_call": {
                    "name": "task_tracker",
                    "arguments": {
                        "command": "plan",
                        "task_list": [{"title": "Read the code", "status": "todo"}],
                    },
                }
            },
            {"text": "SUB-AGENT PLANNED"},
            {"text": "All done."},
        ]
        httpx.post(f"{MOCK}/admin/reset", json={}).raise_for_status()
        httpx.post(
            f"{MOCK}/admin/trajectory/register", json={"name": "run", "turns": turns}
        ).raise_for_status()
        httpx.post(
            f"{MOCK}/admin/trajectory/activate", json={"name": "run"}
        ).raise_for_status()
        response = client.post(
            "/api/conversations",
            json={
                "agent_profile_id": profile_id,
                "autotitle": False,
                "workspace": {"working_dir": str(workdir)},
                "initial_message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Go."}],
                    "run": True,
                },
            },
        )
        response.raise_for_status()
        conversation_id = response.json()["id"]
        deadline = time.time() + 180
        status = ""
        while time.time() < deadline:
            status = client.get(f"/api/conversations/{conversation_id}").json()[
                "execution_status"
            ]
            if status in ("finished", "error", "stuck", "idle"):
                break
            time.sleep(0.5)
        requests = httpx.get(f"{MOCK}/admin/requests").json()["requests"]

        parent = [r for r in requests if "task" in tool_names(r)]
        sub_agent = [r for r in requests if "task" not in tool_names(r)]
        offered = parent[0]["tools"] if parent else []
        task_description = next(
            (
                s["function"]["description"]
                for s in offered
                if s["function"]["name"] == "task"
            ),
            "",
        )
        print("parent tools:", tool_names(parent[0]) if parent else None)
        print("sub-agent tools:", tool_names(sub_agent[0]) if sub_agent else None)
        print("conversation status:", status)

        delegated = tool_results(parent[-1]) if parent else []
        print("profile tools:", [t["name"] for t in json.loads(args.tools_json)])
        print("parent tools:", tool_names(parent[0]) if parent else None)
        ran = bool(sub_agent) and "task_tracker" in tool_names(sub_agent[0])
        refused = any("which this agent does not have" in r for r in delegated)
        if refused:
            print("delegation result:", delegated)
        if args.expect == "pass":
            check("general-purpose is offered", "**general-purpose**" in task_description)
            check("general-purpose ran with task_tracker", ran)
            check(
                "delegation returned the sub-agent's answer",
                any("SUB-AGENT PLANNED" in r for r in delegated),
                delegated,
            )
            check("no scope refusal", not refused, delegated)
        else:
            check("general-purpose is not offered", "**general-purpose**" not in task_description)
            check("delegation refused by scope", refused, delegated)
    finally:
        server.terminate()
        server.wait(timeout=30)
        mock.terminate()
        mock.wait(timeout=20)

    print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
