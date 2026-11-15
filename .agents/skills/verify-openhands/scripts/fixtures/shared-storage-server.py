#!/usr/bin/env python3
"""Bounded no-editor Agent Server fixture sharing one verification run's storage.

Use the installed SDK's Python. Keys are read only from the run's private files;
they remain in the child's environment and are never printed. Stop this parent
with SIGTERM for deterministic child-process-group cleanup.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from uuid import UUID


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def stop_owned_group(child, grace_seconds=5, force_seconds=2):
    pgid = child.pid  # Popen(start_new_session=True) established this owned group.
    forced = False
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace_seconds
    while group_alive(pgid) and time.monotonic() < deadline:
        child.poll()  # Reap the leader without mistaking it for the whole group.
        time.sleep(0.05)
    if group_alive(pgid):
        forced = True
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + force_seconds
    while group_alive(pgid) and time.monotonic() < deadline:
        child.poll()
        time.sleep(0.05)
    child.poll()
    return {"leader_stopped": child.returncode is not None,
            "group_stopped": not group_alive(pgid), "forced": forced}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--seconds", type=int, default=240)
    p.add_argument("--load-id", type=UUID, help="Read native events/count on the existing owned secondary")
    a = p.parse_args()
    if not 1 <= a.seconds <= 300:
        p.error("seconds must be within1..300")
    run = a.run.resolve()
    json.loads((run / "run.json").read_text())
    directory = run / "private/shared-server-5629"
    directory.mkdir(exist_ok=True)
    os.chmod(directory, 0o700)
    if a.load_id:
        import httpx
        metadata = json.loads((directory / "server.json").read_text())
        if metadata.get("stopped_at") or time.time() >= metadata["deadline"]:
            p.error("the bounded owned secondary is stopped or expired")
        os.kill(metadata["pid"], 0)
        port = int(metadata["port"])
        if not 1024 <= port <= 65535:
            p.error("invalid owned secondary port")
        with httpx.Client(trust_env=False, follow_redirects=False, timeout=5) as client:
            response = client.get(
                f"http://127.0.0.1:{port}/api/conversations/{a.load_id}/events/count",
                headers={"X-Session-API-Key": (run / "private/session-key").read_text().strip()},
            )
        print(json.dumps({"status": response.status_code,
                          "event_count": response.json() if response.status_code == 200 else None}))
        raise SystemExit(0)
    env = json.loads((run / "private/launcher-env.json").read_text())
    env.update({
        "OH_SECRET_KEY": (run / "private/encryption-key").read_text().strip(),
        "OH_SESSION_API_KEYS_0": (run / "private/session-key").read_text().strip(),
        "OH_CONVERSATIONS_PATH": str(run / "private/state/dev_conversations"),
        "OH_WORKSPACE_PATH": str(run / "workspace/qa-rechecks"),
        "OH_BASH_EVENTS_DIR": str(directory / "bash-events"),
        "OH_ENABLE_VSCODE": "false",
        "OH_PERSISTENCE_DIR": str(directory / "state"),
    })
    with socket.create_server(("127.0.0.1", 0)) as probe:
        port = probe.getsockname()[1]
    stopped = False
    def stop(_sig, _frame):
        global stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (directory / "server.log").open("wb") as log:
        metadata = {}
        target = directory / "server.json"
        child = subprocess.Popen(
            [sys.executable, "-m", "openhands.agent_server", "--host", "127.0.0.1",
             "--port", str(port)], env=env, cwd=directory,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            metadata = {"pid": child.pid, "pgid": child.pid, "operator_pid": os.getpid(),
                        "port": port, "deadline": time.time() + a.seconds,
                        "shared_storage": str(run / "private/state/dev_conversations")}
            target.write_text(json.dumps(metadata, indent=2))
            os.chmod(target, 0o600)
            while not stopped and child.poll() is None and time.time() < metadata["deadline"]:
                time.sleep(0.2)
        finally:
            cleanup = stop_owned_group(child)
            metadata.update({"stopped_at": time.time(), "exit_code": child.returncode,
                             "cleanup": cleanup})
            target.write_text(json.dumps(metadata, indent=2))
            print(json.dumps({"port": port, **cleanup,
                              "exit_code": child.returncode}))
