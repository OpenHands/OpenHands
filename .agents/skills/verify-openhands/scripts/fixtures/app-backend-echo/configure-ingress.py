#!/usr/bin/env python3
"""Arrange one owned verification run's standard SDK loopback ingress field."""
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", required=True, type=Path)
parser.add_argument("--restore", action="store_true")
args = parser.parse_args()
run = args.run.resolve()
info = json.loads((run / "run.json").read_text())
envfile = run / "private/launcher-env.json"
backup = run / "private/qa-verify-app-backend-ingress.json"
key = "OH_APP_BACKEND_PUBLIC_URL"
env = json.loads(envfile.read_text())

def save_private(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as output:
        json.dump(value, output)
    os.replace(temporary, path)

if args.restore:
    old = json.loads(backup.read_text())
    if old["present"]:
        env[key] = old["value"]
    else:
        env.pop(key, None)
    save_private(envfile, env)
    backup.unlink()
else:
    if backup.exists():
        raise SystemExit("Restore the existing ingress backup before arranging again")
    port = info["ports"]["agentServer"]
    if type(port) is not int or not 1 <= port <= 65535:
        raise SystemExit("Run has no valid owned Agent Server port")
    save_private(backup, {"present": key in env, "value": env.get(key)})
    env[key] = f"http://127.0.0.1:{port}"
    save_private(envfile, env)

print("Owned ingress field restored; restart required" if args.restore else
      "Owned loopback ingress field arranged; restart required")
