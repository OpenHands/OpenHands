#!/usr/bin/env python3
"""Bounded loopback TCP endpoint that accepts connections and never answers.

No request bytes, headers, or credentials are read or recorded. Use a dummy
model/key in the caller. The private port.json records only port/PID/deadline.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import time


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seconds", type=int, default=180)
    a = p.parse_args()
    if not 1 <= a.seconds <= 180:
        p.error("seconds must be within1..180")
    return a


if __name__ == "__main__":
    a = arguments()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(a.output_dir, 0o700)
    held = []
    began = time.time()
    server = socket.create_server(("127.0.0.1", 0))
    server.settimeout(0.5)
    port = server.getsockname()[1]
    metadata = {"pid": os.getpid(), "port": port, "started_at": began,
                "deadline": began + a.seconds, "received_bytes_recorded": 0}
    target = a.output_dir / "port.json"
    target.write_text(json.dumps(metadata, indent=2))
    os.chmod(target, 0o600)
    def stop(_sig, _frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while time.time() < metadata["deadline"] and len(held) < 32:
            try:
                connection, _address = server.accept()
                held.append(connection)
                metadata["accepted_connections"] = len(held)
                target.write_text(json.dumps(metadata, indent=2))
            except TimeoutError:
                pass
    finally:
        for connection in held:
            connection.close()
        server.close()
        metadata["closed_at"] = time.time()
        metadata["accepted_connections"] = len(held)
        target.write_text(json.dumps(metadata, indent=2))
        print(json.dumps({"port": port, "accepted_connections": len(held), "closed": True}))
