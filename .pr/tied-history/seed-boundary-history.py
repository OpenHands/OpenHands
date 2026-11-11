"""Seed an empty, stopped, isolated Control run with SDK-persisted history.

This is deterministic real-backend render data, not model-generated output.
No agent, model, tool, or external service is run. Only the selected run's two
empty conversation states are changed. Its generated encryption key is used
internally to preserve its own state and is never printed or copied.
"""
import argparse
import json
import socket
from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID

from openhands.sdk.conversation.state import ConversationState
from openhands.sdk.event import MessageEvent
from openhands.sdk.io import LocalFileStore
from openhands.sdk.llm import Message, TextContent
from openhands.sdk.utils.cipher import Cipher
from openhands.sdk.workspace import LocalWorkspace

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--tied", type=UUID, required=True)
parser.add_argument("--control", type=UUID, required=True)
args = parser.parse_args()
run = args.run.resolve()
metadata = json.loads((run / "run.json").read_text())
port = metadata["ports"]["agentServer"]
with socket.socket() as probe:
    if probe.connect_ex(("127.0.0.1", port)) == 0:
        parser.error("stop this run's Agent Server before seeding")
if args.tied == args.control:
    parser.error("tied and control must be distinct disposable conversations")

roots = [(name, cid, run / "private/state/dev_conversations" / cid.hex)
         for name, cid in [("TIED", args.tied), ("CONTROL", args.control)]]
for _, _, root in roots:
    if not (root / "base_state.json").is_file():
        parser.error("create each inert conversation with Control api first")
    if (root / "events").exists():
        parser.error("refusing to overwrite existing event history")

cipher = Cipher((run / "private/encryption-key").read_text().strip())
base = datetime(2026, 11, 11, 8, 0, 0)
receipts = []
for name, cid, root in roots:
    state = ConversationState.create(
        id=cid, agent=None,
        workspace=LocalWorkspace(working_dir=str(run / "workspace/tied-boundary")),
        file_store=LocalFileStore(str(root)), cipher=cipher,
    )
    for number in range(1, 101):
        milliseconds = 51 if name == "TIED" and number == 50 else number
        event = MessageEvent(
            source="user",
            llm_message=Message(role="user", content=[
                TextContent(text=f"CANVAS-{name}-{number:03d}")
            ]),
        ).model_copy(update={
            "timestamp": (base + timedelta(milliseconds=milliseconds)).isoformat(
                timespec="microseconds"),
        })
        sequence = state.append_event(event)
        if sequence != number - 1:
            raise RuntimeError("unexpected persisted sequence")
    meta_file = root / "meta.json"
    meta = json.loads(meta_file.read_text())
    meta["title"] = f"{name.title()} timestamp fixture (100 messages)"
    meta_file.write_text(json.dumps(meta))
    receipts.append({
        "kind": name, "conversation": str(cid), "events": len(state.events),
        "markers": 100, "boundaryTimestampsEqual": name == "TIED",
        "modelCalls": 0, "executedCommands": 0,
    })
print(json.dumps({"seeded": True, "cases": receipts}))

