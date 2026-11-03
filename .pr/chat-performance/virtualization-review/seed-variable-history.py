"""Arrange synthetic varied-height history in a stopped, isolated Control run.

Uses SDK serialization and real Agent Server persistence. No model or tools run.
"""
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

from openhands.sdk.conversation.event_store import EventLog
from openhands.sdk.event import MessageEvent
from openhands.sdk.io import LocalFileStore
from openhands.sdk.llm import Message, TextContent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--conversation", required=True)
args = parser.parse_args()
conversation_id = args.conversation.replace("-", "")
if len(conversation_id) != 32 or any(c not in "0123456789abcdef" for c in conversation_id):
    parser.error("conversation must be a hexadecimal UUID")
root = args.run.resolve() / "private/state/dev_conversations" / conversation_id
if not (root / "base_state.json").is_file() or (root / "events").exists():
    parser.error("requires an inert conversation without event history")
events = EventLog(LocalFileStore(str(root)))
start = datetime.now() - timedelta(hours=1)
last = None
for index in range(600):
    text = f"Pagination fixture message {index:04d}."
    text += "\n\nA paragraph of varied height for the persisted-history scroll anchoring check." * ((index * 7) % 9)
    event = MessageEvent(source="agent", llm_message=Message(role="assistant", content=[TextContent(text=text)]))
    event = event.model_copy(update={"timestamp": (start + timedelta(milliseconds=index * 10)).isoformat(), "parent_id": last})
    events.append(event)
    last = event.id
state = json.loads((root / "base_state.json").read_text())
state.update(head_is_empty=False, leaf_event_id=last, execution_status="idle")
(root / "base_state.json").write_text(json.dumps(state))
meta = json.loads((root / "meta.json").read_text())
meta["title"] = "Seeded variable-height history (600 messages)"
(root / "meta.json").write_text(json.dumps(meta))
print(json.dumps({"conversation": args.conversation, "events": len(events), "seeded": True, "modelCalls": 0, "executedCommands": 0}))
