"""Arrange canonical persisted SDK events in empty, stopped Control conversations.

No model, command, or tool runs. Canvas reads these through the real server.
Only the selected disposable conversations and their run-owned state are used.
"""
import argparse
import json
import socket
from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID

from openhands.sdk.conversation.state import ConversationState
from openhands.sdk.event import ActionEvent, MessageEvent, ObservationEvent
from openhands.sdk.io import LocalFileStore
from openhands.sdk.llm import Message, MessageToolCall, TextContent
from openhands.sdk.utils.cipher import Cipher
from openhands.sdk.workspace import LocalWorkspace
from openhands.tools.terminal.definition import TerminalAction, TerminalObservation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--projection', type=UUID, required=True)
parser.add_argument('--tied', type=UUID, required=True)
args = parser.parse_args()
run = args.run.resolve()
metadata = json.loads((run / 'run.json').read_text())
with socket.socket() as probe:
    if probe.connect_ex(('127.0.0.1', metadata['ports']['agentServer'])) == 0:
        parser.error('stop this run\'s Agent Server before seeding')
if args.projection == args.tied:
    parser.error('use two distinct disposable conversations')
roots = [(kind, cid, run / 'private/state/dev_conversations' / cid.hex)
         for kind, cid in [('PROJECTION', args.projection), ('TIED', args.tied)]]
for _, _, root in roots:
    if not (root / 'base_state.json').is_file() or (root / 'events').exists():
        parser.error('only empty, previously API-created conversations may be seeded')
cipher = Cipher((run / 'private/encryption-key').read_text().strip())
base = datetime(2026, 11, 13, 8, 0, 0)
receipts = []
for kind, cid, root in roots:
    state = ConversationState.create(id=cid, agent=None,
        workspace=LocalWorkspace(working_dir=str(run / 'workspace/projection-proof')),
        file_store=LocalFileStore(str(root)), cipher=cipher)
    records = []
    def append(event, millisecond=None):
        seq = len(records)
        event = event.model_copy(update={'timestamp': (base + timedelta(milliseconds=seq if millisecond is None else millisecond)).isoformat(timespec='microseconds')})
        if state.append_event(event) != seq:
            raise RuntimeError('unexpected persisted sequence')
        records.append({'seq': seq, 'id': event.id, 'kind': event.kind,
                        'timestamp': event.timestamp,
                        'actionId': getattr(event, 'action_id', None)})
    def message(role, text, millisecond=None):
        append(MessageEvent(source='user' if role == 'user' else 'agent',
            llm_message=Message(role=role, content=[TextContent(text=text)])), millisecond)
    if kind == 'PROJECTION':
        message('user', 'Seeded history: 49 recorded terminal calls. No commands or models ran.')
        for number in range(1, 50):
            command = f"printf 'recorded-{number:03d}\\n'"
            call = MessageToolCall(origin='completion', id=f'projection-call-{number}',
                name='terminal', arguments=json.dumps({'command': command}))
            action = ActionEvent(thought=[], action=TerminalAction(command=command),
                tool_name='terminal', tool_call_id=call.id, tool_call=call,
                llm_response_id=f'projection-response-{number}', summary=f'Recorded command {number:03d}')
            append(action)
            append(ObservationEvent(action_id=action.id, tool_name='terminal',
                tool_call_id=call.id, observation=TerminalObservation(command=command,
                    exit_code=0, content=[TextContent(text=f'recorded-{number:03d}')])) )
        message('assistant', 'History fixture complete. Scroll up to inspect all 49 recorded calls.\n\n' +
            '\n\n'.join(f'Fixture detail {n}: these persisted events exercise real history pagination without executing a model or command.' for n in range(24)))
        assert records[49]['kind'] == 'ActionEvent' and records[50]['kind'] == 'ObservationEvent'
        assert records[50]['actionId'] == records[49]['id']
    else:
        for number in range(1, 101):
            message('user', f'CANVAS-TIED-{number:03d}', 51 if number == 50 else number)
    meta_path = root / 'meta.json'
    meta = json.loads(meta_path.read_text())
    meta['title'] = 'Tool history boundary (49 calls)' if kind == 'PROJECTION' else 'Tied timestamp control (100 messages)'
    meta_path.write_text(json.dumps(meta))
    receipts.append({'kind': kind, 'conversation': str(cid), 'events': len(records),
        'boundary': records[48:52], 'modelCalls': 0, 'commandsExecuted': 0})
print(json.dumps({'seeded': True, 'cases': receipts}))
