"""Arrange synthetic persisted events for a stopped, isolated Control run.

The Agent Server reads these through its production history path. This fixture
contains no executed commands or model calls; it is render/load evidence only.
"""
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

from openhands.sdk.conversation.event_store import EventLog
from openhands.sdk.event import ActionEvent, MessageEvent, ObservationEvent, SystemPromptEvent
from openhands.sdk.io import LocalFileStore
from openhands.sdk.llm import Message, MessageToolCall, TextContent
from openhands.tools.terminal.definition import TerminalAction, TerminalObservation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--conversation', required=True)
parser.add_argument('--pairs', type=int, default=100)
args = parser.parse_args()
conversation_id = args.conversation.replace('-', '')
if len(conversation_id) != 32 or any(c not in '0123456789abcdef' for c in conversation_id):
    parser.error('conversation must be a hexadecimal UUID')
root = args.run.resolve() / 'private/state/dev_conversations' / conversation_id
if not (root / 'base_state.json').is_file():
    parser.error('create the inert conversation with Control api first')
if (root / 'events').exists():
    parser.error('refusing to overwrite an existing event history')
if not 1 <= args.pairs <= 10000:
    parser.error('--pairs must be between 1 and 10000')
events = EventLog(LocalFileStore(str(root)))
start = datetime.now() - timedelta(hours=1)
last = None

def append(event):
    global last
    event = event.model_copy(update={'timestamp': (start + timedelta(milliseconds=len(events)*10)).isoformat(), 'parent_id': last})
    events.append(event)
    last = event.id

def message(role, text, reasoning=None):
    append(MessageEvent(source='user' if role == 'user' else 'agent', llm_message=Message(role=role, content=[TextContent(text=text)], reasoning_content=reasoning)))

append(SystemPromptEvent(system_prompt=TextContent(text='Inert rendering fixture; no model or tool execution.'), tools=[]))
message('user', 'Seeded rendering fixture: inspect the recorded tool history. No commands were executed and no model was called.')
for i in range(args.pairs):
    command = f"printf 'fixture {i:04d}\\n'"
    call = MessageToolCall(origin='completion', id=f'fixture-call-{i}', name='terminal', arguments=json.dumps({'command':command}))
    action = ActionEvent(thought=[TextContent(text='Fixture thought: inspect the final recorded command.')] if i == args.pairs - 1 else [], reasoning_content='Fixture action reasoning: this belongs to the final command group.' if i == args.pairs - 1 else None, action=TerminalAction(command=command), tool_name='terminal', tool_call_id=call.id, tool_call=call, llm_response_id=f'fixture-response-{i}')
    append(action)
    append(ObservationEvent(action_id=action.id, tool_name='terminal', tool_call_id=call.id, observation=TerminalObservation(command=command, exit_code=0, content=[TextContent(text=f'fixture {i:04d}')])) )
    if (i+1) % 100 == 0:
        message('assistant', f'Fixture checkpoint {(i+1)//100}: {i+1} recorded commands.\n\n' + '\n\n'.join(f'Checkpoint detail {n}: the persisted history is available for scrolling and rendering measurements.' for n in range(12)))
        message('user', 'Continue inspecting the next recorded batch.')
message('assistant', 'Seeded history complete. The input below is available for typing measurements.\n\n' + '\n\n'.join(f'Fixture detail {n}: no external service was called.' for n in range(12)), reasoning='Fixture reply reasoning: this opens directly from its Thinking group.')
state = json.loads((root/'base_state.json').read_text())
state['head_is_empty'] = False
state['leaf_event_id'] = last
state['execution_status'] = 'idle'
(root/'base_state.json').write_text(json.dumps(state))
meta = json.loads((root/'meta.json').read_text())
meta['title'] = f'Seeded tool history ({args.pairs:,} pairs)'
(root/'meta.json').write_text(json.dumps(meta))
print(json.dumps({'conversation':args.conversation,'events':len(events),'toolPairs':args.pairs,'seeded':True,'modelCalls':0,'executedCommands':0}))
