"""Prepare persisted SDK events while this run's Agent Server is stopped.

This arranges history; the browser proof must subsequently read the real server.
No model responses, network responses or browser routes are intercepted.
"""
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

from openhands.sdk.conversation.state import ConversationState
from openhands.sdk.workspace import LocalWorkspace
from openhands.sdk.event import MessageEvent, ActionEvent, ObservationEvent
from openhands.sdk.llm import Message, TextContent, MessageToolCall
from openhands.tools.terminal import TerminalAction, TerminalObservation

fixture_path=Path(sys.argv[1])
fixture=json.loads(fixture_path.read_text(encoding='utf-8-sig'))
run=Path('/mnt/d/GitHub_project/oss-contributions/validation/openhands-r2-runs')/fixture['runName']
cid=UUID(fixture['conversationId'])
state=ConversationState.create(id=cid,agent=None,workspace=LocalWorkspace(working_dir=str(run/'workspace')),persistence_dir=str(run/'private/state/dev_conversations'/cid.hex),max_iterations=1)
assert any(type(event).__name__=='SystemPromptEvent' for event in state.events), 'Initialize through the real send-message API first.'
assert len(state.events)<10, 'Do not append this fixture twice.'
start=datetime.fromisoformat(MessageEvent.model_fields['timestamp'].default_factory())
def stamp(n): return (start+timedelta(milliseconds=n)).isoformat()
def message(n,text,role='user'):
    return MessageEvent(timestamp=stamp(n),source='user' if role=='user' else 'agent',llm_message=Message(role=role,content=[TextContent(text=text)]))
decision=message(0,'old-decision-needle: keep workspace state scoped to the active backend.')
reply=message(1,'Earlier fixture decision: retain the selected conversation context.','assistant')
command="printf 'older-%s-needle: workspace connection failed\\n' tool"
result=subprocess.run(['sh','-c',command],capture_output=True,text=True,check=True)
call_id=str(uuid4())
action=ActionEvent(timestamp=stamp(2),thought=[],action=TerminalAction(command=command),tool_name='terminal',tool_call_id=call_id,tool_call=MessageToolCall(id=call_id,name='terminal',arguments=json.dumps({'command':command}),origin='completion'),llm_response_id=str(uuid4()))
observation=ObservationEvent(timestamp=stamp(3),tool_name='terminal',tool_call_id=call_id,action_id=action.id,observation=TerminalObservation(command=command,exit_code=result.returncode,content=[TextContent(text='fixture output line\n'*80+result.stdout)]))
with state:
    for event in [decision,reply,action,observation]: state.append_event(event)
    for n in range(4,608): state.append_event(message(n,f'History fixture turn {n:04d}: ordinary recent text.','assistant' if n%2 else 'user'))
fixture.update(decisionId=decision.id,toolObservationId=observation.id,seededEvents=len(state.events),seedMethod='SDK1.60.1 ConversationState.append_event with server stopped; no LLM run',executionStatus=str(state.execution_status))
fixture_path.write_text(json.dumps(fixture,indent=2)+'\n',encoding='utf-8')
print(json.dumps(fixture))
