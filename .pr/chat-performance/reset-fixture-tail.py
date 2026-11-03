"""Reset only this recipe's appended measurement events while its server is stopped."""
import argparse,json
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--conversation',required=True)
a=parser.parse_args()
cid=a.conversation.replace('-','')
if len(cid)!=32 or any(c not in '0123456789abcdef' for c in cid): parser.error('UUID required')
root=a.run.resolve()/'private/state/dev_conversations'/cid
files=sorted((root/'events').glob('event-*.json'))
ready=[(i,f,json.loads(f.read_text())) for i,f in enumerate(files) if 'Fixture append readiness check.' in f.read_text()]
if len(ready)!=1: parser.error('one readiness marker required')
i,_,readiness=ready[0]
cutoff=i+2
for f in files[cutoff:]:
 d=json.loads(f.read_text())
 if d['kind']=='MessageEvent':
  text=''.join(x.get('text','') for x in d['llm_message']['content'])
  if not any(text.startswith(prefix+' tail update ') for prefix in ['baseline','after']): parser.error('non-measurement event; refusing reset')
 elif d['kind']!='ConversationStateUpdateEvent': parser.error('unexpected event; refusing reset')
for f in files[cutoff:]: f.unlink()
for f in (root/'events').glob('.eventlog-len-*.marker'): f.unlink()
(root/'events'/f'.eventlog-len-{cutoff}.marker').touch()
state=json.loads((root/'base_state.json').read_text())
state.update(leaf_event_id=readiness['id'],last_user_message_id=readiness['id'],head_is_empty=False,execution_status='idle')
(root/'base_state.json').write_text(json.dumps(state))
print(json.dumps({'retained':cutoff,'removedMeasurementEvents':len(files)-cutoff}))
