"""Copy only seeded history into another stopped, isolated Control conversation."""
import argparse,json,shutil
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-run',type=Path,required=True)
p.add_argument('--target-run',type=Path,required=True)
p.add_argument('--conversation',required=True)
a=p.parse_args();cid=a.conversation.replace('-','')
if len(cid)!=32 or any(c not in '0123456789abcdef' for c in cid): p.error('UUID required')
source=a.source_run.resolve()/'private/state/dev_conversations'/cid
target=a.target_run.resolve()/'private/state/dev_conversations'/cid
if source==target or not (target/'base_state.json').is_file() or (target/'events').exists(): p.error('fresh distinct target conversation required')
shutil.copytree(source/'events',target/'events')
s=json.loads((source/'base_state.json').read_text());t=json.loads((target/'base_state.json').read_text())
for key in ['leaf_event_id','last_user_message_id','head_is_empty']:
 if key in s:t[key]=s[key]
(target/'base_state.json').write_text(json.dumps(t))
m=json.loads((target/'meta.json').read_text());m['title']=json.loads((source/'meta.json').read_text())['title'];(target/'meta.json').write_text(json.dumps(m))
print(json.dumps({'copiedEvents':len(list((target/'events').glob('event-*.json'))),'conversation':a.conversation,'credentialsCopied':False}))
