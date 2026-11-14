"""Repeat the real UI history recipe through the unmodified Control CLI.

The fixture is prepared separately while its own server is stopped. All reads
below observe DOM/network; navigation, scrolling, typing and disclosure clicks
are ordinary UI actions. This does not inject store state or intercept requests.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkout', type=Path, required=True)
p.add_argument('--run', type=Path, required=True)
p.add_argument('--projection', required=True)
p.add_argument('--tied', required=True)
p.add_argument('--label', choices=['before', 'after'], required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
cli = a.checkout / '.agents/skills/verify-openhands/scripts/control-openhands'
feature = 'F06.load-older-history'

def control(*args):
    proc = subprocess.run([str(cli), *args, '--run', str(a.run)], capture_output=True, text=True)
    result = json.loads(proc.stdout)
    if proc.returncode or not result.get('ok'):
        raise RuntimeError(f'Control {args[:3]} failed: {result}')
    return result

def observe():
    return control('browser', 'eval', """(()=>{const q=s=>[...document.querySelectorAll(s)];
      const titles=q('[data-testid=generic-event-message-title]').map(e=>e.textContent.trim());
      const summaries=q('[data-testid=event-group-toggle]').map(e=>e.textContent.trim());
      const markers=q('[data-testid=user-message]').map(e=>e.textContent.match(/CANVAS-TIED-[0-9]{3}/)?.[0]).filter(Boolean);
      const sc=document.querySelector('[data-testid=chat-scroll-container]');
      return {titles,summaries,toolRows:titles.length,command025:titles.filter(t=>t.includes('Recorded command 025')).length,
        markers,markerCount:markers.length,markerUnique:new Set(markers).size,
        inSequence:markers.every((v,i)=>i===0||markers[i-1]<v),
        scroll:sc?{top:sc.scrollTop,height:sc.scrollHeight,client:sc.clientHeight}:null,
        horizontalOverflow:document.documentElement.scrollWidth>innerWidth};})()""")['value']

results = []
for viewport in ['desktop', 'phone']:
    control('browser', 'viewport', viewport)
    for attempt in [1, 2]:
        for kind, cid in [('projection', a.projection), ('tied', a.tied)]:
            control('browser', 'network', '--clear')
            control('browser', 'goto', '/conversations/' + cid)
            control('browser', 'wait', 'testid=chat-input')
            control('browser', 'wait', 'text=History fixture complete.' if kind == 'projection' else 'text=CANVAS-TIED-100')
            initial = observe()
            name = f'{a.label}-{kind}-{viewport}'
            recording = kind == 'projection' and viewport == 'desktop' and attempt == 1
            if recording:
                control('browser', 'record', 'start', '--feature', feature, '--name', name, '--max-seconds', '90')
                time.sleep(1)
            control('browser', 'scroll', 'testid=chat-scroll-container', '--by', '-100000')
            control('browser', 'wait', 'text=Seeded history: 49 recorded' if kind == 'projection' else 'text=CANVAS-TIED-001')
            loaded = observe()
            artifacts = []
            if kind == 'projection':
                control('browser', 'click', 'testid=event-group-toggle')
                expanded = observe()
                control('browser', 'scroll', 'testid=generic-event-message-title >> has-text=Recorded command 025 >> nth=0')
                if attempt == 1:
                    shot = control('browser', 'screenshot', '--feature', feature, '--name', name)
                    artifacts.append(Path(shot['path']).name)
                control('browser', 'click', 'testid=generic-event-message-title >> has-text=Recorded command 025 >> nth=-1 >> role=button[name="Expand"][exact]')
                control('browser', 'wait', 'text=recorded-025')
                detail = control('browser', 'text', 'testid=event-group-content')['text']
                control('browser', 'fill', 'testid=chat-input', 'Projection verification draft; not submitted.')
                draft = control('browser', 'text', 'testid=chat-input')['text']
                control('browser', 'fill', 'testid=chat-input', '')
            else:
                expanded = None
                detail = None
                draft = None
            if recording:
                time.sleep(1)
                video = control('browser', 'record', 'stop')
                artifacts.append(Path(video['path']).name)
            network = control('browser', 'network', '--filter', 'events/search', '--last', '10')
            row = {'viewport':viewport,'attempt':attempt,'kind':kind,'initial':initial,'loaded':loaded,
                'expanded':expanded,'detailContainsBoundaryOutput':None if detail is None else 'recorded-025' in detail,
                'draftPreserved':None if draft is None else draft=='Projection verification draft; not submitted.',
                'requests':[{'query':r.get('query'),'status':r.get('status')} for r in network['recent']],
                'artifacts':artifacts}
            results.append(row)
            a.output.write_text(json.dumps({'label':a.label,'checks':results},indent=2)+'\n')
            print(json.dumps({'viewport':viewport,'attempt':attempt,'kind':kind,'summary':loaded['summaries'],
                'toolRows':expanded['toolRows'] if expanded else None,'boundaryRows':expanded['command025'] if expanded else None,
                'markers':loaded['markerCount'],'draftPreserved':row['draftPreserved']}),flush=True)
