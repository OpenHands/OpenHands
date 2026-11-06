// Re-run the existing Control verbs; this does not inject application state.
import { spawnSync } from 'node:child_process';
const run = process.argv[2];
const cli = '.agents/skills/verify-openhands/scripts/control-openhands';
function call(args) {
 const p = spawnSync(cli, [...args, '--run', run], { encoding:'utf8' });
 if(p.status) throw new Error(p.stdout+p.stderr);
 return JSON.parse(p.stdout);
}
for(let i=0;i<190;i++) {
 call(['browser','scroll','testid=chat-scroll-container','--by','10000000']);
 call(['browser','scroll','testid=chat-scroll-container','--by','-10000000']);
 call(['browser','wait','testid=loading-older-events','--state','hidden']);
 if(i%10===0) {
  const state=call(['browser','eval',`({firstLoaded: document.body.textContent.includes('Seeded rendering fixture:'), groups: document.querySelectorAll('[data-testid=event-group]').length, domNodes: document.querySelectorAll('*').length, scrollHeight: document.querySelector('[data-testid=chat-scroll-container]').scrollHeight})`]);
  console.log(JSON.stringify({step:i,...state.value}));
  if(state.value.firstLoaded) break;
 }
}
