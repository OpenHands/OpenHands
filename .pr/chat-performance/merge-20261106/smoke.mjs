// Drive existing Control verbs against an inert real-backend conversation.
import { spawn } from 'node:child_process';
import { copyFileSync, writeFileSync } from 'node:fs';
import { basename } from 'node:path';
const [run, conversation, viewport = 'desktop'] = process.argv.slice(2);
const dir = '.pr/chat-performance/merge-20261106';
const cli = '.agents/skills/verify-openhands/scripts/control-openhands';
async function call(args) {
  return new Promise((resolve, reject) => {
    const child = spawn(cli, [...args, '--run', run]);
    let out = '', err = '';
    child.stdout.on('data', chunk => { out += chunk; });
    child.stderr.on('data', chunk => { err += chunk; });
    child.on('close', code => code ? reject(new Error(out + err)) : resolve(JSON.parse(out)));
  });
}
async function check(expression) {
  const result = await call(['browser', 'eval', expression]);
  if (!Object.values(result.value).every(Boolean)) throw new Error(JSON.stringify(result.value));
  return result.value;
}
const draft = `Unsent ${viewport} draft survives incoming events and disclosure changes.`;
await call(['browser', 'viewport', viewport]);
await call(['browser', 'scroll', 'testid=chat-scroll-container', '--by', '10000000']);
await call(['browser', 'fill', 'testid=chat-input', '']);
await call(['browser', 'errors', '--clear']);
if (viewport === 'desktop') await call(['browser', 'record', 'start', '--feature', 'F06.thinking', '--name', `merge-${viewport}`]);
const incomingText = `Merge integration update for ${viewport}; agent execution remains disabled.`;
await Promise.all([
  call(['browser', 'type', 'testid=chat-input', draft]),
  new Promise(r => setTimeout(r, 150)).then(() => call(['api', 'POST', `/api/conversations/${conversation}/events`, '--write', '--data', JSON.stringify({ role: 'user', content: [{ type: 'text', text: incomingText }], run: false })])),
]);
await call(['browser', 'wait-text', incomingText]);
const typing = await check(`({draft:document.querySelector('[data-testid=chat-input]').textContent===${JSON.stringify(draft)},focused:document.activeElement===document.querySelector('[data-testid=chat-input]'),historyLoaded:document.body.textContent.includes('Seeded rendering fixture:')})`);
await call(['browser', 'click', 'testid=event-group-toggle >> nth=-1']);
const replyReasoning = await check(`({content:document.querySelector('[data-testid=thinking-content]')?.textContent==='Fixture reply reasoning: this opens directly from its Thinking group.',noNestedDisclosure:document.querySelectorAll('[data-testid=collapsible-thinking-toggle]').length===0})`);
await call(['browser', 'click', 'testid=event-group-toggle >> nth=-2']);
await call(['browser', 'click', 'testid=collapsible-thinking-toggle']);
const expandedSteps = await call(['browser', 'count', 'testid=event-group-content >> role=button[name="Collapse"][exact]']);
if (expandedSteps.count) await call(['browser', 'click', 'testid=event-group-content >> role=button[name="Collapse"][exact] >> nth=-1']);
await call(['browser', 'click', 'testid=event-group-content >> role=button[name="Expand"][exact] >> nth=-1']);
const actionReasoning = await check(`(()=>{const groups=[...document.querySelectorAll('[data-testid=event-group]')];const actionGroup=groups.at(-2);const thoughts=[...document.querySelectorAll('[data-testid=agent-message]')].filter(el=>el.textContent.includes('Fixture thought: inspect the final recorded command.'));return {reasoningInToolGroup:actionGroup?.textContent.includes('Fixture action reasoning: this belongs to the final command group.'),commandDetails:actionGroup?.textContent.includes('fixture 0099'),thoughtOnce:thoughts.length===1,thoughtBeforeGroup:!!(thoughts[0]?.compareDocumentPosition(actionGroup)&Node.DOCUMENT_POSITION_FOLLOWING)}})()`);
await call(['browser', 'click', 'testid=event-group-toggle >> nth=-2']);
await call(['browser', 'click', 'testid=event-group-toggle >> nth=-2']);
const retention = await check(`({draft:document.querySelector('[data-testid=chat-input]').textContent===${JSON.stringify(draft)},replyReasoning:document.querySelector('[data-testid=thinking-content]')?.textContent.includes('Fixture reply reasoning:')})`);
const shot = await call(['browser', 'screenshot', '--feature', 'F06.thinking', '--name', `merge-${viewport}`]);
const video = viewport === 'desktop' ? await call(['browser', 'record', 'stop']) : null;
const errors = await call(['browser', 'errors', '--app-only', '--no-warnings']);
for (const item of [shot, video].filter(Boolean)) copyFileSync(item.path, `${dir}/${basename(item.path)}`);
const result = {viewport, seededHistory: true, liveModel: false, typing, replyReasoning, actionReasoning, retention, errors, artifacts:[shot, video].filter(Boolean).map(item => basename(item.path))};
writeFileSync(`${dir}/${viewport}.json`, JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));
// Return to the identical collapsed disclosure state for the next viewport.
await call(['browser', 'click', 'testid=event-group-toggle >> nth=-2']);
await call(['browser', 'click', 'testid=event-group-toggle >> nth=-1']);
