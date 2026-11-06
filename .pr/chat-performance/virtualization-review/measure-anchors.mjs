// Use the public Control OpenHands CLI for every browser operation.
// Run: node .pr/virtualization-review/measure-anchors.mjs RUN_DIR LABEL
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { writeFile } from "node:fs/promises";
import path from "node:path";
const exec = promisify(execFile);
const [run, label] = process.argv.slice(2);
if (!run || !label) throw new Error("Pass the isolated run directory and evidence label");
const cli = path.resolve(".agents/skills/verify-openhands/scripts/control-openhands");
const call = async (...args) => {
  const { stdout } = await exec(cli, [...args, "--run", run], { maxBuffer: 20 * 1024 * 1024 });
  return JSON.parse(stdout);
};
const readExpression = `(() => {
 const container=document.querySelector('[data-testid="chat-scroll-container"]');
 const viewportTop=container.getBoundingClientRect().top;
 const rows=[...document.querySelectorAll('[data-testid="agent-message"]')].flatMap(el=>{
  const match=el.textContent.match(/Pagination fixture message (\\d+)/);
  if(!match)return [];
  const box=el.getBoundingClientRect();
  return [{id:Number(match[1]),top:Math.round(box.top-viewportTop),height:Math.round(box.height)}];
 });
 return {scrollTop:container.scrollTop,scrollHeight:container.scrollHeight,clientHeight:container.clientHeight,virtual:!!document.querySelector('[data-testid="virtualized-message-list"]'),mountedRows:document.querySelectorAll('[data-testid="virtualized-message-row"]').length,domNodes:document.querySelectorAll('*').length,spinner:!!document.querySelector('[data-testid="loading-older-events"]'),rows};
})()`;
const read = async () => (await call("browser", "eval", readExpression)).value;
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const results = [];
await call("browser", "record", "start", "--feature", "F06.virtualization-review", "--name", label, "--max-seconds", "300");
try {
  for (let step = 0; step < 12; step += 1) {
    const initial = await read();
    await call("browser", "scroll", "testid=chat-scroll-container", "--by", String(200-initial.scrollTop));
    await pause(200);
    let before = await read();
    if (before.scrollTop < 100) {
      await call("browser", "scroll", "testid=chat-scroll-container", "--by", String(200-before.scrollTop));
      await pause(200);
      before = await read();
    }
    const anchor = before.rows.filter(row => row.top >= 0).sort((a,b)=>a.top-b.top)[0];
    if (!anchor) throw new Error("No visible message to anchor");
    const scrollBy = 50-before.scrollTop;
    // Observe geometry only. The scroll itself is driven by Control's scroll verb.
    const observations = call("browser", "eval", `new Promise(resolve => {
      const frames=[];const started=performance.now();let last='';
      const sample=()=>{const state=${readExpression};const serialized=JSON.stringify(state);
        if(serialized!==last){frames.push({ms:Math.round(performance.now()-started),...state});last=serialized;}
        if(performance.now()-started<1200)requestAnimationFrame(sample);else resolve(frames);};sample();
    })`);
    await pause(80);
    await call("browser", "scroll", "testid=chat-scroll-container", "--by", String(scrollBy));
    const frames = (await observations).value;
    await call("browser", "wait", "testid=loading-older-events", "--state", "hidden");
    await pause(200);
    const after = await read();
    const afterAnchor = after.rows.find(row => row.id === anchor.id);
    const result = { step, anchorId:anchor.id, beforeTop:anchor.top, afterTop:afterAnchor?.top ?? null, scrollBy, drift:afterAnchor ? afterAnchor.top-anchor.top+scrollBy : null, before, after, frames };
    results.push(result);
    console.log(JSON.stringify({step,virtual:after.virtual,anchorId:anchor.id,drift:result.drift,domNodes:after.domNodes,mountedRows:after.mountedRows}));
  }
} finally {
  const recording = await call("browser", "record", "stop", "--gif");
  const output = path.join(run,"evidence", "F06.virtualization-review", `${label}-anchors.json`);
  await writeFile(output,JSON.stringify({label,run,results,recording},null,2));
  console.log(JSON.stringify({output,recording}));
}
