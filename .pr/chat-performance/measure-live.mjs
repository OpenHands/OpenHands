// Observational probes wrap existing Control UI/API verbs. No app state injection.
import { spawn } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import { basename } from 'node:path';
const [run, conversation, label, viewport='desktop'] = process.argv.slice(2);
const cli='.agents/skills/verify-openhands/scripts/control-openhands';
async function call(args) {
 return new Promise((resolve,reject)=>{
  const p=spawn(cli,[...args,'--run',run]); let out='',err='';
  p.stdout.on('data',x=>out+=x); p.stderr.on('data',x=>err+=x);
  p.on('close',code=>code ? reject(new Error(out+err)) : resolve(JSON.parse(out)));
 });
}
const wait=ms=>new Promise(r=>setTimeout(r,ms));
await call(['browser','viewport',viewport]);
await call(['browser','scroll','testid=chat-scroll-container','--by','10000000']);
await call(['browser','fill','testid=chat-input','']);
await call(['browser','errors','--clear']);
await call(['browser','record','start','--feature','F05.long-conversation-performance','--name',`${label}-${viewport}`]);
const probe=`new Promise(resolve => {
 const t0=performance.now(), longTasks=[], inputFrames=[], frames=[];
 const input=document.querySelector('[data-testid=chat-input]');
 const heapBefore=performance.memory?.usedJSHeapSize;
 let last=t0, raf;
 const sample=now=>{ frames.push(now-last);last=now;raf=requestAnimationFrame(sample); };raf=requestAnimationFrame(sample);
 const observer=new PerformanceObserver(list=>{for(const e of list.getEntries())longTasks.push(e.duration);});observer.observe({type:'longtask'});
 const onInput=()=>{const start=performance.now();requestAnimationFrame(()=>requestAnimationFrame(()=>inputFrames.push(performance.now()-start)));};input.addEventListener('input',onInput);
 const summarize=xs=>{const a=[...xs].sort((a,b)=>a-b);return {count:a.length,median:a[Math.floor(a.length/2)]??null,p95:a[Math.floor(a.length*.95)]??null,max:a.at(-1)??null};};
 setTimeout(()=>{cancelAnimationFrame(raf);observer.disconnect();input.removeEventListener('input',onInput);resolve({durationMs:performance.now()-t0,longTasks: summarize(longTasks),longTaskTotalMs:longTasks.reduce((a,b)=>a+b,0),inputToSecondFrameMs:summarize(inputFrames),frameGapMs:summarize(frames),domNodes:document.querySelectorAll('*').length,groups:document.querySelectorAll('[data-testid=event-group]').length,heapBefore,heapAfter:performance.memory?.usedJSHeapSize,inputLength:input.textContent.length,scroll:{top:document.querySelector('[data-testid=chat-scroll-container]').scrollTop,height:document.querySelector('[data-testid=chat-scroll-container]').scrollHeight}});},6500);
})`;
const measuring=call(['browser','eval',probe]);
await wait(400);
const text='Typing stays available while recorded conversation updates arrive. Continue reviewing the output and keep this draft intact.';
const start=performance.now();
const typing=call(['browser','type','testid=chat-input',text]).then(result=>({elapsedMs:performance.now()-start,...result}));
const appending=(async()=>{const times=[];for(let i=0;i<6;i++){await wait(160);const t=performance.now();const result=await call(['api','POST',`/api/conversations/${conversation}/events`,'--write','--data',JSON.stringify({role:'user',content:[{type:'text',text:`${label} tail update ${i+1}: persisted fixture response.`}],run:false})]);times.push({elapsedMs:performance.now()-t,ok:result.ok,status:result.status});}return times;})();
const [measurement,typed,appends]=await Promise.all([measuring,typing,appending]);
const behavior=await call(['browser','eval',`({draftMatches:document.querySelector('[data-testid=chat-input]').textContent===${JSON.stringify(text)},composerFocused:document.activeElement===document.querySelector('[data-testid=chat-input]'),oldestHistoryPresent:document.body.textContent.includes('Seeded rendering fixture:'),lastIncomingEventPresent:document.body.textContent.includes(${JSON.stringify(label+' tail update 6: persisted fixture response.')})})`]);
const screenshot=await call(['browser','screenshot','--feature','F05.long-conversation-performance','--name',`${label}-${viewport}`]);
const video=await call(['browser','record','stop']);
const publicArtifact=({url,path,...rest})=>({...rest,path:basename(path)});
const result={label,viewport,seededHistory:true,liveModel:false,probe:'read-only PerformanceObserver and input-to-second-animation-frame; includes 10ms Playwright key delay',measurement:measurement.value,behavior:behavior.value,typed,appends,screenshot:publicArtifact(screenshot),video:publicArtifact(video)};
writeFileSync(`.pr/chat-performance/${label}-${viewport}.json`,JSON.stringify(result,null,2));
console.log(JSON.stringify(result,null,2));
