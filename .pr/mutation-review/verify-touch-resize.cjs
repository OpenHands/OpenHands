const { chromium, devices } = require('playwright');
const fs = require('node:fs');
(async()=>{
 const browser = await chromium.launch({headless:true,channel:"chrome"});
 for (const [label,port] of [[process.argv[2] || 'before',Number(process.argv[3] || 18458)]]) {
  const context = await browser.newContext({...devices['iPhone 13'], viewport:{width:900,height:900}});
  const page = await context.newPage();
  await page.addInitScript(()=>{
   const listeners = [];
   const add = EventTarget.prototype.addEventListener;
   const remove = EventTarget.prototype.removeEventListener;
   const capture = opts => typeof opts === 'boolean' ? opts : !!opts?.capture;
   EventTarget.prototype.addEventListener = function(type, callback, options) {
    if (this instanceof Element && this.id === 'resize-grip' && ['touchmove','touchend'].includes(type)) listeners.push({target:this,type,callback,capture:capture(options)});
    return add.call(this,type,callback,options);
   };
   EventTarget.prototype.removeEventListener = function(type, callback, options) {
    const index = listeners.findIndex(x=>x.target===this && x.type===type && x.callback===callback && x.capture===capture(options));
    if(index>=0)listeners.splice(index,1);
    return remove.call(this,type,callback,options);
   };
   window.resizeListenerCount = () => listeners.length;
   localStorage.setItem('openhands-onboarded','1');
   localStorage.setItem('analytics-consent','false');
   localStorage.setItem('openhands-telemetry-consent','denied');
   localStorage.setItem('openhands-telemetry-first-use','true');
   localStorage.setItem('openhands-backends',JSON.stringify([{id:'default-local',name:'Local',host:location.origin,apiKey:'',kind:'local'}]));
   localStorage.setItem('openhands-active-backend',JSON.stringify({backendId:'default-local',orgId:null}));
  });
  page.on('console',m=>console.log(label,'console',m.type(),m.text().slice(0,400)));
  page.on('requestfailed',req=>console.log(label,'requestfailed',req.url(),req.failure()?.errorText));
  page.on('pageerror',e=>console.log(label,'error',e.message));
  await page.goto(`http://localhost:${port}/conversations/1`,{waitUntil:'domcontentloaded',timeout:90000});
  try { await page.locator('#resize-grip').waitFor({state:'attached',timeout:60000}); } catch(e) {await page.screenshot({path:`${__dirname}/${label}-failure.png`});console.log(label, await page.locator('body').innerText());throw e;}
  const consent = page.getByTestId('telemetry-consent-form');
  await consent.waitFor({state:'visible',timeout:15000});
  if (await consent.isVisible()) {
   await consent.getByRole('checkbox').uncheck();
   await consent.getByRole('button',{name:'Confirm preferences'}).click();
   await consent.waitFor({state:'hidden'});
  }
  const result = await page.evaluate(() => {
   const grip = document.querySelector('#resize-grip');
   const hit = grip.firstElementChild;
   const input = document.querySelector('[data-testid="chat-input"]');
   const rect = hit.getBoundingClientRect();
   const y = rect.y+2;
   const touch = (type, pos) => {
    const point = new Touch({identifier:1,target:hit,clientX:rect.x+30,clientY:pos});
    hit.dispatchEvent(new TouchEvent(type,{bubbles:true,cancelable:true,touches:type==='touchend'?[]:[point],changedTouches:[point],targetTouches:type==='touchend'?[]:[point]}));
   };
   const initialHeight = input.offsetHeight;
   touch('touchstart',y);
   const listenersDuringDrag = window.resizeListenerCount();
   touch('touchmove',y-80);
   touch('touchend',y-80);
   const heightAfterEnd = input.offsetHeight;
   const listenersAfterEnd = window.resizeListenerCount();
   touch('touchmove',y-140);
   return {initialHeight,heightAfterEnd,heightAfterLaterMove:input.offsetHeight,listenersDuringDrag,listenersAfterEnd};
  });
  console.log(label,'RESULT',JSON.stringify(result));
  fs.writeFileSync(`${__dirname}/${label}-browser-result.json`,JSON.stringify(result,null,2));
  await page.screenshot({path:`${__dirname}/${label}-initial.png`});
  console.log(label,await page.locator('body').innerText());
  await context.close();
 }
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
