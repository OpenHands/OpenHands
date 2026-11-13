export function activate(host) {
  return host.registerPage("echo", ({container}) => {
    const wrapper=document.createElement("section");
    wrapper.style.cssText="padding:1rem;height:100%;min-height:32rem";
    const heading=document.createElement("h1"); heading.textContent="QA owned bridge echo";
    const status=document.createElement("p"); status.dataset.testid="qa-bridge-capability";
    status.textContent=host.appBackendView?"App backend view available":"App backend view unavailable";
    const result=document.createElement("pre"); result.dataset.testid="qa-bridge-result";result.textContent="Waiting for real iframe";
    const frameArea=document.createElement("div");frameArea.style.cssText="height:28rem";
    wrapper.append(heading,status,result,frameArea);container.append(wrapper);
    const receive=(event)=>{
      const frame=frameArea.querySelector("iframe");
      if(!frame || event.source!==frame.contentWindow || !event.data || event.data.kind!=="qa-bridge-echo-result")return;
      // Only the fixture's explicit safe result fields; no cookie or key access.
      result.textContent=JSON.stringify({http:event.data.http,websocket:event.data.websocket});
    };
    window.addEventListener("message",receive);
    const dispose=host.appBackendView?.mount({container:frameArea});
    return ()=>{dispose?.();window.removeEventListener("message",receive);wrapper.remove();};
  });
}
