#!/usr/bin/env python3
"""Build the #18445 before/after evidence page from the real browser screenshots.

The two PNGs are produced by repro-18445.mjs against the real built Canvas
bundle and the real agent-server session socket. This script embeds them in a
single dark-themed page that also records the observed resume cursor and the
rendered message count, then Playwright screenshots it to .pr/18445-evidence.png.
"""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
before_png = Path("/tmp/18445-before.png")
after_png = Path("/tmp/18445-after.png")
before_json = Path("/tmp/18445-before.json")
after_json = Path("/tmp/18445-after.json")

out_html = ROOT / ".pr/18445-browser-evidence.html"


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def facts(p: Path) -> dict:
    if not p.exists():
        return {}
    return json.loads(p.read_text())


bf, af = facts(before_json), facts(after_json)


def cursor(f: dict) -> str:
    urls = f.get("wsUrls") or []
    if not urls:
        return "n/a"
    return urls[0].split("after_seq=")[-1]


HTML = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<style>
  body {{ margin:0; background:#0d1117; color:#c9d1d9; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:13px; }}
  .wrap {{ padding:18px 20px 24px; }}
  h1 {{ font-size:15px; margin:0 0 4px; color:#e6edf3; font-family:system-ui,sans-serif; }}
  .sub {{ color:#8b949e; font-family:system-ui,sans-serif; font-size:12px; margin-bottom:14px; line-height:1.5; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
  .panel {{ border:1px solid #30363d; border-radius:8px; overflow:hidden; }}
  .bar {{ padding:6px 10px; font-size:12px; font-weight:600; font-family:system-ui,sans-serif; }}
  .before .bar {{ background:#3d1418; color:#ff7b72; border-bottom:1px solid #30363d; }}
  .after  .bar {{ background:#12261a; color:#3fb950; border-bottom:1px solid #30363d; }}
  img {{ display:block; width:100%; }}
  .cap {{ padding:8px 10px; border-top:1px solid #30363d; line-height:1.6; }}
  .fail {{ color:#ff7b72; }} .pass {{ color:#3fb950; }} .dim {{ color:#8b949e; }}
  .cmd {{ color:#79c0ff; }}
  .facts {{ margin-top:14px; border:1px solid #30363d; border-radius:8px; }}
  pre {{ margin:0; padding:10px 12px; white-space:pre-wrap; line-height:1.5; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Issue #18445 &mdash; completed conversation opens with an empty chat window</h1>
  <div class="sub">
    Reproduction through the <b>real</b> built Canvas bundle and the <b>real</b> agent-server
    session socket (<span class="cmd">ws://&hellip;/sockets/session/&lt;id&gt;</span>). Only the cloud
    API responses are mocked to create the bug condition:
    <span class="cmd">GET /api/v1/app-conversations</span> returns the completed conversation,
    <span class="cmd">GET /api/v1/conversation/&lt;id&gt;/events/search</span> returns <b>500</b> (the failing
    initial tail load), and <span class="cmd">GET /api/conversations/&lt;id&gt;/events/count</span> passes through
    to the live server and returns <b>200</b>. Conversation has 3 seeded user turns on the server.
  </div>

  <div class="grid">
    <div class="panel before">
      <div class="bar">BEFORE (base main) &mdash; silently empty chat</div>
      <img src="data:image/png;base64,{b64(before_png)}" alt="before: empty chat window" />
      <div class="cap">
        socket resume cursor: <span class="fail">after_seq={cursor(bf)}</span> (count&minus;1 &rarr; history skipped)<br/>
        rendered user messages: <span class="fail">{bf.get('userMessageCount', '?')}</span><br/>
        <span class="dim">the failed tail search was swallowed into an empty page, so the socket resumed past the whole log</span>
      </div>
    </div>
    <div class="panel after">
      <div class="bar">AFTER (PR head) &mdash; history replays</div>
      <img src="data:image/png;base64,{b64(after_png)}" alt="after: chat with replayed history" />
      <div class="cap">
        socket resume cursor: <span class="pass">after_seq={cursor(af)}</span> (replay from the start)<br/>
        rendered user messages: <span class="pass">{af.get('userMessageCount', '?')}</span><br/>
        <span class="dim">the failed tail load no longer seeds an empty page; the socket replays the log</span>
      </div>
    </div>
  </div>

  <div class="sub" style="margin-top:14px">
    A video of both runs, concatenated in order, is at <span class="cmd">.pr/18445-repro.webm</span>.
    The standalone driver is <span class="cmd">.pr/repro-18445.mjs</span>. Reproduce with:
    <span class="cmd">node .pr/repro-18445.mjs http://127.0.0.1:18300 &lt;conversationId&gt; /tmp/out</span>
    after swapping the built bundle under <span class="cmd">build/</span> between the base and PR-head commits.
  </div>

  <div class="facts">
    <pre><span class="dim">// raw observations from repro-18445.mjs (real browser, real socket)</span>
BEFORE  {{ "seededVisible": {str(bf.get('seededVisible')).lower()}, "userMessageCount": {bf.get('userMessageCount', '?')}, "wsUrls": {json.dumps(bf.get('wsUrls', []))} }}
AFTER   {{ "seededVisible": {str(af.get('seededVisible')).lower()}, "userMessageCount": {af.get('userMessageCount', '?')}, "wsUrls": {json.dumps(af.get('wsUrls', []))} }}</pre>
  </div>
</div>
</body>
</html>
"""

out_html.write_text(HTML)
print("wrote", out_html)
