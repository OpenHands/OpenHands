# Pair a phone with Build Remote Agent

Agent Canvas can use **Build Remote Agent** as a pairing device: the paid
iOS/Android app spectates (and can inject into) the desktop OpenHands session
through the free MIT `gbr-agent`. Phone and PC never open ports to each other.

Skills and MCP catalog entries live in [`OpenHands/extensions`](https://github.com/OpenHands/extensions)
(`skills/gbr`). The Software Agent SDK example is
[`07b_gbr_mcp.py`](https://github.com/OpenHands/software-agent-sdk/blob/main/examples/01_standalone_sdk/07b_gbr_mcp.py).
This page is the Canvas-side how-to only.

Website: https://grokbuildremote.com/
Agent: https://github.com/LinespottingOrg/GrokBuildRemote-Agents (MIT)
Protocol: `gbr/1` · need agent **v0.6.0+**

Not affiliated with xAI or SpaceX.

## Install + pair

Run Canvas as usual (`agent-canvas` or `npm run dev`). In another terminal:

```bash
# macOS / Linux
curl -fsSL https://grokbuildremote.com/install.sh | bash
gbr-agent version          # must print v0.6.0 or newer
gbr-agent pair             # QR in browser + printed 8-char code
gbr-agent run              # leave running
```

```powershell
# Windows
irm https://grokbuildremote.com/install.ps1 | iex
gbr-agent version
gbr-agent pair
gbr-agent run
```

Phone: open Build Remote Agent → **Scan QR from computer** (or type the 8-char
code). Sessions appear in the app. **Unpair** in Settings before changing PCs.
Force-close is not enough.

## Attach this agent

After `gbr-agent run`:

- HTTP Bot API: `http://127.0.0.1:8788`
- MCP stdio: clone the agent repo and run `node mcp/gbr-mcp/bin/gbr-mcp.js`

Canvas itself stays on its ingress (`127.0.0.1:8000` by default). Do not add a
Canvas-native pair protocol.

```bash
curl -sS http://127.0.0.1:8788/health
curl -sS http://127.0.0.1:8788/v1/sessions
```

Phone is spectator. Orchestration stays on OpenHands (or a Grok bot / Claude
Cowork talking to the same Bot API).

Do not commit mailbox keys. Phone **Settings → Bot API** is the only place the
relay key is copied.

Self-hosting Canvas on a VM is a different remote path — see [SELF_HOSTING.md](SELF_HOSTING.md).
