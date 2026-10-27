Title: [Feature]: Maritime as a sandbox provider in Add Backend → Providers

### Problem or Motivation

Agent Canvas can run conversations in managed cloud sandboxes through one provider today, DigitalOcean Managed Agents, and the Providers tab says "More providers coming soon." People who want each agent in its own cloud VM, without running a VM, Docker host or Kubernetes cluster, have one option.

[Maritime](https://maritime.sh) hosts agents in Firecracker micro-VMs, one per agent, each with its own disk and a public HTTPS and WebSocket hostname. An idle VM sleeps with its memory kept and wakes on the next request. That maps directly onto an Agent Server backend: no tunnel is needed, and the backend is an ordinary remote agent-server.

This was discussed with Robert Brennan (All Hands) on 2026-10-21 as a sandbox provider integration following the DigitalOcean pattern.

### Desired Behavior

As a Canvas user, I open Add Backend → Providers, pick **Maritime**, and paste a Maritime API key. Canvas lists the OpenHands agents it created on my Maritime account with their state (running, sleeping, starting, failed). I can create a new one (name and memory size) and Canvas opens it when it is ready, or open an existing one, which wakes it if it sleeps. The opened agent becomes my active backend and I land in its latest conversation. I can put an agent to sleep and delete it (with confirmation). My Maritime API key stays out of the page, as the DigitalOcean token does.

### Acceptance Criteria

- [ ] With the desktop app (and with the web build when its server opts in), Add Backend → Providers lists Maritime next to DigitalOcean, and the "More providers coming soon" note stays.
- [ ] Signing in with a valid Maritime API key shows the account's plan and the OpenHands agents Canvas created on it, and no other Maritime agents. An invalid or under-scoped key shows why it was refused.
- [ ] Creating an agent with a name and memory size starts an OpenHands agent-server on Maritime and, once it answers, makes it the active backend; conversations, files and the terminal work against it.
- [ ] Opening a sleeping agent wakes it and connects to it.
- [ ] Putting an agent to sleep and deleting it work; deleting asks for confirmation and removes its backend from Canvas.
- [ ] A backend for a Maritime agent the user is not on is not polled in the background (it would keep the VM awake).
- [ ] The Maritime API key is never readable from page JavaScript.

### Alternatives Considered

- Documenting Maritime as a manual remote backend (create the agent by hand, paste URL and key into the Agent Server form). It works today but needs the user to create keys, set the port and CORS by hand, and gives no lifecycle management, which is what the Providers tab exists for.

### Additional Context

The agent-server image used is `ghcr.io/maritime-sh/openhands-agent-server:<agent-server version>`: the published `ghcr.io/openhands/agent-server:<version>-python` image plus a launcher that binds `$PORT` (Maritime routes an agent's hostname to `$PORT`). Source and publishing workflow: https://github.com/maritime-sh/openhands-agent-server
