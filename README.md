<a name="readme-top"></a>

<div align="center">
  <img src="https://assets.openhands.dev/logo-whitebackground.png" alt="OpenHands logo" width="340">
  <h1 align="center" style="border-bottom: none">Agent Canvas</h1>
  <p align="center">
    <strong>The self-hosted developer control center for coding agents and automations.</strong>
  </p>
  <p align="center">
    Run OpenHands, Claude Code, Codex, Gemini, or any ACP-compatible agent across local, remote, and cloud backends.
  </p>
</div>
<div align="center">
  <a href="https://github.com/OpenHands/incubator-program"><img src="https://img.shields.io/badge/status-beta-blue?style=for-the-badge" alt="Project status beta"></a>
  <a href="https://github.com/OpenHands/OpenHands/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/OpenHands/OpenHands/ci.yml?branch=main&style=for-the-badge" alt="CI status"></a>
  <a href="https://www.npmjs.com/package/@openhands/agent-canvas"><img src="https://img.shields.io/npm/v/%40openhands%2Fagent-canvas?style=for-the-badge&logo=npm" alt="npm version"></a>
  <a href="https://docs.openhands.dev/openhands/usage/agent-canvas/backends"><img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Documentation"></a>
  <a href="https://go.openhands.dev/slack"><img src="https://img.shields.io/badge/Slack-Join%20the%20community-611f69?logo=slack&logoColor=white&style=for-the-badge" alt="Join us on Slack"></a>
</div>
<div align="center">
  <a href="#quickstart">Quickstart</a> |
  <a href="./docs/README.md">Docs</a> |
  <a href="./docs/SELF_HOSTING.md">Self-Hosting</a> |
  <a href="https://docs.openhands.dev/openhands/usage/agent-canvas/acp-agents">ACP Agents</a> |
  <a href="https://docs.openhands.dev/openhands/usage/agent-canvas/prebuilt-automations">Automations</a> |
  <a href="https://go.openhands.dev/slack">Slack</a>
</div>
<p align="center">
  <img src="https://assets.openhands.dev/screenshot/automation-preview.png" alt="Agent Canvas automation preview" width="100%">
</p>
<hr>

OpenHands Agent Canvas turns your coding agents into a self-hosted, always-on engineering team. It's a developer control center for starting conversations and automating everyday tasks — like generating reports that publish to Slack or automatically decomposing GitHub issues into tasks.

It runs locally on your machine by default, but can connect to multiple “agent backends”, e.g. running agents in Docker containers, on VMs, or within your company infrastructure. You can optionally choose to run agents on OpenHands Cloud or OpenHands Enterprise infrastructure.

Agent Canvas runs the open source OpenHands agent out-of-the-box, but can use any third-party agent like Claude Code and Codex.

|                                                                                                                      |                                                                                                                                          |
| -------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| [**Self-host your way**](https://docs.openhands.dev/openhands/usage/agent-canvas/backend-setup/vm)                   | Run agents locally, in Docker, on VMs, or anywhere you can run an agent server backend                                                   |
| [**Switch between different backends**](https://docs.openhands.dev/openhands/usage/agent-canvas/backends)            | Switch between local, remote, and cloud agents without losing focus                                                                      |
| [**Create automations**](https://docs.openhands.dev/openhands/usage/agent-canvas/prebuilt-automations)               | Create automations and workflows that integrate with Slack, GitHub, Linear, and more. Run on a schedule or in response to webhook events |
| [**Integrate with the tools you use**](https://docs.openhands.dev/openhands/usage/agent-canvas/prebuilt-automations) | Connect your automations with third-party services like Slack, GitHub, Notion, and more to automate workflows                            |
| [**Bring your own model**](https://docs.openhands.dev/openhands/usage/settings/llm-settings#llm-profiles)            | Use with any LLM                                                                                                                         |
| [**Use with any agent**](https://docs.openhands.dev/openhands/usage/agent-canvas/acp-agents)                         | Use with OpenHands, Claude Code, Codex, Gemini, or any agent with Agent-Client Protocol (ACP).                                           |

If you have questions or feedback, please open a GitHub issue or join the [#proj-agent-canvas channel in Slack](https://openhands.dev/joinslack).

## Quickstart

You can install OpenHands to run agents on any machine: on your laptop, on a dedicated computer like a Mac Mini,
or on a server in the cloud.

The most powerful way to run OpenHands is on a server in the cloud. This allows your agents to continue running
even when your laptop is shut, and makes it easier to trigger your agents through third-party services
like Slack, GitHub, and Datadog. See [SELF_HOSTING.md](docs/SELF_HOSTING.md) for details, especially with respect to security hardening.

Notably, you can run the backend in _multiple different environments_, and switch between
them from the same Agent Canvas frontend. E.g. you can share an Agent Server with your team for agents doing
code review and dependency updates, then have your personal agents running on your laptop.

### Option 1: Without a Sandbox

> [!WARNING]
> This runs the agent-server directly on the machine you're installing on — the agent will have full access to your filesystem!

**Prerequisites**: Node.js 22.12.x or later, `uv`

```sh
npm install -g @openhands/agent-canvas
agent-canvas
```

The `agent-canvas` command starts the full local stack by default. You can also split it when you want to run pieces separately:

```sh
agent-canvas --frontend-only  # static frontend + ingress only
agent-canvas --backend-only   # agent server + automation backend + ingress only
```

### Option 2: With a Docker Sandbox

**Prerequisites**:

- Docker: Docker Desktop on macOS/Windows, or Docker Engine/Docker Desktop on Linux.
- A host directory for `PROJECTS_PATH` containing the project folders you want the agent to access. Create it before starting the container.

**macOS / Linux:**

```sh
export PROJECTS_PATH="$HOME/projects"  # directory containing your project folders
mkdir -p "$PROJECTS_PATH" "$HOME/.openhands"

docker run -it --rm \
  -p 8000:8000 \
  -v "$HOME/.openhands:/home/openhands/.openhands" \
  -v "${PROJECTS_PATH}:/projects" \
  ghcr.io/openhands/agent-canvas:1.15.0 # x-release-please-version
```

**Windows (PowerShell / Windows Terminal):** See [README.windows.md](./README.windows.md) for the equivalent commands.

The agent will be able to access any project under `PROJECTS_PATH`.

### Option 3: From Source

> [!WARNING]
> This runs the agent-server directly on the machine you're installing on — the agent will have full access to your filesystem!

**Prerequisites**: Node.js 22.12.x or later, `npm`, `uv` (for running the agent server via `uvx`)

```sh
git clone https://github.com/OpenHands/OpenHands.git
cd OpenHands
npm install
npm run dev
```

---

Access the UI at [http://localhost:8000](http://localhost:8000) for the npm/source launchers, or [http://localhost:8000/canvas](http://localhost:8000/canvas) for the Docker image. You can add additional backends directly from the UI.

## Live Preview for real applications

Agent Canvas includes a real **Live Preview** for applications created by the agent. It does not render a mock, replay terminal output, or synthesize a page from chat messages. When the agent creates an application and starts its web server, the preview loads the application served by that process.

The end-to-end flow is:

```text
User prompt
  -> OpenHands agent
  -> real workspace changes
  -> npm / pnpm / yarn command
  -> real web server
  -> sandbox port forwarding
  -> browser-reachable URL
  -> Live Preview iframe
```

### How the URL is discovered

For Cloud sandboxes, the existing OpenHands forwarding contract is used directly. The sandbox publishes browser-reachable entries in `exposed_urls`; application entries use the `WORKER_*` naming convention. Agent Canvas polls the sandbox metadata while the runtime is starting and while a running sandbox has not published a worker URL yet. The first valid HTTP(S) worker URL is then used as the preview source.

The browser never needs to know the process ID, internal port, command output, or forwarding implementation. It receives only the final browser-reachable URL. When the agent edits files, the existing workspace mutation counter adds a cache-busting version to the preview URL, causing the real application to be fetched again.

### Starting an application

The agent can use the project’s normal development command, for example:

```sh
npm run dev -- --host 0.0.0.0
# or
pnpm dev --host 0.0.0.0
# or
npm start -- --host 0.0.0.0
```

The command must start a web server inside the sandbox using the runtime’s exposed worker service. The agent can call the Canvas UI control with `open_tab` and `tab="preview"` after the server is ready; an optional workspace entrypoint path can be supplied when the application also contains static files.

### Self-hosted Docker forwarding

A permanent self-hosted deployment must expose the Agent Server and the worker services through the same reverse-proxy topology. Configure the public sandbox URL pattern with `OH_SANDBOX_CONTAINER_URL_PATTERN` (or the legacy `SANDBOX_CONTAINER_URL_PATTERN`) using the `{port}` placeholder. The official [Docker Sandbox reverse-proxy guide](https://docs.openhands.dev/openhands/usage/sandboxes/docker#self-hosting-behind-a-reverse-proxy) describes the required routing.

For a single sandbox using host networking, OpenHands documents fixed service ports: `8000` for the Agent Server, `8001` for VS Code, and `8011`/`8012` for worker services. Dynamic multi-sandbox deployments must route the per-sandbox host ports generated by the sandbox provider instead of hard-coding one application port.

### Permanent deployment requirements

A static frontend build alone is not sufficient for the full experience. A permanent deployment needs the Agent Canvas web server, an Agent Server backend, a workspace-capable sandbox, and a reverse proxy that can reach the sandbox’s forwarded worker URLs. See [SELF_HOSTING.md](docs/SELF_HOSTING.md) for security hardening and [Configuration Options](https://docs.openhands.dev/openhands/usage/advanced/configuration-options) for the sandbox URL configuration.

# Architecture

Agent Canvas is powered by the [OpenHands Agent Server](https://github.com/OpenHands/software-agent-sdk/tree/main/openhands-agent-server/openhands/agent_server), a REST API for running multiple agents on a single machine. Each Agent Server runs on a single host/port; the Agent Canvas can connect to multiple Agent Servers and easily flip between them.

You can run an Agent Server anywhere:

- Directly on your laptop (be careful!)
- On a dedicated machine like a Mac Mini
- On a virtual machine in the cloud
- Inside OpenHands Cloud (our commercial offering)

The Agent Server is often paired with an [Automation Server](https://github.com/OpenHands/automation), which lets you set up agents that run on a schedule or in response to events.

<img width="1456" height="1258" alt="image" src="https://github.com/user-attachments/assets/cb6de6f5-ac30-4d04-a76a-b5c259f0c163" />

### Repository boundaries

Agent Canvas is part of a multi-repository OpenHands system. Changes should go to the repository that owns the behavior:

| Repository                                                                        | Responsibility                                                                                            |
| --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| [`OpenHands/OpenHands`](https://github.com/OpenHands/OpenHands)                   | Agent Canvas frontend, user-facing control center, backend selection, and local-stack orchestration.      |
| [`OpenHands/software-agent-sdk`](https://github.com/OpenHands/software-agent-sdk) | Python SDK, Agent Server, agents, tools, conversations, workspaces, events, and the canonical server API. |
| [`OpenHands/typescript-client`](https://github.com/OpenHands/typescript-client)   | Browser-compatible TypeScript client for the Agent Server API.                                            |
| [`OpenHands/automation`](https://github.com/OpenHands/automation)                 | Automation definitions, scheduling, webhooks, run history, and dispatching.                               |

The Agent Server API is implemented by the SDK and consumed through the TypeScript client by Agent Canvas. The automation service decides when work runs and dispatches conversations to the Agent Server/SDK, which decides what runs. See [`AGENTS.md`](./AGENTS.md) for contributor-specific boundaries and the required custom code-review guide.

## More documentation

- [Documentation index](./docs/README.md)
- [Architecture overview](./docs/architecture.md)
- [Development guide](./docs/DEVELOPMENT.md)
- [Self-hosting guide](./docs/SELF_HOSTING.md)
