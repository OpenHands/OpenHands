<a name="readme-top"></a>
<div align="center">
  <img src="https://assets.openhands.dev/logo-whitebackground.png" alt="OpenHands logo" width="340">
  <p align="center">
    <strong>The self-hosted developer control center for coding agents and automations.</strong>
  </p>
  <p align="center">
    Run OpenHands, Claude Code, Codex, Gemini, or any ACP-compatible agent across local, remote, and cloud backends.
  </p>
</div>
<div align="center">
  <a href="https://github.com/OpenHands/incubator-program"><img src="https://img.shields.io/badge/status-beta-blue?style=for-the-badge" alt="Project status beta"></a>
  <a href="https://github.com/OpenHands/agent-canvas/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/OpenHands/agent-canvas/ci.yml?branch=main&style=for-the-badge" alt="CI status"></a>
  <a href="https://www.npmjs.com/package/@openhands/agent-canvas"><img src="https://img.shields.io/npm/v/%40openhands%2Fagent-canvas?style=for-the-badge&logo=npm" alt="npm version"></a>
  <a href="https://docs.openhands.dev/openhands/usage/agent-canvas/backends"><img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Documentation"></a>
  <a href="https://go.openhands.dev/slack"><img src="https://img.shields.io/badge/Slack-Join%20the%20community-611f69?logo=slack&logoColor=white&style=for-the-badge" alt="Join us on Slack"></a>
</div>
<div align="center">
  <a href="#quickstart">Quickstart</a> |
  <a href="https://docs.openhands.dev/overview/introduction">Docs</a> |
  <a href="https://docs.openhands.dev/openhands/usage/agent-canvas/backend-setup/vm">Self-Hosting</a> |
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

|    |    |
|---|---|
| [**Self-host your way**](https://docs.openhands.dev/openhands/usage/agent-canvas/backend-setup/vm) | Run agents locally, in Docker, on VMs, or anywhere you can run an agent server backend |
| [**Switch between different backends**](https://docs.openhands.dev/openhands/usage/agent-canvas/backends) | Switch between local, remote, and cloud agents without losing focus |
| [**Create automations**](https://docs.openhands.dev/openhands/usage/agent-canvas/prebuilt-automations) | Create automations and workflows that integrate with Slack, GitHub, Linear, and more. Run on a schedule or in response to webhook events  |
| [**Integrate with the tools you use**](https://docs.openhands.dev/openhands/usage/agent-canvas/prebuilt-automations) | Connect your automations with third-party services like Slack, GitHub, Notion, and more to automate workflows |
| [**Bring your own model**](https://docs.openhands.dev/openhands/usage/settings/llm-settings#llm-profiles) | Use with any LLM |
| [**Use with any agent**](https://docs.openhands.dev/openhands/usage/agent-canvas/acp-agents) | Use with OpenHands, Claude Code, Codex, Gemini, or any agent with Agent-Client Protocol (ACP). |


If you have questions or feedback, please open a GitHub issue or join the [#proj-agent-canvas channel in Slack](https://openhands.dev/joinslack).

> [!NOTE]
> The code in this repo is moving! See [Agent Canvas transition FAQ](https://github.com/OpenHands/OpenHands/issues/14841)
>
> The source code for OpenHands Agent and Agent Server lives in [OpenHands/software-agent-sdk](https://github.com/OpenHands/software-agent-sdk).
>
> The source code for Agent Canvas lives in [OpenHands/agent-canvas](https://github.com/OpenHands/agent-canvas).

## Quickstart

You can install OpenHands to run agents on any machine: on your laptop, on a dedicated computer like a Mac Mini,
or on a server in the cloud.

The most powerful way to run OpenHands is on a server in the cloud. This allows your agents to continue running
even when your laptop is shut, and makes it easier to trigger your agents through third-party services
like Slack, GitHub, and Datadog. See the [self-hosting guide](https://docs.openhands.dev/openhands/usage/agent-canvas/backend-setup/vm) for details, especially with respect to security hardening.

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
  ghcr.io/openhands/agent-canvas:1
```

**Windows (PowerShell / Windows Terminal):** See [README.windows.md](https://github.com/OpenHands/agent-canvas/blob/main/README.windows.md) for the equivalent commands.

The agent will be able to access any project under `PROJECTS_PATH`.

### Option 3: From Source

> [!WARNING]
> This runs the agent-server directly on the machine you're installing on — the agent will have full access to your filesystem!

**Prerequisites**: Node.js 22.12.x or later, `npm`, `uv` (for running the agent server via `uvx`)

```sh
git clone https://github.com/OpenHands/agent-canvas.git
cd agent-canvas
npm install
npm run dev
```

---

Access the UI at [http://localhost:8000](http://localhost:8000). You can add additional backends directly from the UI.

# Architecture

Agent Canvas is powered by the [OpenHands Agent Server](https://github.com/OpenHands/software-agent-sdk/tree/main/openhands-agent-server/openhands/agent_server), a REST API for running multiple agents on a single machine. Each Agent Server runs on a single host/port; the Agent Canvas can connect to multiple Agent Servers and easily flip between them.

You can run an Agent Server anywhere:

- Directly on your laptop (be careful!)
- On a dedicated machine like a Mac Mini
- On a virtual machine in the cloud
- Inside OpenHands Cloud (our commercial offering)

The Agent Server is often paired with an [Automation Server](https://github.com/OpenHands/automation), which lets you set up agents that run on a schedule or in response to events.

<img width="1456" height="1258" alt="image" src="https://github.com/user-attachments/assets/cb6de6f5-ac30-4d04-a76a-b5c259f0c163" />

## More documentation

- [Documentation index](https://docs.openhands.dev/overview/introduction)
- [Architecture overview](#architecture)
- [Development guide](./Development.md)
- [Self-hosting guide](https://docs.openhands.dev/openhands/usage/agent-canvas/backend-setup/vm)
