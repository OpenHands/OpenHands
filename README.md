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

Install Docker Desktop on macOS/Windows, or Docker Engine/Docker Desktop on Linux.
Windows users can follow the [PowerShell quickstart](./README.windows.md).

The steps below are for macOS and Linux. Settings, secrets, and conversations
are saved in `~/.openhands`. Your projects are mounted at `/projects` inside the
container.

**1. Prepare the host directories**

Set `PROJECTS_PATH` to the directory containing the projects you want the agent
to work on. Create both directories before starting Docker so it does not
create them owned by root.

```sh
export PROJECTS_PATH="$HOME/projects"  # directory containing your project folders
mkdir -p "$PROJECTS_PATH" "$HOME/.openhands"
```

On **Linux Docker Engine**, the container's `openhands` user (uid 10001) needs
write access to these directories. Run this on the host, then start the
container below:

```sh
find "$HOME/.openhands" "$PROJECTS_PATH" -type d -exec chmod a+rwX {} +
find "$PROJECTS_PATH" -type f -exec chmod a+rw {} +
```

This grants **all host users** directory write access and read/write access to
existing project files, so uid 10001 can save edits as well as create files.
Existing executable permissions are preserved. Files under `~/.openhands`
keep their permissions, including the `600` secret files; keep `PROJECTS_PATH`
separate from that state directory. Use only a workspace you are comfortable
sharing with other host users. Anyone who can traverse your home directory can
replace files in the writable state directories, including the session key.
Existing host-owned keys and databases may also be inaccessible to uid 10001;
grant that uid access without making secret keys world-readable. Docker Desktop on macOS/Windows handles
bind-mount permissions, so you can skip these commands there.

**2. Start Agent Canvas**

```sh
docker run -it --rm \
  -p 8000:8000 \
  -v "$HOME/.openhands:/home/openhands/.openhands" \
  -v "${PROJECTS_PATH}:/projects" \
  ghcr.io/openhands/agent-canvas:1.12.0 # x-release-please-version
```

Open [http://localhost:8000/canvas](http://localhost:8000/canvas).

<details>
<summary>Linux permission troubleshooting</summary>

If startup reports `Permission denied`, check the host directory named in the
error. On SELinux hosts, check whether the bind mount needs a `:z` or `:Z` label.
If Docker created it owned by root, run the commands with
`sudo` on the host:

```sh
sudo find "$HOME/.openhands" "$PROJECTS_PATH" -type d -exec chmod a+rwX {} +
sudo find "$PROJECTS_PATH" -type f -exec chmod a+rw {} +
```

Keep the state-directory command scoped to `-type d`. It preserves the `600`
permissions on `secret-key.txt` and `api-key.txt`; a recursive `chmod -R` would make those
secrets readable by other host users. The file-mode change applies only to
project files, which become readable and writable by all host users.

Keep the default container user. Running with `--user` can prevent creation of
`/home/openhands/.config`, which browser tools need. Startup checks this path
(or `XDG_CONFIG_HOME` when set) and fails early if it is unwritable. If you
override the user, configure a writable `XDG_CONFIG_HOME` as well as writable
state and workspace mounts.

</details>

<details>
<summary>Alternative: store state in a named Docker volume</summary>

A fresh named volume inherits the image directory's ownership, so uid 10001 can
write state without changing permissions on `~/.openhands`.

Use the `PROJECTS_PATH` prepared in step 1. On Linux Docker Engine, grant
directory and existing-file write access to that workspace before starting the
container:

```sh
find "$PROJECTS_PATH" -type d -exec chmod a+rwX {} +
find "$PROJECTS_PATH" -type f -exec chmod a+rw {} +
```

The same permission caveats apply: all host users gain directory write access
and read/write access to existing project files. Existing executable permissions
are preserved. Keep this workspace separate from `~/.openhands`.

```sh
docker run -it --rm \
  -p 8000:8000 \
  -v openhands-state:/home/openhands/.openhands \
  -v "${PROJECTS_PATH}:/projects" \
  ghcr.io/openhands/agent-canvas
```

Reuse `openhands-state` to keep settings and conversations across container
removals. Docker manages this volume separately from `~/.openhands` and does not
copy existing bind-mounted state into it. To find its location for inspection
or backup, run `docker volume inspect openhands-state`.

</details>

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

- [Documentation index](./docs/README.md)
- [Architecture overview](./docs/architecture.md)
- [Development guide](./docs/DEVELOPMENT.md)
- [Self-hosting guide](./docs/SELF_HOSTING.md)
