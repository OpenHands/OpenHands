# Windows quickstart (PowerShell)

This doc contains **Windows-specific** command syntax for running Agent Canvas on Windows, with or without a **Docker sandbox**.

For the main install options and overall context, see [README.md](./README.md).

## Option 1: Without a Docker Sandbox (Windows, PowerShell)

This runs the agent-server directly on the Windows host — the agent will have full access to your filesystem. It is verified on Windows 11 (x64) with Node.js 24 and `uv`; WSL and Docker are not required.

**Prerequisites**:

- [Node.js](https://nodejs.org/) 24 or later — check with `node --version`
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/): `winget install --id=astral-sh.uv -e`

```powershell
npm install -g @openhands/agent-canvas
agent-canvas
```

Open [http://localhost:8000/canvas](http://localhost:8000/canvas) in your browser, then pick an agent and configure an LLM provider in the setup wizard.

The first launch downloads the agent-server and automation backends through `uvx` (a few hundred packages from PyPI, several minutes); later launches reuse the uv cache. The launcher exits instead of retrying if that download fails, so rerun `agent-canvas` once the network is available. Canvas listens on ports 8000 (ingress), 18000 (agent-server), and 18001 (automation), plus an internal static frontend port (3001 by default) — if startup reports that a port is already in use, stop the process holding it and run `agent-canvas` again.

## Option 2: With a Docker Sandbox (Windows)

**Prerequisites**:

- Docker Desktop for Windows
- A host directory for `PROJECTS_PATH` containing the project folders you want the agent to access (create it before starting the container)

```powershell
docker pull ghcr.io/openhands/agent-canvas:1.34.0 # x-release-please-version

$env:PROJECTS_PATH = Join-Path $HOME "projects"  # directory containing your project folders
New-Item -ItemType Directory -Force -Path $env:PROJECTS_PATH, (Join-Path $env:USERPROFILE ".openhands") | Out-Null

docker run -it --rm `
  -p 127.0.0.1:8000:8000 `
  -e AGENT_CANVAS_ALLOW_LAN_SESSION_KEY=true `
  -v "$($env:USERPROFILE)\.openhands:/home/openhands/.openhands" `
  -v "$($env:PROJECTS_PATH):/projects" `
  ghcr.io/openhands/agent-canvas:1.34.0 # x-release-please-version
```

Open [http://localhost:8000/canvas](http://localhost:8000/canvas) in your browser.

The agent will be able to access any project under `PROJECTS_PATH`.

The quickstart restricts the published port to host loopback before explicitly enabling session-key injection. If you publish the port on a LAN or public interface, omit `AGENT_CANVAS_ALLOW_LAN_SESSION_KEY`, set `LOCAL_BACKEND_API_KEY` to a strong value, and enter that value in the UI.

## Option 3: With Multiple Docker Sandboxes (WSL 2)

For per-conversation containers, run Canvas inside a Linux distribution under WSL 2, with Docker Desktop's WSL integration enabled for that distribution. Install Node.js 24 or later and `uv` inside WSL, and confirm that `docker info` succeeds there.

Run the [multiple-sandbox quickstart](./README.md#option-3-with-multiple-docker-sandboxes) in the WSL shell, using workspace paths inside that Linux environment. The native Windows host is not supported by this runtime, which uses POSIX user/group IDs.
