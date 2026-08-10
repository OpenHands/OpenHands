# Windows quickstart (PowerShell)

This doc contains **Windows-specific** command syntax for running Agent Canvas with the **Docker sandbox**.

For the main install options and overall context, see [README.md](./README.md).

## Option 2: With a Docker Sandbox (Windows)

**Prerequisites**:

- Docker Desktop for Windows
- A host directory for `PROJECTS_PATH` containing the project folders you want the agent to access (create it before starting the container)

```powershell
docker pull ghcr.io/openhands/agent-canvas:1.12.0 # x-release-please-version

$env:PROJECTS_PATH = Join-Path $HOME "projects"  # directory containing your project folders
New-Item -ItemType Directory -Force -Path $env:PROJECTS_PATH, (Join-Path $env:USERPROFILE ".openhands") | Out-Null

docker run -it --rm `
  -p 8000:8000 `
  -v "$($env:USERPROFILE)\.openhands:/home/openhands/.openhands" `
  -v "$($env:PROJECTS_PATH):/projects" `
  ghcr.io/openhands/agent-canvas:1.12.0 # x-release-please-version
```

Open [http://localhost:8000/canvas](http://localhost:8000/canvas) in your browser.

The agent will be able to access any project under `PROJECTS_PATH`.
