# Migrating from the OpenHands Local GUI to Agent Canvas

Agent Canvas replaces the legacy OpenHands OSS Local GUI, but it does not
perform a complete in-place migration of the old application's state. Both
applications use `~/.openhands` by default, so make a backup before starting
Agent Canvas and plan to re-enter credentials and review your settings.

This guide covers the legacy single-user Local GUI. It does not cover moving
data from OpenHands Cloud or Enterprise.

## Before you start

1. Stop the legacy Local GUI and any containers that mount `~/.openhands`.
2. Back up the entire directory, including hidden files and permissions.
3. Keep the backup until you have verified your settings and credentials in
   Agent Canvas.

On macOS or Linux:

```sh
cp -a "$HOME/.openhands" "$HOME/.openhands.local-gui-backup"
```

On Windows PowerShell:

```powershell
Copy-Item -Recurse -Force "$HOME\.openhands" "$HOME\.openhands.local-gui-backup"
```

> [!IMPORTANT]
> Do not copy individual files from the backup over files that Agent Canvas has
> already created. The two applications use different storage shapes, and a
> partial overwrite can leave credentials unreadable. Restore the complete
> backup only after stopping Agent Canvas, or use it as a reference while you
> reconfigure the new application.

## What is and is not reused

Agent Canvas points its local Agent Server at `~/.openhands`, so a limited
subset of legacy settings can be read. The rest must be recreated or reviewed.

| Legacy Local GUI data                                                               | Agent Canvas behavior                                                                                   | What you should do                                                                             |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Agent settings                                                                      | The compatible `agent_settings` block is reused and upgraded by the Agent Server.                       | Review the selected agent and model after first launch.                                        |
| Conversation settings                                                               | The compatible `conversation_settings` block is reused and upgraded.                                    | Review limits and other conversation defaults.                                                 |
| LLM API key in legacy settings                                                      | A plaintext legacy key cannot be decrypted when Agent Canvas starts with its generated `OH_SECRET_KEY`. | Re-enter the key in an LLM Profile.                                                            |
| Embedded `llm_profiles`                                                             | Legacy embedded profiles are not imported into the current profile store.                               | Recreate each profile under **Settings > LLM Profiles**.                                       |
| Custom secret names and descriptions                                                | Compatible entries may be listed.                                                                       | Verify every entry; do not assume its value migrated.                                          |
| Plaintext custom secret values                                                      | Values cannot be decrypted under the default generated `OH_SECRET_KEY`.                                 | Re-enter each value under **Settings > Secrets**.                                              |
| Git provider tokens                                                                 | Legacy `provider_tokens` are not consumed by the local Agent Server.                                    | Reconnect each Git provider or save the required token as directed by the current integration. |
| Language, analytics consent, sound notifications, Git identity, and disabled skills | Legacy top-level fields are not imported into the current `misc_settings.app_preferences` block.        | Review these preferences in Agent Canvas.                                                      |
| Conversation history                                                                | Legacy Local GUI conversations are not automatically listed or imported.                                | Keep the backup if you need the old event files; start new Agent Canvas conversations.         |
| `config.toml`                                                                       | Agent Canvas and its Agent Server do not use the legacy Local GUI `config.toml`.                        | Translate the settings you still need to the current UI or environment variables.              |

> [!WARNING]
> A profile or secret name appearing in the UI does not prove that its value is
> usable. Test the profile and re-enter sensitive values before relying on it.

## Install and start Agent Canvas

Choose one of the supported launch modes. The npm launcher runs the Agent
Server directly on the host and therefore gives the agent host filesystem
access. The Docker image is the isolated option.

### npm launcher

```sh
npm install -g @openhands/agent-canvas
agent-canvas
```

Open <http://localhost:8000>.

### Docker

Create a projects directory and continue mounting the same `~/.openhands`
directory after you have backed it up:

```sh
export PROJECTS_PATH="$HOME/projects"
mkdir -p "$PROJECTS_PATH" "$HOME/.openhands"

docker run -it --rm \
  -p 8000:8000 \
  -v "$HOME/.openhands:/home/openhands/.openhands" \
  -v "${PROJECTS_PATH}:/projects" \
  ghcr.io/openhands/agent-canvas:1.12.0 # x-release-please-version
```

Open <http://localhost:8000/canvas>. On Windows, use the equivalent commands in
the [Windows installation guide](../README.windows.md).

For the latest install commands and image version, use the
[project README](../README.md) rather than copying a version from this guide.

## Recreate credentials and profiles

### 1. Recreate LLM profiles

Open **Settings > LLM Profiles** and recreate every profile you still need.
Agent Canvas stores current profiles separately under
`~/.openhands/profiles/`; it does not import the legacy embedded
`settings.json.llm_profiles` collection.

For each profile:

1. Select the provider and model.
2. Re-enter the API key.
3. Re-enter a custom base URL if the profile used one.
4. Save and test the profile before moving to the next one.
5. Select the profile that should be active by default.

### 2. Re-enter custom secrets

Open **Settings > Secrets**. Re-enter each value that an agent, MCP server, or
automation needs. Legacy plaintext values can become unreadable after Agent
Canvas generates an encryption key, even when their names remain visible.

Agent Canvas persists its generated encryption key at
`~/.openhands/agent-canvas/secret-key.txt`. Preserve that file with the rest of
`~/.openhands` after migration. Losing or changing it makes values encrypted
with the previous key unreadable.

### 3. Reconnect Git providers

The legacy Local GUI stored provider tokens in a shape that the local Agent
Server does not consume. Reconnect GitHub, GitLab, Bitbucket, or Azure DevOps
using the current Agent Canvas integration. Verify repository access before
starting work that depends on private repositories.

### 4. Reconfigure MCP and search

Agent Canvas exposes MCP integrations under its MCP settings. If the legacy
setup used `SEARCH_API_KEY`, `TAVILY_API_KEY`, or the old Tavily proxy, install
and configure the current Tavily MCP server instead. Recreate any other MCP
servers and re-enter their credentials as secrets.

## Review application preferences

Review each of these after the first launch:

- language;
- analytics consent;
- sound notifications;
- Git author name and email;
- disabled skills;
- selected agent and LLM profile;
- workspace paths and mounted project directories.

Agent Canvas stores current frontend-owned preferences in the Agent Server's
`misc_settings.app_preferences` block. It does not import the legacy Local GUI's
top-level preference fields.

## Environment-variable mapping

The following table maps common legacy Local GUI settings to their Agent Canvas
equivalents.

| Legacy setting                                               | Agent Canvas equivalent                                                                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| `OH_PERSISTENCE_DIR`                                         | Still selects the Agent Server persistence root. The normal launchers default to `~/.openhands`.                                     |
| `FILE_STORE_PATH`                                            | Legacy fallback only; use `OH_PERSISTENCE_DIR`.                                                                                      |
| `SESSION_API_KEY`                                            | For normal use, set `LOCAL_BACKEND_API_KEY`; the launcher maps or generates the Agent Server session key.                            |
| Backend port `3000`                                          | The normal Agent Canvas entry point is port `8000`; internal services use separate ports.                                            |
| `VITE_BACKEND_BASE_URL` / `VITE_BACKEND_HOST` on port `3000` | Point frontend-only or development setups at the current Agent Server or ingress URL. See the [development guide](./DEVELOPMENT.md). |
| `WORKSPACE_BASE` or `workspace_base`                         | Use `VITE_WORKING_DIR` for the default local working directory, or mount a host `PROJECTS_PATH` at `/projects` with Docker.          |
| Legacy runtime and sandbox variables                         | Choose the npm host mode, the all-in-one Docker image, or a separately managed Agent Server backend.                                 |
| `LLM_BASE_URL` and provider defaults                         | Configure an LLM Profile in the UI.                                                                                                  |
| `SEARCH_API_KEY` / `TAVILY_API_KEY`                          | Configure Tavily as an MCP server.                                                                                                   |
| `OH_SECRET_KEY`                                              | Protects current settings and secrets. The launchers generate and persist one when it is not supplied.                               |

Do not reuse an environment variable only because its name still appears in an
old deployment file. Compare it with the current
[README](../README.md), [self-hosting guide](./SELF_HOSTING.md), and
[development guide](./DEVELOPMENT.md).

## Changed behavior to review

- The npm launcher and source development stack open at
  `http://localhost:8000`; the all-in-one Docker UI is served under `/canvas`.
- The npm launcher runs the Agent Server on the host without a sandbox. Use the
  Docker image when host filesystem isolation is required.
- Docker workspaces come from host directories mounted under `/projects`.
- Agent Canvas can connect to multiple Agent Server backends.
- LLM configurations are named profiles rather than the legacy embedded profile
  collection.
- MCP integrations have a dedicated settings experience.
- The full local stack includes the automation backend.
- The Terminal tab is an event transcript, not an interactive shell.
- The Browser tab renders browser-tool state; it is not a general-purpose
  manually controlled browser.
- Legacy Local GUI conversation history is not automatically imported.

## Verify the migration

Before deleting the backup, verify all of the following:

- [ ] The expected agent and active LLM profile are selected.
- [ ] Each LLM profile can make a successful request.
- [ ] Required custom secrets have been re-entered and tested.
- [ ] Git providers can list or access the expected repositories.
- [ ] MCP servers connect successfully.
- [ ] Git author name and email are correct.
- [ ] Language, analytics, sound, and disabled-skill preferences are correct.
- [ ] Workspace paths point only to directories the agent should access.
- [ ] Any required legacy conversation data remains preserved in the backup.

Keep `~/.openhands.local-gui-backup` until you have completed this checklist.
There is currently no automatic rollback or complete legacy conversation
importer.
