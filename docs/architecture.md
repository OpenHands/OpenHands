# Agent Canvas architecture

Agent Canvas is a React and TypeScript frontend for running and monitoring OpenHands agents across local, remote, and hosted environments. It is adapted from the OpenHands frontend to talk directly to the OpenHands Agent Server and related automation services.

## System boundaries

Agent Canvas is responsible for:

- Rendering the agent conversation, terminal, browser, files, settings, and automation UI.
- Managing frontend state for conversations, backend selection, settings, profiles, and local metadata.
- Translating UI actions into OpenHands Agent Server API calls.
- Packaging the UI both as a standalone app and as library entrypoints for host applications.

Agent Canvas is not responsible for:

- Executing agent actions directly.
- Providing the sandbox or workspace isolation layer.
- Hosting LLM provider credentials outside the configured backend.
- Running scheduled or event-triggered automations without an automation backend.

## Runtime services

The primary backend is the [OpenHands Agent Server](https://github.com/OpenHands/software-agent-sdk/tree/main/openhands-agent-server/openhands/agent_server). Agent Canvas can connect to one or more Agent Server instances and switch between them from the UI.

Optional runtime services include:

- An ingress service that routes frontend, Agent Server, and automation traffic behind one local origin.
- An Automation Server for scheduled or event-triggered agent runs.
- OpenHands Cloud APIs for hosted sandbox and organization workflows.

Agent Canvas stack launchers expose runtime service information through the backend `/server_info.runtime_services` field. The frontend forwards that backend-provided information into new conversations as an agent context suffix so agents can use the correct URLs instead of guessing ports.

## Frontend modules

The most important source areas are:

- `src/api/`: service adapters for Agent Server, cloud, settings, git, skills, automations, and backend registry behavior.
- `src/components/`: route and feature UI, including conversation, chat, browser, files, settings, backend, automation, and onboarding components.
- `src/hooks/`: reusable React Query, state, and feature hooks.
- `src/stores/`: Zustand state stores for conversation and UI state.
- `src/i18n/`: translation resources and generated bundles.
- `src/mocks/`: MSW handlers for mock-mode development and tests.
- `bin/` and `scripts/`: CLI and development stack launchers.

## Runtime modes

Agent Canvas supports several modes:

| Mode                                 | Purpose                                                                                                                                             |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `npm run dev:docker`                 | Starts the UI with an Agent Server in a Docker sandbox. This is the default safer local workflow.                                                   |
| `npm run dev:dangerously-dockerless` | Starts the UI and Agent Server directly on the host. This is useful for servers and trusted environments, but the agent has host filesystem access. |
| `npm run dev:automation`             | Starts the local stack with an automation backend.                                                                                                  |
| `npm run dev:mock`                   | Runs the frontend against MSW mocks for UI development and tests.                                                                                   |
| `npm run build`                      | Builds the standalone application.                                                                                                                  |
| `npm run build:lib`                  | Builds library entrypoints for embedding Agent Canvas components.                                                                                   |

## Packaging and distribution

The npm package is `@openhands/agent-canvas`. The package exposes:

- The `agent-canvas` binary for launching a local stack.
- A standalone app build.
- Library entrypoints for browser, conversation, files, settings, sidebar, terminal, and i18n modules.

Tagged releases are published through the `Publish to npm` GitHub Actions workflow using npm trusted publishing and provenance.

## Quality gates

The main CI workflow installs dependencies with `npm ci` and runs:

- Typecheck, ESLint, and Prettier through `npm run lint`.
- Unit and component tests through `npm test`.
- Standalone app build through `npm run build`.
- Library build through `npm run build:lib`.
- Package verification through `npm pack --dry-run`.

Additional workflows run optional live end-to-end QA.

## Security posture

Local operation can give agents access to user workspaces. The README and self-hosting guide call out this risk and recommend Docker sandbox mode for laptop usage. Self-hosted deployments should use normal server hardening practices, authentication, HTTPS, firewall rules, and careful workspace scoping.
