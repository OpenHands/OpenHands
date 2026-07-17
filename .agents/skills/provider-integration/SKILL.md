# Secure LLM Provider Integration

## Trigger

Use when adding or changing an LLM provider, model catalog, API base URL, API key flow, authentication flow, or provider-specific settings.

## Required Inputs

- Provider name
- Provider documentation or confirmed OpenAI-compatible endpoint
- Model identifier
- Credential source
- Target surfaces: settings UI, CLI, server, runtime, deployment

## Workflow

1. Locate the existing OpenHands LLM configuration and LiteLLM integration paths.
2. Confirm whether the provider is natively supported or OpenAI-compatible.
3. Keep model name, base URL, and credentials configurable.
4. Reuse existing settings and secret-storage abstractions.
5. Keep credentials server-side and out of frontend bundles.
6. Add model-list entries only after confirming model capability and naming.
7. Add focused tests for configuration mapping, validation, masking, and failure behavior.
8. Document local, Codespaces, and production secret setup without including real values.
9. Run relevant backend checks, frontend checks, tests, and build.

## Canonical OpenCode Go Variables

- `OPENCODE_GO_API_KEY`
- `OPENCODE_GO_BASE_URL`
- `OPENCODE_GO_MODEL`

Do not hard-code an endpoint or model that has not been confirmed from current provider documentation or the user's account dashboard.

## Authentication Boundary

Website login, OpenAI API authentication, and browser-based ChatGPT/Codex login are separate systems. Do not exchange, scrape, or repurpose browser session credentials as generic model API credentials.

## Verification

- Secret is ignored by Git and never printed.
- Missing configuration fails with a useful message.
- Provider configuration maps into the existing OpenHands LLM client.
- Existing providers remain unchanged.
- Tests and build results are reported accurately.
