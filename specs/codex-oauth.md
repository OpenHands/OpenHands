# Codex ChatGPT authentication

- CAO-001: Onboarding and Agent settings offer the same server-owned ChatGPT
  device-code login for Codex. Browser responses contain only safe status,
  verification instructions and an opaque polling handle.
- CAO-002: Cancelling, unmounting or switching backends invalidates the active
  login attempt. Late responses cannot mark another attempt/backend connected.
- CAO-003: Connected status comes from the Agent Server and is checked on reload
  and periodically. Disconnect invalidates status and secret metadata together.
- CAO-004: OpenAI API keys remain available. Manual auth.json is an advanced
  fallback. Cloud App-API backends without this server contract retain fallbacks.

The Agent Server and TypeScript client implementation belongs to
OpenHands/software-agent-sdk. The Canvas PR depends on its released client and
server version; compatibility pins must be updated to that release before merge.
