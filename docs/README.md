# Agent Canvas docs

This directory contains the project documentation.

- [Architecture](./architecture.md): system boundaries, runtime modes, and quality gates.
- [Using ACP agents](./ACP_AGENTS.md): onboard and configure external agents (Claude Code, Codex, Gemini CLI, Pi, OpenCode).
- [Development guide](./DEVELOPMENT.md)
- [Canvas Extensions manual testing](./CANVAS_EXTENSIONS_TESTING.md)
- [Self-hosting guide](./SELF_HOSTING.md)
- [Integrating DefenseClaw](./DefenseClaw.md): run the DefenseClaw security governance layer alongside the Agent Server.
- [Testing matrix](./TESTING_MATRIX.md): release smoke-test coverage across installers, operating systems, and agents.
- [Reply-to-message intent](./intent/reply-message.md): confirmed product intent for quoting a replied-to message inside the follow-up.
- [Reply-to-message spec](./specs/reply-message.md): design for quoting a replied-to message inside the outgoing message (frontend-only envelope-in-text).
- [Feature verification map](../.agents/skills/verify-openhands/references/feature-map/README.md): every user-facing feature with entry points and rerunnable `control-openhands` recipes; start with the [verify-openhands skill](../.agents/skills/verify-openhands/SKILL.md) to launch and drive the real app.
