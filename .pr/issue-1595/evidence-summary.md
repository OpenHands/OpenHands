# Issue #1595 live evidence

Generated: 2026-07-13T09:46:43Z

## Scope

This evidence compares current `main` and PR #1584 with the real Agent Canvas frontend and the full local service stack (`npm run dev:static`: static frontend, ingress, Agent Server, automation backend). Both subscription runs used the same ChatGPT subscription profile:

- model: `openai/gpt-5.5`
- auth_type: `subscription`
- subscription_vendor: `openai`
- configured base_url: `https://chatgpt.com/backend-api/codex`
- subscription status: connected, account identifiers redacted

No raw tokens, API keys, cookies, account IDs, session keys, or conversation IDs are included in the committed artifacts.

## SHAs and stack

- current main SHA: `ed02badba37c3ff7aadd8be3c7499fb71d84a5d8`
- PR head SHA: `6728c155dff20a480ae9408ab51609c4fa4920da`
- Agent Server version: `1.35.0`
- SDK version: `1.35.0`
- workflow: static production frontend build served through the local ingress, not MSW or a mocked frontend

## Results

### Current main

- artifact directory: `.pr/issue-1595/current-main/`
- visible UI result: `Missing scopes: api.responses.write`
- browser outcome: `missing-scopes-visible`
- sanitized client `POST /api/conversations` payload: `agent_settings.llm.base_url` absent
- sanitized server event evidence: `ConversationErrorEvent` from `environment` containing `Missing scopes: api.responses.write`

### PR head

- artifact directory: `.pr/issue-1595/pr-head/`
- visible UI result: assistant completed the conversation with `ISSUE1595_PR_SUCCESS`
- browser outcome: `agent-reply-visible`
- sanitized client `POST /api/conversations` payload: `agent_settings.llm.base_url` present with `https://chatgpt.com/backend-api/codex`
- sanitized server event evidence: agent `MessageEvent` containing `ISSUE1595_PR_SUCCESS`

### PR head non-subscription

- artifact directory: `.pr/issue-1595/pr-head-non-sub/`
- sanitized client request: API-key mode, `is_subscription: false`, no subscription vendor/auth fields added, and no subscription base URL added
- live completion result: not marked successful
- access blocker: the real frontend and Agent Server reached the configured non-subscription LLM profile, but the available API-key credential was rejected by the LLM proxy with HTTP 401 `token_not_found_in_db`

## Artifacts

- combined GIF: `.pr/issue-1595/current-main-then-pr.gif`
- compact sanitized evidence: `.pr/issue-1595/client-server-evidence.json`
- current main result JSON: `.pr/issue-1595/current-main/result.json`
- PR head result JSON: `.pr/issue-1595/pr-head/result.json`
- PR head non-subscription result JSON: `.pr/issue-1595/pr-head-non-sub/result.json`

The GIF is generated from the real browser recordings only: current main first, then PR head.
