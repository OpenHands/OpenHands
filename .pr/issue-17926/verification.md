# Conversation draft persistence — issue #17926

The fix flushes a pending conversation draft through the existing conversation store before switching conversations or unmounting. It retains the input's owning conversation ID and original DOM element. Layout cleanup reads the rendered text before detachment, preserving multiline drafts and observing the synchronous input clear performed by submission. The ordinary 500 ms typing debounce, home sessionStorage path, task-ID transfer, and confirmed-delivery clear behavior remain covered.

## Real app setup

- Base source: `8bb2293410c3488828a532b52691b7aaac10797d`.
- Production app: `node bin/agent-canvas.mjs`, static frontend at `http://127.0.0.1:8000`.
- Real local OpenHands Agent Server 1.50.1 and automation 1.17.0; same backend processes, isolated state, generated authentication, and ports before and after.
- Two idle OpenHands conversations, created through the real API without an initial message, API key, or inference. Their unused model setting is `openai/gpt-5.4-mini`.
- The global active agent profile points to official `@agentclientprotocol/codex-acp` 2.1.1. It is unauthenticated and unused; no ACP process, initialization, session, account read, login, or model request was invoked. Selecting this real local profile enables the composer in the product's supported global agent-switching state.
- System Chromium and Playwright record the real UI. Consent overlays are dismissed. Keyboard editing and normal navigation links trigger the behavior; no API response mocks, DOM overrides, or simulated restored drafts are used.

Only unsent draft persistence is verified. This scenario does not depend on model execution. These captures do not claim to verify sending messages or LLM responses.

## Reproduction

The same browser script runs against both frontend builds. It seeds older drafts through the composer and waits for the normal debounce, replaces them with newer text, leaves within 500 ms, and returns. It also checks conversation ownership and native multiline editing.

1. Save an older draft in conversation A; replace it with a quick edit and select B. Return to A.
2. Save an older draft in A; replace it with a quick edit, select Settings, and use browser Back.
3. Type three lines using native Shift+Enter, select New Chat, and return to A.

The before recording restores the older drafts in the first two cases and loses the multiline draft. The after recording restores the latest text, including line breaks. Conversation B retains its own draft throughout.

| Flow | Before: input to route | Before restoration | After: input to route | After restoration |
| --- | ---: | --- | ---: | --- |
| A → B → A | 257.9 ms | Older saved draft | 141.0 ms | Latest edit |
| Settings → Back | 146.4 ms | Older saved draft | 137.9 ms | Latest edit |
| Multiline → New Chat → A | 291.6 ms | Empty | 199.1 ms | All three lines |

[Before video](before.mp4), [after video](after.mp4), [before observations](before-results.json), and [after observations](after-results.json) show the actual runs. [Build provenance](build-manifest.json) records source and compiled-asset fingerprints. The same running production launcher/static server served the fixed compiled assets from the fix worktree; its launcher files are byte-identical to the base. The original base build is preserved separately.

A second genuine Chromium run uses an Android-tablet user agent to exercise the app's native Enter-to-newline behavior. It creates `First draft line<div>Second draft line</div><div>Third draft line</div>` through keyboard actions. Navigation occurs in 311.9 ms before and 184.9 ms after; all three lines return only after the fix. [Native-block before](native-blocks-before.webm), [native-block after](native-blocks-after.webm), and the corresponding observations establish this browser representation. This is browser emulation, not a claim of physical-device testing.

The real app also makes ordinary workspace/bash/auth-session requests and a public feature-flag POST to `z.openhands.dev/flags/`, which is listed in the observations despite telemetry opt-out. Request metadata contains methods and URLs only, without headers or credential bodies. Real backend readback confirms both conversations remain idle with zero events and no ACP subprocess was launched.

## Verification commands

Each shell activates `/workspace/.openhands-setup/activate.sh` (Node 24.15.0 / npm 10.5.0).

```sh
npm test -- __tests__/hooks/use-draft-persistence.test.tsx
npm run lint
npm test -- --maxWorkers=3
npm run build
npm run build:lib
```

Focused TDD coverage has 48 passing tests. The original navigation regression tests failed before the repair; the multiline lifecycle test also failed with passive cleanup and passed with layout cleanup. The new navigation cases use the actual durable store and remount/readback.

Lint passes with 376 existing warnings and no errors, matching untouched upstream. App and library builds pass. The complete suite finishes with 8,103 passes, three failures, and seven TODOs across 767 files. All three failures are launcher tests reproduced on untouched upstream: the default `/home/agent/.openhands/agent-canvas` directory cannot be created in this environment.

A retry of both failing files using the supported `OH_CANVAS_SAFE_STATE_DIR=/workspace/17926-evidence/test-state` override passes 65 of 66 tests, including both SIGHUP cases. The remaining missing-uvx test discards the parent environment, so it cannot inherit that override and still exits before reaching its target behavior. No tests are skipped or weakened, no permissions are expanded, and no unrelated launcher code is changed. The complete suite is therefore not reported as green.

## Artifact handling

The MP4 files are faithful transcodes of Playwright's WebM recordings. They contain actual app frames, without generated scenes or synthetic result overlays. This directory contains temporary reviewer artifacts. This is a fork branch, so remove `.pr/issue-17926/` before merge; do not rely on automatic cleanup.
