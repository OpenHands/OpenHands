# F02 — Conversations and their lifecycle

Start real work, review tool activity, resume it, and manage its history. Source:
[conversation route](../../../../src/routes/conversation.tsx),
[chat](../../../../src/components/features/chat),
[conversation controls](../../../../src/components/features/conversation).

## Sub-features

- `F02.create`: new conversation, streaming reply and real tool execution.
- `F02.workspace-context`: two workspace-backed conversations visible and correctly grouped.
- `F02.composer`: attachments, profile selection, available input modes and send/stop states.
- `F02.lifecycle`: stop/resume, approval/confirmation when enabled, reload and older history.
- `F02.manage`: rename, export, delete/cancel and overview metadata; sharing success is F11.

## How to get to it (user POV)

Use New Chat/home launcher or a workspace's add-conversation control; reopen from
the sidebar or `/conversations/:conversationId`. Use the composer and conversation
menu rather than creating conversations through a fixture API and calling that UI proof.

## Driving it with Playwright

Preconditions: authorized real LLM/ACP and bounded spend, empty disposable git
workspace. Use harmless prompts that edit/test only files in that workspace.

- **Create:** fill `interactive-chat-box` and click `submit-button`; request a tiny
  Python function plus a terminal assertion. Wait for actual assistant/tool events,
  then independently read the resulting file and run its assertion. A model's
  claim that it ran a test is insufficient.
- **Second conversation:** return to the same workspace and launch another small
  task; verify both conversations remain visible and point to the correct workspace.
- **Composer:** use `chat-input-llm-profile` for an approved alternate profile;
  attach a generated non-sensitive image through the real upload input. Verify the
  rendered attachment and agent response, and record unsupported provider modes.
- **Lifecycle:** stop a bounded task, resume, reload and reopen the same ID. Verify
  retained messages/tool events, load older history when present, and check no
  duplicate events appear. Exercise approval controls using an innocuous action.
- **Manage:** rename the disposable conversation, inspect overview, cancel a delete,
  export to private evidence, then delete only the run-owned fixture. Verify removal
  after reload and inspect desktop/mobile composer and menu geometry.

## Gotchas

Keep exported transcripts private until reviewed; prompts/tool definitions and
embedded service URLs can be sensitive. Do not issue arbitrary destructive prompts
just to reach approval states. A successful stopped/reloaded chat does not prove
Terminal history (F03). Cloud resume has sandbox prerequisites distinct from local.
