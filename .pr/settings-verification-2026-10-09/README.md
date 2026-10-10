# Settings verification — 9 October 2026

Verified executable source: `57530c7664cc02c4715f84610618a1c504ab6082` (#16958).
Baseline: `95dcdcdf12bd78f02c52b80cccf2f5f4f35242a4` (main).
Both used isolated production `bin/agent-canvas.mjs` stacks, real Agent Server/SDK
1.53.0, the pinned Automation service, Node 25.9.0, installed Google Chrome,
English UI, and 1440×1000 viewports. Doctor confirmed authenticated settings,
unauthenticated rejection, compatible server versions, build inputs and Automation
health before driving. No API response or model response was mocked.

## Result

**Independent encrypted expiry: reproduced on main, corrected by the PR.**
After a newer display-settings read, main reused an expired encrypted entry and
created a conversation with `max_iterations=1`. The PR fetched encrypted settings
again and created it with the real server's current `max_iterations=2`. Both
conversation POSTs returned 201; GET of each created conversation returned the
persisted limits shown in the original CLI outputs:
[main](conversation-limit-main.json), [PR](conversation-limit-head.json).

The display-settings expiry and Git identity round-trip passed on both versions.
The two display screenshots show that common passing step, not a fabricated
before/after difference.

## TTL reproduction

A controlled browser `Date` was fixed at the times below using the documented
`browser clock --fixed` command. Server time was unchanged; browser reload was
used only at initialization, not between cache reads.

| Browser time | Real action                                                                                                                                                       |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 00:00:00     | Reload after arranging Git identity `QA cache original` and `max_iterations=1`; open Application settings to populate display settings.                           |
| 00:01:00     | New Chat → submit `QA cache verification. Do not run any tools.`; the normal creation path populates encrypted settings.                                          |
| 00:01:00     | API PATCH changes the server's Git identity to `QA cache updated`, without invalidating the page's caches.                                                        |
| 00:05:01     | Navigate through the sidebar to Settings → Application; display settings refresh and show the updated Git identity on both versions.                              |
| 00:05:01     | API PATCH changes the server's conversation limit to 2, without invalidating the page's caches.                                                                   |
| 00:06:01     | New Chat → submit `QA independent encrypted expiry. Do not run any tools.`; the encrypted entry is now older than five minutes, while display settings are fresh. |
| 00:06:01     | Read the created conversation's actual `max_iterations`: main 1, PR 2.                                                                                            |

The profile was named `qa-cache`, with a dummy key, `openai/gpt-4o-mini`, and
`http://127.0.0.1:9/v1` as its base URL, saved with `llm set --no-validate`.
It deliberately cannot execute a model. Completion failures are expected and are
not a test result. This check exercises the real settings/encryption and
conversation-creation boundary; it makes no LLM, tool-execution, or provider claim.

Use a fresh isolated run on each revision. The relevant reproducible commands are:

```sh
export PATH="$PWD/.agents/skills/verify-openhands/scripts:$PATH"
control-openhands launch --new --print-run
# Export the exact returned path as OH_VERIFY_RUN; run doctor and onboard --skip.
QA_SETTINGS_DUMMY_KEY=qa-only-not-a-real-credential control-openhands llm set \
  --profile qa-cache --model openai/gpt-4o-mini \
  --base-url http://127.0.0.1:9/v1 --api-key-env QA_SETTINGS_DUMMY_KEY --no-validate
control-openhands api PATCH /api/settings --write --data \
  '{"misc_settings_diff":{"app_preferences":{"git_user_name":"QA cache original","git_user_email":"qa-cache@example.com","user_consents_to_analytics":false}},"conversation_settings_diff":{"max_iterations":1}}'
control-openhands browser clock --fixed 2026-10-10T00:00:00Z
control-openhands browser reload
control-openhands browser click 'testid=backend-selector-settings-link' --expect-url /settings
control-openhands browser click 'testid=sidebar-settings-/settings/app' --expect-url /settings/app
control-openhands browser clock --fixed 2026-10-10T00:01:00Z
control-openhands browser click 'role=link[name="New Chat"]' --expect-url /conversations
control-openhands browser fill 'testid=chat-input' 'QA cache verification. Do not run any tools.'
control-openhands browser click 'testid=submit-button' --expect-url '/conversations/[^?]+'
control-openhands api PATCH /api/settings --write --data \
  '{"misc_settings_diff":{"app_preferences":{"git_user_name":"QA cache updated"}}}'
control-openhands browser clock --fixed 2026-10-10T00:05:01Z
control-openhands browser click 'testid=backend-selector-settings-link' --expect-url /settings
control-openhands browser click 'testid=sidebar-settings-/settings/app' --expect-url /settings/app
control-openhands browser value 'testid=git-user-name-input'
control-openhands api PATCH /api/settings --write --data \
  '{"conversation_settings_diff":{"max_iterations":2}}'
control-openhands browser clock --fixed 2026-10-10T00:06:01Z
control-openhands browser click 'role=link[name="New Chat"]' --expect-url /conversations
control-openhands browser fill 'testid=chat-input' 'QA independent encrypted expiry. Do not run any tools.'
control-openhands browser click 'testid=submit-button' --expect-url '/conversations/[^?]+'
# Read the new ID from browser url, then:
control-openhands api GET /api/conversations/<id> --pick max_iterations
```

## Reviewed captures

- Main: [warm/read/refetch video](ttl-main-warm-and-refresh.mp4), [encrypted-expiry video](ttl-main-encrypted-expiry.mp4).
- PR: [warm/read/refetch video](ttl-head-warm-and-refresh.mp4), [encrypted-expiry video](ttl-head-encrypted-expiry.mp4).
- [Main display refresh](display-main-refreshed.png), [PR display refresh](display-head-refreshed.png).

Videos show real UI actions; the safe original API outputs establish the persisted
conversation limit. The clock changes and out-of-band fixture writes are stated
above rather than disguised as real elapsed time. GIFs are previews of the same
recordings. Raw logs, downloaded trajectories, storage, and session keys are not
published.

## Retry protection and other limits

The PR's live outage test stopped only its owned Agent Server, entered Application
settings, and selected a healthy second real backend during settings retries.
Its original request still made three attempts to the original origin, including
two after the new backend's successful settings read. See the actual
[request timeline](head-pinned-retry-network.json) and [video](head-pinned-retry.mp4).
The deterministic pinned-retry regression test fails against main and passes with
this implementation. Baseline UI timing did not reproduce a redirected retry in
the live attempts, so this remains defensive retry protection, not a claimed
live before/after retry defect.

Returning A → B → A while leaving the Application form mounted retains the prior
form's input value on both main and the PR. Navigating to Secrets and back remounts
the form and shows the correct backend data. This is an existing form-state
limitation, not evidence that the PR cache mixes backend settings.

Cloud/account flows, live LLM completion and other agent execution were outside
this settings-cache check. Existing focused and full CI tests cover the shared
service and its consumers. All owned settings stacks were stopped and their ports
confirmed closed; private state was purged while these artifacts were retained.

Verification performed by an AI agent for Engel Nyst (@enyst).
