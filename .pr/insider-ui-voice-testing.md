# Insider App: isolated UI and voice verification

Historical verification from September 15, 2026 for the matching
`gpt/insider-ui-voice` branches of Agent Canvas, Agent Server, and the
[standalone Insider App](https://github.com/enyst/insider). This evidence concerns
the Canvas integration in this draft; provider implementations belong to the
separate App and Agent Server changes.

## Test setup

An isolated Canvas frontend, matching Agent Server branch, scripted model,
and temporary persistence directory were used. The App was installed through
the authenticated installation endpoint from its local source checkout, then
enabled. Model responses were scripted; conversation storage, execution, event
delivery, compaction, App loading, and the browser UI were real.

To reproduce with the matching branches:

1. Install Canvas dependencies with `npm ci`, then run `npm run build:app`.
2. Start the matching Agent Server and point a development Canvas frontend at it;
   use isolated persistence and a backend profile with a working model.
3. Install the standalone Insider App by its absolute checkout path in Apps,
   enable it, and send a first message from its Projects page.
4. Open the Cat in regular chat and return through its badge. Confirm the ID
   remains unchanged, the Cat is identifiable in the sidebar, and the dock
   leaves the composer accessible.
5. Use `/new` and send a message to create a distinct Cat. Resume the first one.
6. Use `/condense` while idle, confirm the ID and full saved history remain,
   then send a follow-up. A matching active voice call must end before the
   compaction request, so a subsequent call starts with fresh context.

Voice negotiation requires the matching server broker and configured provider.
The silent native transport fixture described below did not establish provider
or microphone support. No local runtime files or credentials are included.

## Verified behavior

1. A first message on the App page creates a tagged, saved Insider controller.
   Opening it in regular chat keeps the exact conversation ID. A follow-up from
   regular chat is persisted there and receives its prior context.
2. `/new` from a regular Insider opens the App's New Cat flow. Its next message
   creates a distinct ID; existing Cats remain available in the resume picker.
3. Successful creation and picker selection use a canonical App URL containing
   the selected conversation ID. Reload resumes that Cat.
4. Cat icons identify the App navigation item, regular conversation header,
   conversation cards, and compact sidebar rows. The header badge opens the
   same Cat in the App. Internal identity tags do not duplicate the badge or
   disappear when unrelated tags are edited.
5. Legacy top-level conversations with `smolpaws: insider` and no `insiderrole`
   remain resumable and recognized by the voice broker. Child conversations and
   explicit non-controller roles do not receive this identity.
6. `/condense` in the regular UI persisted a 319-character summary covering
   five forgotten event IDs. The conversation ID stayed unchanged, and the full
   event log retained both original user turns and both final answers. The
   command itself did not become a user message or execute another tool.
7. The subsequent model request included that summary and the new question,
   omitted a forgotten turn, and returned a visible answer in the same chat.
   This verifies model-context compaction, not merely an accepted HTTP request.
8. Starting voice without `OPENAI_API_KEY` reports a setup error before asking
   for microphone permission. The server accepts the legacy Cat identity here
   too; missing credentials do not masquerade as a missing conversation.

The first quick scripted test reused a mock tool-call ID across separate
trajectories. Compaction verification therefore used a fresh conversation with
one coherent trajectory and unique tool-call IDs; it did not rely on recovery
from malformed fixture history.

## Voice scope and limits

Odie provides an optional `registerCompanion` App capability: a shell-owned
surface whose mount survives page navigation. Backend or App changes dispose
the activation and surface. The request helper remains bound to the original
backend so late voice-resource cleanup cannot target a different backend.
The controls occupy a dock below the page outlet, rather than covering the
composer. The final 1280-by-720 browser check showed the composer and Send
button clear of the dock, with no page-width overflow. The same Cat remained
selected across App-to-chat navigation; a temporary unsent layout-check draft
was cleared afterwards.

The optional `onConversationContextChangeRequested` host subscription also
ends a matching voice call before regular Canvas submits a permitted compaction
request. It is scoped to backend, organization, and connection revision. It
signals intent rather than successful compaction; even a failed request may
have ended the call. Start voice again after compaction completes. Older hosts
without this capability need a manual voice restart after regular compaction.

The SDK supplies authenticated availability, WebRTC SDP setup, and scoped
hangup endpoints. It keeps the standard OpenAI API key on the server and uses
`gpt-realtime-2.1`. Initial voice context comes from the actual current,
condensed conversation view: at most 40 visible text blocks and 12,000
characters, excluding reasoning and tool internals. Spoken requests are relayed
to the saved controller; its confirmed result is spoken back. The exact audio
transcript is not a saved conversation log.

Client tests cover media startup, missing credentials, failed playback,
interruption, stale work, typed/voice submission races, same-Cat navigation,
controller changes, and call cleanup, including startup that resolves late.
Answer tests withhold a proposed Finish message while the controller is running;
approval, pause, and error states return status rather than proposed answer text.
Before speaking a final answer, the App requires the voice broker to confirm
`run_active: false` and `execution_status: finished`. The broker checks the
actual run task, including stop hooks, without waiting or cancelling it. The App
then reloads saved events to select the final reply. Older brokers missing these
fields cannot confirm completion, so the App directs the user to the saved chat
instead of announcing success.
Spoken hangup ends the call; it does not cancel the controller's work.

**The user-approved silent native browser test passed on September 15.** The
test replaced microphone acquisition with silent AudioContext tracks and used
two real browser RTCPeerConnections. Only the isolated voice API's fetch/XHR
requests were intercepted. Availability retained the real backend's
`run_active` and `execution_status`; SDP setup and hangup used the local peer.
The App's posted offer was used unchanged, including its original ICE timing.
Neither `audio.play()` nor the native peer's transport was stubbed.

Observed native results:

- Both directions transferred 947 audio RTP packets / 32,826 bytes in one
  recorded snapshot. The App's audio element had a live stream, was playing,
  and advanced to 18.717 seconds. The input was synthetic silence.
- Mute and unmute changed the same live sender track's `enabled` value. Stop
  speaking sent `response.cancel` and `output_audio_buffer.clear` across the
  native data channel, paused playback, and retained the connected peer.
  A subsequent synthetic playback-start event resumed the audio element.
- App-to-regular navigation retained the same controller, call, peer, and
  track, with one microphone-substitute acquisition and one setup request.
- Regular `/new` closed the previous peer and channel, ended its input track,
  and issued the correctly scoped hangup before opening a new Cat draft.
- On a fresh Cat, sending the same synthetic voice function call twice produced one persisted user request
  and one answer returned over the native data channel. A typed follow-up
  continued in that Cat while voice stayed connected.
- Regular `/condense` issued its scoped hangup before the compaction POST,
  closed the matching call, and retained the conversation ID. Restarting voice
  created a new call for that Cat; its next delegated request received the
  saved post-compaction answer over the data channel.
- End call, a synthetic explicit `end_voice_call`, and unexpected remote
  data-channel closure each closed the App peer/channel, ended the input
  track, cleared playback, and requested scoped cleanup. Unexpected closure
  also displayed the text-conversation fallback.

Across all five calls, every App peer and channel was closed, every input track
was ended, and every scoped hangup was observed **before** fixture cleanup.
Cleanup then closed all ten fixture AudioContexts, restored the native peer
constructor and microphone API, deleted the fixture, and cleared interception
rules. Final read-only inspection found no fixture, microphone override, or
audio stream attached to playback.

The final backend proof records one condensation with six summarized event IDs,
all retained in the full 35-event history. Both simulated voice requests were
saved once. Exactly five scripted model requests ran; the last included the
exact saved coral summary and new question, while omitting the two summarized
earlier requests. The private temporary fixtures and state are not included in
this PR; the observations above are historical reproduction notes.

**This September 15 fixture did not verify actual microphone capture, permission
UX, VAD, audible speech, or live OpenAI negotiation/account access.** No OpenAI key was available,
no real microphone audio was captured, and no provider voice request was sent.
The simulated provider used silent media and synthetic function-call events;
its successful transport does not establish speech recognition or quality.

Voice-resource ownership and broker limits are reviewed in the separate Agent
Server change. This Canvas change provides lifecycle disposal and retains each
App request helper's owning backend for asynchronous cleanup.

## Historical automated checks (September 15)

- Insider App: 43 tests, including build consistency and localization.
- Odie: 395 focused tests passed, one existing TODO; type checking, touched
  source lint/formatting, and production build passed.
- SDK: 115 focused tests and one authenticated real HTTP test passed;
  pre-commit checks passed.
- Architecture documentation: six pages and 122 local links, assets, and
  anchors validated.

The recommended interaction is one remembered default Insider, with explicit
New and Resume choices. Condense preserves continuity while reducing the
model's context; New creates an independent conversation. Requesting Condense
ends that Cat's active voice call. Start voice after compaction completes so the
next connection starts with the current summary.

## Publication checks (September 19, 2026)

The tested source branch is based on upstream commit `23ca81c9ff`; it has not
been rebased onto subsequent UI changes. The following command passed 348 tests
in 18 files, with one pre-existing TODO (Node 24.19.0):

```sh
npm test -- --maxWorkers=2 \
  __tests__/api/agent-server-conversation-service.test.ts \
  __tests__/components/chat/chat-interface.test.tsx \
  __tests__/components/features/chat/user-assistant-event-message.test.tsx \
  __tests__/components/features/conversation-panel/conversation-tag-display.test.ts \
  __tests__/components/features/conversation-panel/edit-conversation-tags-modal.test.tsx \
  __tests__/hooks/chat/use-slash-command.test.ts \
  __tests__/hooks/mutation/use-new-conversation-command.test.tsx \
  __tests__/routes/root-layout.test.tsx \
  src/components/features/canvas-extensions/canvas-extensions-runtime.test.tsx \
  src/hooks/use-compact-context-action.test.tsx \
  src/components/features/conversation/insider-cat-badge.test.tsx \
  src/utils/insider-message.test.ts \
  src/api/no-direct-agent-server-calls.test.ts \
  src/api/canvas-extensions-service.test.ts \
  src/extensions/canvas-extension-module-loader.test.ts \
  __tests__/api/agent-server-adapter.test.ts \
  src/api/agent-server-adapter.test.ts \
  __tests__/components/features/conversation/conversation-name.test.tsx
```

`npm run typecheck`, ESLint and Prettier on changed source files, and
`npm run check-translation-completeness` and `npm run build:app` also passed.
The build reports existing dependency sourcemap and chunking warnings. No new
provider or microphone session was needed for these publication checks.

## Upstream integration (September 19, 2026)

Merged upstream `a07364828c8f202e7745c6bce3dcef3915ae7ac1` into the tested
Insider branch without rewriting history. The two textual conflicts were
resolved by retaining both translation additions and following upstream's
conversation-service test split. The Insider parent/tag assertion now lives in
`agent-server-conversation-service-regressions.test.ts`; moved upstream tests
were not duplicated.

The integrated tree passed 514 focused tests across 23 files, with one existing
TODO. This includes the original Insider host/UI coverage plus conversation
service regressions, compaction, history loading, websocket routing, and
conversation status integration. Validation used the updated upstream lockfile
with TypeScript client 1.49.2 and extensions 0.22.1. The running test instance was
left unchanged during this isolated integration.

Type checking, changed-source ESLint/Prettier, translation completeness, and
`npm run build:app` also passed after the merge.
