# F30 — Insider Cat and its travelling Voice companion

Insider Cat is an installed App with a Projects page, a saved OpenHands agent
conversation, and Voice controls that follow that conversation around Canvas.
The App owns its page, controller selection and audio; Canvas owns the host
capabilities, navigation, identity badges and context-change notification. Agent
Server owns profile launch, tools, skills and the conversation-bound realtime
handoff. A working microphone or a spoken answer alone does not prove delegation.

Source: `src/components/features/canvas-extensions/`, `src/routes/canvas-extension-page.tsx`, `src/types/canvas-extension.ts`, `src/utils/insider-cat.ts`, `src/utils/insider-message.ts`, `src/components/features/conversation/`, `src/components/features/conversation-panel/`, `src/hooks/chat/use-slash-command.ts`, `src/hooks/mutation/use-new-conversation-command.ts`, `src/hooks/mutation/use-condense-conversation.ts`, `src/hooks/use-compact-context-action.ts`, `src/services/conversation-context-events.ts`, `src/api/agent-server-adapter.ts`.

## Sub-features

- `F30.entry`: the enabled App contributes a Cat icon and a Projects entry; its page and nested conversation URLs load inside the selected backend.
- `F30.create`: the first accepted typed request creates exactly one top-level Cat with its own ID, configured workspace and durable user event; opening New Cat alone creates nothing.
- `F30.profile-capabilities`: a new Cat inherits the active OpenHands agent profile's tools and enabled skills, Canvas confirmation/security settings, plus Insider instructions and advertised runtime context; changing an LLM does not supply missing tools.
- `F30.selected-work`: selecting work preserves the draft and supplies that worker's full identity as data to the next typed Cat request without messaging the worker.
- `F30.work-filter`: title/ID and workspace filters apply to loaded work; coverage and Load more distinguish a partial list from the whole backend.
- `F30.resume`: old saved Cats can be selected and resumed with their original ID and history; remembered selections are scoped to backend and organization and revalidated.
- `F30.invalid-controller`: missing, worker or child IDs cannot become the active Cat or receive a Voice call; legacy controller identity must not produce a misleading badge-to-page dead end.
- `F30.continuity`: messages sent from Projects and the regular conversation view share one saved history; page navigation preserves an unsent App draft while the App stays active.
- `F30.badges`: Cat identity is visible in the normal conversation header and conversation list/panel, with a link back to that Cat; worker and child conversations remain distinct.
- `F30.new-command`: New Cat and `/new` prepare a fresh Cat through the App, including `/new` from regular chat; neither sends the command to the old Cat nor silently creates an ordinary worker.
- `F30.condense-app`: App `/condense` compacts the chosen saved conversation without changing its ID or sending an ordinary agent turn; running/approval states explain the refusal and preserve the command draft.
- `F30.condense-host`: regular-view compaction ends that Cat's current Voice call before requesting compaction; a failed compaction does not silently retain a call with stale context.
- `F30.companion`: leaving Projects exposes one compact companion in the shell, with the same Cat and call; returning restores page controls without overlapping duplicate controls.
- `F30.voice-availability`: Talk explains missing selection or unavailable Voice, checks availability before acquiring a microphone, and reports a useful setup or retry action without disabling typed chat.
- `F30.voice-start`: Talk establishes a conversation-bound call, displays connecting/listening state and releases resources after a denied permission or failed connection.
- `F30.voice-handoff`: a spoken task reaches the saved OpenHands Cat exactly once, runs its real tools and returns the result through Voice; failures are visible and an unrelated or stale result is never presented as success.
- `F30.voice-mute`: Mute toggles microphone input independently of output playback, saved agent work and ending the call.
- `F30.voice-output`: Stop speaking suppresses audible output; Codex stays suppressed until Resume audio, while the OpenAI API transport cancels the current response; neither silently ends the call or mutes input.
- `F30.voice-end`: End call releases microphone, peer connection and provider call while preserving the saved conversation and already accepted agent work.
- `F30.voice-cleanup`: choosing another Cat, New Cat, changing backend, disabling or updating the App disposes the old call; late setup/results cannot revive or cross into the new selection.
- `F30.run-attention`: the Cat rests, wakes, listens, speaks and works with text status alongside its pose; approval, paused, error and unknown states remain visible and actionable.
- `F30.phone-motion`: the page and companion remain contained and usable on phone/tablet, keyboard focus is visible, and reduced motion removes animation without hiding state.

## How to get to it (user POV)

- Install and enable the reviewed [Insider App](https://github.com/enyst/insider)
  through **Customize → Apps** (`F20.install`, `F20.enable-confirm`). Generic
  installation, trust, update and uninstall are owned by F20, not this family.
- Open its Cat rail entry (`sidebar-canvas-extension-insider-cat-projects`) or
  `/extensions/insider-cat/projects`. The registered title can appear as
  **Insider Cat Projects** when the backend drops `nav_label` (F20's rail-label
  issue); do not require a different label to find the entry.
- Open a saved Cat from the App's conversation picker, the regular conversation
  header's **Open Insider Cat** badge, or
  `/extensions/insider-cat/projects/conversations/<cat-id>`.
- **New Cat**, App `/new`, regular Cat chat `/new`, and
  `/extensions/insider-cat/projects/new` reach the same fresh-Cat flow.
- **Full conversation** opens `/conversations/<cat-id>`; normal list/panel entry
  and the phone `/conversations/<cat-id>/panel` route remain available (F04/F07).
- **Talk**, **Mute**, **Stop speaking**/**Resume audio**, and **End call** are on
  Projects; away from Projects the companion supplies the same call controls.

## Driving it with control-openhands

Preconditions:

- **Draft recipes, not yet live-proven.** Record each actual result in the run's
  evidence ledger; source review and map validation are not passes. Unexecuted
  branches stay `not-run`, and an observed defect keeps its expected result.
- Use one owned, doctored local stack with this Canvas build and a compatible
  Agent Server. Set `OH_VERIFY_RUN` to that run explicitly and run
  `control-openhands doctor`. Save `control-openhands status` and
  `control-openhands api GET /server_info` as runtime evidence. No Cloud support
  is implied by successful local checks.
- Review the App source and record its exact revision before installing its
  prepared bundle with F20. A local source path refers to the Agent Server's
  filesystem, not the browser's. Record
  `control-openhands api GET /api/canvas-extensions/installed`. App source is
  maintained in `enyst/insider`; it is not a bundled Canvas fixture.
- Configure an authorized real LLM and an active OpenHands agent profile through
  F10/F13. Record `control-openhands api GET /api/agent-profiles` and the profile's
  tools, enabled skills and confirmation policy. The profile for capability
  checks must actually supply Terminal, the file editor, default enabled skills
  and `openhands-api`; a model-only profile is not an equivalent precondition.
  Keep secret values out of recorded prompts and artifacts.
- Run `control-openhands fixture git-repo --name qa-f30`, then
  `control-openhands conversation start --workspace qa-f30 --prompt "Reply QA_F30_WORKER. Do not run tools or change files." --wait`.
  Record its `<worker-id>` and full `<workspace>` from
  `control-openhands conversation status <worker-id>`. This owned worker supplies
  a workspace option and is retained until all recipes finish. Record its event
  count before selected-work checks.
- First run with no Cats in this owned backend. The create recipe supplies
  `<cat-a>`; new-command later supplies `<cat-b>`. Substitute these printed IDs
  literally. Do not infer identity from a generated title or truncated label.
- Voice-positive branches require the backend's configured provider to be
  available and a granted, usable microphone plus audio output in a secure
  browser context. API-key and Codex-auth transports are separate runs, not
  interchangeable coverage. `browser media --voice` observes media; it does not
  generate speech, grant microphone permission or prove audibility. If a fixture
  supplies audio, label that evidence synthetic and retain a separate physical
  microphone/listening check. Never turn an intercepted response into a live
  provider pass. Availability/error UI can be tested without successful audio.
- Before each Voice recipe, return to `<cat-a>` and settle any earlier agent
  turn. Start a fresh call only when the prior recipe ended it. Right before a
  running/approval-state check, send its bounded prompt, not at initial setup.
- Backend-isolation and cleanup branches need a second owned stack configured
  through F25, and approval checks need a profile with confirmation enabled.
  Optional legacy/invalid-ID fixtures must be disposable and record their exact
  tags/parent relationship from the real API before use. They are arrange state,
  not proof that the App can create those types.
- Desktop viewport and English labels unless specified. The App exposes
  `data-action`/`data-role` selectors, not Canvas-style test IDs. Scope them to
  `.insider-app` or `.insider-voice` because both views can exist in the DOM.

- **Entry and resting state (`F30.entry`).** Run
  `control-openhands browser click 'testid=sidebar-canvas-extension-insider-cat-projects' --expect-url '/extensions/insider-cat/projects(\?|$)'`,
  `control-openhands browser snapshot '.insider-app'`,
  `control-openhands browser attr '.insider-app .insider-cat-avatar' data-pose`
  and `control-openhands browser screenshot --feature F30.entry --name projects`.
  The page shows the SVG Cat, one Talk button, conversation controls, composer
  and Your work. With no Cat running, its pose is `sleeping`. Read the rail
  element's snapshot and SVG to verify its Cat icon. Repeat via
  `control-openhands browser goto /extensions/insider-cat/projects`; the same
  page loads on the same backend. No conversation is created by either visit.

- **First saved Cat (`F30.create`).** Record
  `control-openhands api GET /api/conversations/count`. Click
  `control-openhands browser click '.insider-app [data-action="new-cat"]'`
  and read the count again: unchanged. Select
  `control-openhands browser select '.insider-app [data-action="workspace"]' '<workspace>'`,
  fill `control-openhands browser fill '.insider-app [data-action="draft"]' 'Reply QA_F30_CAT_A. Do not run tools or change files.'`,
  then `control-openhands browser click '.insider-app [data-action="send"]'`.
  `control-openhands browser wait-url '/projects/conversations/'` and
  `control-openhands browser url` print `<cat-a>`. Wait with
  `control-openhands conversation wait <cat-a> --timeout 180`; inspect
  `control-openhands api GET /api/conversations/<cat-a>` and
  `control-openhands conversation events <cat-a> --kinds MessageEvent --last 10`.
  There is one new top-level conversation tagged `smolpaws: insider` and
  `insiderrole: controller`, the chosen workspace, and exactly one user request.
  Reload and inspect the App history: the accepted request remains, the composer
  is clear, and there is no second Cat or duplicate user event.

- **Real tools, skills and backend access (`F30.profile-capabilities`).** Open
  `control-openhands browser click '.insider-app [data-action="open-controller"]' --expect-url '/conversations/'`.
  Use `F07.agent-tools-modal` and `F07.skills-modal` to inspect the actual system
  message, configured tools and loaded skills, not the model's self-report.
  Run `control-openhands conversation events <cat-a> --grep INSIDER_CONTROLLER --from-start`
  and `control-openhands conversation events <cat-a> --grep RUNTIME_SERVICES --from-start`;
  compare addresses/credential-reference names with `/server_info` privately.
  Send through the real composer:
  `control-openhands browser fill 'testid=chat-input' 'Use your loaded openhands-api skill and Terminal to read the conversation count on the owning Agent Server. Make only that read-only API query; never print authentication values. Report the count and which API you used. Do not inspect other backends or change files.'`,
  then `control-openhands browser press Enter --selector 'testid=chat-input'`.
  After `control-openhands conversation wait <cat-a> --fresh --timeout 180`, read
  `control-openhands conversation events <cat-a> --kinds ActionEvent,ObservationEvent --last 20`
  and independently `control-openhands api GET /api/conversations/count`.
  Require a real tool action/observation and matching scoped count, not a prose
  claim or a count of loaded cards. Read the saved agent configuration and policy
  against the active profile; the stored profile itself must remain unchanged.

- **Selected worker and display (`F30.selected-work`).** Return with
  `control-openhands browser goto /extensions/insider-cat/projects/conversations/<cat-a>`.
  Fill `control-openhands browser fill '.insider-app [data-action="draft"]' 'QA_F30_SELECTED: acknowledge the selected work only. Do not message it, run tools or change files.'`
  and `control-openhands browser fill '.insider-app [data-action="search"]' '<worker-id>'`.
  Click `control-openhands browser click '.insider-app [data-action="select-worker"]'`;
  `control-openhands browser value '.insider-app [data-action="draft"]'` stays
  unchanged. Read `control-openhands browser attr '.insider-app [data-role="target"] span' title`:
  the full worker ID. Send, wait for `<cat-a>` to finish, and read both Cats' and
  worker's events. Only `<cat-a>` gains a user request containing the selected
  worker's full ID, title, workspace and status as data. In the App history and
  regular chat, the displayed user text omits the transport envelope; arbitrary
  user-authored JSON is not stripped. Clear selection with
  `control-openhands browser click '.insider-app [data-role="target"] button'`.

- **Work filters and coverage (`F30.work-filter`).** With `<worker-id>` still in
  search, `control-openhands browser count '.insider-app .cat-card'` is `1`.
  Fill the search with `QA_F30_NO_MATCH` and the count is `0`, with readable empty
  text. Clear it, select `<workspace>` using
  `control-openhands browser select '.insider-app [data-action="project"]' '<workspace>'`,
  and read `control-openhands browser text '.insider-app [data-role="coverage"]'`.
  With more than one page of owned fixtures, click the visible
  `.insider-app [data-action="load-more"]` until coverage says all pages loaded;
  verify no duplicate IDs against paginated API results. Without that fixture,
  record pagination `not-run`, not passed. Refresh with
  `control-openhands browser click '.insider-app [data-action="refresh"]'` and
  compare the filtered cards and coverage. Reset both filters (project value
  `''`); the Cat controllers are not ordinary Your work cards.

- **Same conversation in both views (`F30.continuity`).** On `<cat-a>` Projects,
  fill the draft with `QA_F30_UNSENT`, open Full conversation and use its composer
  to send `QA_F30_REGULAR: acknowledge only; do not run tools or change files.`
  Wait for the saved turn, then click
  `control-openhands browser click 'role=link[name="Open Insider Cat"][exact]' --expect-url '/projects/conversations/<cat-a>'`.
  The picker still selects `<cat-a>`, its history includes the regular-view
  request, and `browser value` of the App draft remains `QA_F30_UNSENT`.
  The saved user events contain no `QA_F30_UNSENT`. Clear this draft before the
  next recipe. Reload: accepted history remains; unsent drafts are not promised
  across reloads, disablement or backend changes.

- **Badges in regular views (`F30.badges`).** Open Full conversation and run
  `control-openhands browser snapshot 'testid=chat-pane-header'` and
  `control-openhands browser count 'testid=chat-pane-header >> testid=insider-cat-badge'`:
  one labelled Cat badge links to its App URL. Inspect its row in the normal
  conversation panel (`F04`), expanded and compact, then inspect `<worker-id>`:
  Cat styling belongs to the controller, not every conversation in its workspace.
  Use `control-openhands browser viewport phone`, visit
  `/conversations/<cat-a>/panel`, and capture
  `control-openhands browser screenshot --feature F30.badges --name phone-panel`.
  The identity remains distinguishable alongside normal status/archived styling.
  Return to desktop and `<cat-a>` Projects.

- **New Cat from both composers (`F30.new-command`).** Save the current count
  and `<cat-a>` user events. In App draft enter `/new` and click Send: the new-Cat
  view opens, count and old events stay unchanged, and no ordinary worker is
  created. Select `<workspace>` and send
  `QA_F30_CAT_B: acknowledge only; do not run tools or change files.` Record the
  new URL as `<cat-b>`, wait, and read its API record: new ID, controller tags,
  same configured profile capabilities. Open `<cat-a>` in regular chat, fill
  `testid=chat-input` with `/new`, press Escape to close suggestions, then Enter
  with that selector. Wait for `/extensions/insider-cat/projects/new`; again no
  new conversation is created until the first send. Return to `<cat-a>` without
  sending. The old Cat is still resumable. Also inspect the regular slash menu:
  `/new` is offered for a local Cat even though ordinary local chats differ.

- **Resume and scoped selection (`F30.resume`).** On Projects use
  `control-openhands browser select '.insider-app [data-action="controller"]' '<cat-b>'`.
  Wait for its nested URL, reload, and read the select value and recent history:
  `<cat-b>` remains selected, without duplicating it or replacing its configuration.
  Select `<cat-a>` and repeat. A genuinely fresh browser selection with multiple
  Cats must require an explicit choice; use `browser reset`/onboarding only in
  the owned run, then reopen Projects. With a second owned backend, use F25 to
  switch away/back and verify each selected ID against that backend's API.
  Browser storage alone does not prove scoping or controller validity.

- **Invalid and legacy identity (`F30.invalid-controller`).** First visit
  `/extensions/insider-cat/projects/conversations/<worker-id>` and inspect the
  status, disabled Send and Talk, and request log: the App must not turn that
  ordinary worker into a Cat. Repeat with a missing owned ID and a tagged child
  fixture. No create or Voice POST is allowed. For a legacy top-level Cat with
  only `smolpaws: insider`, open its regular-view Cat badge: either a supported
  migration/resume or an explicit recovery path is required; silently showing a
  resumable badge that only leads to a dead end is a failure. Do not broaden the
  controller rule to accept children or arbitrary workers to make this pass.
  Return to the valid `<cat-a>` URL and verify its controls recover.

- **Compaction from the App (`F30.condense-app`).** Settle `<cat-a>`, note its
  message-event count and ID, fill its App draft with `/condense`, clear the
  network log, then click Send. Inspect
  `control-openhands browser network --filter 'condense|voice/realtime' --bodies`,
  the App notice, and
  `control-openhands conversation events <cat-a> --kinds CondensationRequest,Condensation --last 5`.
  The ID remains the same, accepted compaction clears the draft, and no user
  MessageEvent containing `/condense` or extra agent turn is submitted. With an
  active call, it ends before compaction; Talk afterwards starts from current
  saved context. During bounded running/approval work, repeat: a clear refusal
  preserves `/condense` and sends no compaction request. An error response keeps
  an actionable notice rather than reporting success. Clear any retained draft.

- **Compaction from regular Canvas (`F30.condense-host`).** With a live call on
  `<cat-a>`, use Full conversation (do not reload) and F05's context-window
  **Compact context** action, or regular `/condense` when offered. Clear the
  network log before the action and inspect its `condense|voice/realtime` requests,
  companion controls and `browser media` afterwards. The selected Cat's call
  ends on the context-change request, even when compaction fails. An unrelated
  conversation's compaction must not end `<cat-a>`'s call. This is a host
  notification check; App-only `/condense` does not cover it.

- **Availability without audio (`F30.voice-availability`).** In the new-Cat
  view, `browser enabled` on `.insider-app [data-action="voice-start"]` is false
  with a short selection instruction. Select `<cat-a>`, inspect
  `control-openhands api GET /api/conversations/<cat-a>/voice`, clear network and
  media logs, then click Talk. On an owned run with unavailable provider setup,
  the availability GET precedes any realtime POST; no call or microphone is
  acquired, readable reason appears, and typed chat still works. Missing API key
  offers Settings → Secrets; missing Codex sign-in/install names that remedy
  without dumping provider errors or trying the other provider automatically.
  Capture `control-openhands browser screenshot --feature F30.voice-availability --name unavailable`.
  Test absent host companion support separately if a supported older host is
  available; disabling Voice must not disable typed Cat chat.

- **Start a call (`F30.voice-start`).** With confirmed availability and microphone
  permission, select `<cat-a>`, clear `browser media`/`browser network`, and click
  `.insider-app [data-action="voice-start"]`. Observe waking/connecting, then
  listening text and peaked ears; End call and Mute become usable. Inspect
  `browser network --filter 'voice'` and `browser media --voice`; save a screenshot.
  A successful SDP response alone is insufficient: require a connected peer,
  live enabled microphone track, and a usable call. In a separate denied-permission
  drive, require a capture request with `status: error`,
  `errorName: NotAllowedError` and no granted audio tracks, explanatory text,
  no peer or realtime POST, and a working typed fallback in the same Cat.
  The configured Chromium headless shell supports native automatic denial:
  end any call, record the original browser flags, then run
  `control-openhands browser stop --run "$OH_VERIFY_RUN"` followed by
  `CONTROL_OPENHANDS_BROWSER_ARGS='--use-fake-device-for-media-stream --use-fake-ui-for-media-stream=deny' control-openhands browser start --run "$OH_VERIFY_RUN"`.
  Reopen `<cat-a>` Projects, clear the network log, click Talk, then inspect
  `control-openhands browser media --voice` and
  `control-openhands browser network --filter 'voice/realtime'`.
  Replace the normal fake-UI allow flag; do not combine it with `=deny` or
  `--auto-accept-camera-and-microphone-capture`. This is Chromium's native
  automatic media denial, without application or network interception; it is
  not a physical microphone or human permission-prompt click. Its Permissions
  API state may remain `prompt`, so require the actual `NotAllowedError` instead
  of assuming `voice.microphonePermission` becomes `denied`. Merely removing
  fake UI from this headless shell can produce `NotSupportedError`, which is a
  different test. Send `Reply TEXT_STILL_WORKS only.` through the App's draft
  and Send controls, and require the saved request and displayed reply.
  Stop the owned browser, restore its original flags, start it again and require
  a successful call retry. Record the denial and recovery separately.

- **Travelling call (`F30.companion`).** While the call is live, open Full
  conversation by click. `control-openhands browser visible '.insider-voice'`
  is true; its title belongs to `<cat-a>`, and `.insider-voice [data-action="end"]`
  operates the same call. There is no second realtime POST merely from navigation.
  Click the Cat badge back to Projects: the companion is hidden and the page's
  controls are active. Repeat via an unrelated normal conversation and the phone
  panel route. Inspect `browser media` across the transitions and capture
  `control-openhands browser screenshot --feature F30.companion --name regular-view`.
  The dock takes layout space and must not cover the composer or leave a blank
  duplicate control block on Projects.

- **Saved agent handoff (`F30.voice-handoff`).** Record `<cat-a>`'s newest event
  and speak a unique bounded task: “QA_F30_VOICE: use Terminal to run
  `printf QA_F30_TOOL_RESULT` only, then tell me its result.” Read
  `control-openhands conversation events <cat-a> --kinds MessageEvent,ActionEvent,ObservationEvent --last 30`
  and `control-openhands browser media --voice`. Require exactly one saved user request,
  the named terminal action and matching observation, a settled agent answer
  newer than the watermark, and corresponding returned speech. The real saved
  Cat, not a tool-free realtime/Codex reasoning session, performs the action.
  Do not infer success from an acknowledgment or transcript alone. Codex browser
  traffic must not duplicate its server-dispatched handoff; the API transport's
  `send_to_insider` must target the same ID. Repeat with a known bounded tool
  error: the Voice result reports failure and does not reuse the prior answer.
  Check a continuation referring to the saved result after ending/restarting
  Voice; context must come from the same saved Cat. Record each provider/input
  method separately; absent real speech input is a missing prerequisite. Take
  two `--voice` samples while output is active: matching `voice.peerConnections`
  IDs must show increasing inbound-audio counters where supported. Nonzero or
  increasing audio energy supports decoded audio, not physical speaker audibility.

  **Correction during a pending task.** Use a fresh call on a settled controller
  with Terminal enabled and no pending approval. Say: “Use Terminal to run only
  `sleep 20; printf 'OLD SEVENTEEN\n'`. When it finishes, tell me the result.”
  Five seconds after that utterance ends, say: “Please do not say the old result;
  tell me NEW TWENTY NINE instead.” Save the spoken-input timing and verify in
  the saved events that the first terminal action was still pending when the
  correction arrived. If it was not pending, repeat the timing probe rather
  than treating it as a pass. Recorded microphone input may supply these same
  utterances and gap through Chromium's existing audio-fixture flags; label
  synthetic input and the real provider response separately.

  Inspect the controller's MessageEvent, ActionEvent and ObservationEvent rows
  and the returned Voice transcript/audio through completion and for at least
  60 seconds afterward. The accepted first request and its OLD SEVENTEEN tool
  result remain durable; that superseded answer must not be spoken. A busy
  correction must be described as **not sent**, with a clear instruction to
  repeat it after the current task finishes. Voice must not claim it accepted,
  forwarded or queued the correction; confirm that no second saved request
  exists for that unaccepted correction. This is not cancelling the running
  tool. After it settles, say: “Use Terminal to calculate 2 + 3 and tell me the
  result.” Require one fresh saved request, the real tool observation and spoken
  answer 5, proving that suppression did not disable the next uninterrupted
  task. Record this branch separately from a normal handoff; transcript-only
  proof does not establish physical speaker silence or all audio barge-in races.

- **Mute input (`F30.voice-mute`).** During that call click
  `control-openhands browser click '.insider-app [data-action="mute"]'`;
  `control-openhands browser attr '.insider-app [data-action="mute"]' aria-pressed`
  is `true`, its label becomes Unmute, and status explains muted input. Verify
  spoken input during the muted interval creates no new request; existing output
  can continue. `browser media --voice` shows the captured microphone track's
  `enabled: false` while its `readyState` remains `live`. Click again:
  `aria-pressed` is `false`, the track is enabled, the same call listens and
  accepts a new bounded request. Neither click sends a new call setup or end.

- **Stop output (`F30.voice-output`).** During real output click
  `control-openhands browser click '.insider-app [data-action="interrupt"]'`.
  With Codex the label becomes Resume audio and
  `control-openhands browser media --voice` shows that call's output element
  muted/paused in `voice.audioElements` (including detached elements); its
  microphone-track enabled state remains unchanged. Navigate to regular
  chat, allow status polling and late transcript updates, and require suppression
  to persist. Click `.insider-voice [data-action="interrupt"]` to resume the
  current stream without creating a new request/call. With OpenAI API, the
  current response is cancelled instead; do not expect a Codex Resume control.
  Media state proves browser suppression, not what a physical speaker emitted.

- **End the call (`F30.voice-end`).** Click the visible view's
  `[data-action="end"]`, then inspect `browser network --filter 'voice/realtime'`,
  `browser media --voice`, its status and Talk button. There is one scoped hangup;
  that call's audio element has `hasSrcObject: false`, its captured microphone
  tracks have `readyState: ended`, its peer is `closed`, and controls return to
  idle. A garbage-collected object with only a stale live state does not prove
  cleanup. The
  saved Cat ID/history remain. Reopen Projects, reload and verify saved events.
  Repeat while a previously accepted bounded tool task is running: ending Voice
  must not pause/delete that OpenHands task or duplicate it on reconnection.

- **Disposal and late completion (`F30.voice-cleanup`).** Start a fresh call on
  `<cat-a>` immediately before each separate transition: choose `<cat-b>`, click
  New Cat, switch to the second owned backend through F25, disable the App through
  F20, then update the reviewed App through F20. After each, require old media and
  call disposal, no stale transcript/answer in the new view and no automatic call
  on the new selection. Restore enabled App/backend/`<cat-a>` between branches.
  Also click End while setup is connecting: any late returned call ID is hung up
  once, never revived. A call-state race needs its own evidence; a normal idle
  disable is not coverage of pending setup or a handoff in flight.

- **Work and attention states (`F30.run-attention`).** Send the bounded typed
  prompt `Run only: sleep 15. Then reply QA_F30_DONE.` in `<cat-a>` and observe
  `data-pose=working` with text status during the run; when it settles, inactive
  Voice returns to sleeping. Repeat on an owned confirmation-enabled Cat:
  waiting-for-confirmation must show an attention pose and approval text, and
  Full conversation reaches the real approval controls (F06). Pause through the
  regular UI and require a paused label, not success. For an actual backend or
  provider error, preserve the error text/status and recovery controls. A run
  ending is not itself proof that the requested task succeeded.

- **Phone, keyboard and reduced motion (`F30.phone-motion`).** On `<cat-a>`
  Projects run `control-openhands browser viewport phone`,
  `control-openhands browser bbox '.insider-app'`,
  `control-openhands browser screenshot --feature F30.phone-motion --name page-phone`.
  The controls fit with no horizontal overflow or hidden Talk button. Tab through
  picker, New Cat, voice actions and composer: focus remains visible. Repeat at
  `browser viewport tablet`. On a live call navigate to regular chat and check
  `.insider-voice` and the composer bounds: the dock must leave usable content.
  Run `control-openhands browser media --reduced-motion reduce`; poses/status and
  controls remain readable without breathing/listening animations. Restore
  `--reduced-motion no-preference`, desktop viewport and end the owned call.

## Gotchas

- These recipes require the separately installed App as well as the Canvas and
  SDK changes. Record all three revisions. A test of the bundled demo App is not
  an Insider test; an App source edit without rebuilding/updating its prepared
  bundle is not a test of that edit.
- The App's role skill is launch context, not a replacement for real profile
  tools or enabled skills. Existing Cats retain their saved configuration;
  reconnecting Voice or changing the LLM does not repair tool-free Cats.
- An empty `skills` array in conversation metadata does not establish that no
  skills are available: enabled skills can be discovered or invoked dynamically.
  Inspect the actual system prompt, skill invocation and saved tool observations.
  Conversely, listing a skill in prose does not prove it was loaded or used.
- Count the owning backend with its real API. Project-card counts, latest-event
  limits and a filesystem event-line count cannot establish its conversation total.
- Codex handoffs are dispatched on Agent Server; the browser handles media and
  call status. The API transport uses `send_to_insider`. The SDK's stateless
  `/ask_agent` operation is not a durable task runner. SDK verification owns
  relay deduplication, run settlement, stale-answer rejection and provider cleanup;
  this family requires their observable end-to-end effects in Canvas.
- A voice acknowledgment without a saved handoff is not a saved conversation
  turn. Do not claim every utterance is persisted until actual events prove it.
- App discovery and Canvas badges accept a top-level Cat with the explicit
  controller role or an absent legacy role. Children, explicit other roles and
  unmarked conversations are excluded. Verify legacy resume separately from
  worker/child rejection; a legacy text continuation or connected call alone
  does not prove a spoken response on that conversation.
- Use in-app clicks to test call continuity. `browser goto`, reload, a fresh tab
  or a full browser navigation can tear down the document and are different
  lifecycle tests; a re-created call does not prove continuity.
- The App polls recent history (50 events) and status. Wait for the fresh run,
  and consult the full saved history before retrying a send with an uncertain
  outcome. Never auto-resubmit to make a missing acknowledgment disappear.
- An accepted compaction clears `/condense`; refused or failed compaction must
  remain distinguishable. App and host compaction are separate entry points.
  The same-conversation Voice cleanup is required even if compaction fails.
- Do not publish runtime URLs, credentials, private transcripts or arbitrary
  captured network bodies. Keep fixture markers non-sensitive. Evidence must
  state whether audio was live, synthetic, intercepted or only visually observed.
- At the end run `control-openhands browser errors --app-only` and record each
  ID/entry point with expected/actual results and artifacts. End calls, restore
  settings/backend/permissions, and stop only the owned verification run after
  its evidence has been saved. Product failures stay failures; no exceptions
  are added to make a deficient run look complete.
