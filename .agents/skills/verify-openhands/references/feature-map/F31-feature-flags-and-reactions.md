# F31 — Feature flags and conversation reactions

Settings → Experimental lists the features a user can turn on or off for
themselves. The agent server supplies the list (`GET /api/settings/experimental`),
including each feature's label and description, so a new feature appears here
without a Canvas change. Each switch saves at once with
`PATCH /api/settings/experimental/{key}`. Flags are read when a conversation
starts, so a change applies to conversations started after it.

The first feature is **Conversation reactions** (`conversation_reactions`, off
by default). While it is on, every new OpenHands-agent conversation gets the
agent's `react` and `unreact` tools. The agent's reactions show as emoji chips
on the person's message, and the `react` tool call itself stays out of the
chat. On the agent's messages the person can add their own reaction from a
small picker, or take it off by clicking their chip. While it is off, existing
chips still show but the picker is gone, the reactions route answers 409, and
new conversations do not get the tools. Local backend only: Cloud lists no
features.

Source: `src/routes/experimental-settings.tsx`, `src/api/experimental-features-service/experimental-features-service.api.ts`, `src/hooks/query/use-experimental-features.ts`, `src/hooks/mutation/use-update-experimental-feature.ts`, `src/components/features/chat/message-reactions.tsx`, `src/components/conversation-events/chat/event-message-components/user-assistant-event-message.tsx`, `src/components/conversation-events/chat/event-message-components/finish-event-message.tsx`, `src/components/conversation-events/chat/event-content-helpers/should-render-event.ts`, `src/constants/settings-nav.tsx`, `src/components/features/command-menu/command-menu-items.tsx`.

## Sub-features

- `F31.open`: the page opens from the settings navigation (`sidebar-settings-/settings/experimental`), the command menu ("Experimental") and `/settings/experimental`, and lists **Conversation reactions** with its description, off on a fresh run.
- `F31.toggle`: clicking the switch turns the flag on at once, without Save; `GET /api/settings/experimental` agrees, and the switch is still on after a reload. Clicking again turns it off.
- `F31.empty`: an agent server without feature flags (older than software-agent-sdk#6609) shows "This agent server has no experimental features to turn on or off." instead of an error.
- `F31.agent-reaction`: with the flag on, a new conversation's agent has `react` and `unreact`; when asked to react first, its 👀 chip appears on the person's message and stays after the answer (`testid=reaction-eyes`, `data-sources=agent`), and no "Reacted" tool card is shown.
- `F31.user-reaction`: with the flag on, the `add-reaction` button under the agent's answer opens a picker of six emoji; choosing ❤️ shows a pressed `reaction-heart` chip (`data-sources=user`) that survives a reload; clicking the chip takes the reaction off.
- `F31.off`: with the flag off, existing chips still show but are not clickable and there is no `add-reaction`; `POST /api/conversations/<id>/reactions` answers 409 `Conversation reactions are turned off`; a conversation started afterwards has no `react`/`unreact` tools.
- `F31.confirm-pending`: with Confirmation Mode on and Security Analyzer **None** (always confirm), the agent's `react` call waits for approval and shows in the chat as its own row ("Add eyes reaction to …") above **Cancel** / **Continue**, so the person sees what they approve. **Continue** records the reaction: the row goes, the 👀 chip shows. **Cancel** shows the row as **Rejected**, no chip and no `ReactionEvent`.
- `F31.narration`: text the agent writes with its `react` call (the action's thought) stays in the chat next to the chip after the call finishes and after a reload; only the tool row is dropped.
- `F31.react-while-running`: the person can react to an earlier answer while the agent runs a long command; the chip appears and the run carries on and finishes once, without a second run. The reaction is recorded when the agent's current step ends (the step holds the conversation's state lock), so the chip can take as long as the running command.
- `F31.backend-scope`: each backend keeps its own switch: turning it on on one backend leaves another backend's page showing its own value, and switching back shows the first backend's. A switch flipped while the browser is offline, followed by a backend switch and reconnect, still reaches the backend it was flipped on (that server changes, the other does not, and the page for the now active backend shows that backend's value).
- `F31.docker`: on a server with the Docker runtime, a conversation started from Home with **Docker** picked gets `react`/`unreact` while the flag is on: the agent's 👀 and the person's ❤️ work as in Local mode and survive a reload. Turning the flag off takes effect at once for the Docker conversation (no `add-reaction`, the reactions route answers 409 from the host), and a Docker conversation started afterwards has no reaction tools.

## How to get to it (user POV)

- Sidebar settings gear (`backend-selector-settings-link`), then **Experimental** in the settings navigation (`sidebar-settings-/settings/experimental`).
- Direct URL `/settings/experimental`.
- Command menu (`Control+k` or `Meta+k`): search "Experimental".
- Reactions: in any conversation started while the flag is on (`/conversations/<id>`).

## Driving it with control-openhands

Preconditions:

- Baseline state on an Agent Server that has feature flags (software-agent-sdk#6609 or later): `control-openhands launch --sdk-path <software-agent-sdk checkout>` until it is released. `control-openhands api GET /api/settings/experimental` lists `conversation_reactions`.
- `F31.agent-reaction` and later need an active LLM profile (`control-openhands llm preset deepseek`).
- Use the Paper theme for screenshots (`F16.paper-theme`).

- **Open and turn on (`F31.open`, `F31.toggle`).**
  ```sh
  control-openhands browser goto /
  control-openhands browser press Control+k
  control-openhands browser type 'testid=command-menu >> role=combobox' Experimental
  control-openhands browser press Enter
  control-openhands browser url                                   # ends in /settings/experimental
  control-openhands browser snapshot 'testid=experimental-settings-screen'   # checkbox "Conversation reactions" + description
  control-openhands browser screenshot --feature F31.open --name off
  control-openhands browser click 'testid=experimental-settings-screen >> text=Conversation reactions'
  control-openhands api GET /api/settings/experimental                # enabled: true
  control-openhands browser reload
  control-openhands browser eval "document.querySelector('[data-testid=experimental-switch-conversation_reactions]').checked"   # true
  ```
  The label is clicked, not the hidden checkbox: the switch's visual covers it.
- **The agent reacts (`F31.agent-reaction`).**
  ```sh
  control-openhands browser goto /
  control-openhands conversation start --prompt "Please react to this message with the eyes emoji first, then tell me in one short sentence what 17 times 23 is." --wait --timeout 300
  control-openhands api GET /api/conversations/<id>               # agent.tools names include react, unreact
  control-openhands browser goto /conversations/<id>
  control-openhands browser wait 'testid=reaction-eyes' --timeout 20000
  control-openhands browser eval "document.querySelector('[data-testid=reaction-eyes]').dataset.sources"   # agent
  control-openhands browser count 'text=Reacted :eyes:'           # 0
  control-openhands browser screenshot --feature F31.agent-reaction --name eyes
  ```
  The agent's reaction stays on after its answer: the events hold one agent `ReactionEvent` for `eyes` with `removed: false`. A chip that is gone after the answer means the agent took it off, which the `react` tool guidance no longer asks for.
- **The person reacts (`F31.user-reaction`).**
  ```sh
  control-openhands browser click 'testid=add-reaction'
  control-openhands browser screenshot --feature F31.user-reaction --name picker
  control-openhands browser click 'testid=reaction-picker-heart'
  control-openhands browser wait 'testid=reaction-heart' --timeout 15000
  control-openhands browser reload
  control-openhands browser wait 'testid=reaction-heart' --timeout 20000
  control-openhands browser eval "(e=>e.dataset.sources+' '+e.getAttribute('aria-pressed'))(document.querySelector('[data-testid=reaction-heart]'))"   # user true
  ```
  Clicking `testid=reaction-heart` again takes it off (`browser count 'testid=reaction-heart'` is `0` once the event arrives).
- **Turned off (`F31.off`).**
  ```sh
  control-openhands browser goto /settings/experimental
  control-openhands browser click 'testid=experimental-settings-screen >> text=Conversation reactions'
  control-openhands api POST /api/conversations/<id>/reactions --write --data '{"emoji":"+1"}'   # status 409
  control-openhands browser goto /conversations/<id>
  control-openhands browser count 'testid=add-reaction'           # 0; existing chips still show
  control-openhands browser goto /
  control-openhands conversation start --prompt "Please react to this message with the eyes emoji first, then say hi." --wait --timeout 300
  control-openhands api GET /api/conversations/<new-id>           # no react / unreact in agent.tools
  ```

- **Always confirm (`F31.confirm-pending`).**
  Arrange in Settings → Verification: click the **Confirmation Mode** label, open all sections (`testid=sdk-section-all-toggle`), choose **None** in the `Security Analyzer` combobox, click `testid=save-button` (`api GET /api/settings` shows `conversation_settings.confirmation_mode` `true`, `security_analyzer` `none`).
  ```sh
  control-openhands browser goto /
  control-openhands conversation start --prompt "QA_F31_CONFIRM Put a 👀 reaction on this message, then tell me what 3+3 is in one line." --wait --until waiting_for_confirmation --timeout 180 --stay
  control-openhands browser text 'testid=chat-interface'          # "Add eyes reaction to the user's message", "Do you want to continue with this action?"
  control-openhands browser screenshot --feature F31.confirm-pending --name pending
  control-openhands browser click 'testid=action-reject-button'   # Cancel
  control-openhands conversation wait <id> --until idle,finished --timeout 120
  control-openhands browser text 'testid=chat-interface'          # the row now reads "Rejected"
  control-openhands browser count 'testid=reaction-eyes'          # 0
  ```
  Repeat with a new conversation and `testid=action-confirm-button` (Continue): the conversation finishes, the row is gone and `testid=reaction-eyes` is `1`. Turn Confirmation Mode off again afterwards.
- **Narration stays (`F31.narration`).**
  ```sh
  control-openhands browser goto /
  control-openhands conversation start --prompt "In your first step, write the sentence 'Looking into it now.' as text AND call the react tool with eyes in that same response. Then tell me what 2+2 is in one line." --wait --timeout 180 --stay
  control-openhands api GET '/api/conversations/<id>/events/search'   # the react ActionEvent's thought holds "Looking into it now."
  control-openhands browser wait 'testid=reaction-eyes'
  control-openhands browser reload
  control-openhands browser text 'testid=chat-interface'          # "Looking into it now." above the answer; no "Reacted :eyes:" row
  control-openhands browser screenshot --feature F31.narration --name kept
  ```
  If the model puts no text in the call's thought, run it again: the check needs a non-empty thought.
- **React while the agent runs (`F31.react-while-running`).**
  ```sh
  control-openhands browser goto /
  control-openhands conversation start --prompt "QA_F31_RUNNING Reply with the single word hello." --wait --timeout 120 --stay
  control-openhands conversation send <id> --prompt "Run exactly this terminal command: sleep 30 && echo qa-slept . Then reply with its output."
  control-openhands conversation events <id> --last 2             # repeat until the terminal ActionEvent is there
  control-openhands browser click 'testid=add-reaction'
  control-openhands browser click 'testid=reaction-picker-heart'
  control-openhands conversation status <id>                      # running
  control-openhands browser wait 'testid=reaction-heart' --timeout 45000
  control-openhands conversation wait <id> --timeout 120          # finished
  control-openhands conversation events <id> --last 8             # terminal action, its observation, the user ReactionEvent, one agent answer
  ```
  Time the click to the chip: it is about as long as the command's remaining time, because the reaction waits for the step that holds the state lock. Record the time; a chip that never comes, an error toast, or a second agent run is a failure.
- **Per backend (`F31.backend-scope`).**
  Needs the second stack from F25's Preconditions added as `QA_Second` (F25 Add bullet), started from the same SDK checkout so it has feature flags. Arrange: on Local the flag is on; on QA_Second turn it on in the UI too.
  ```sh
  control-openhands browser goto /settings/experimental              # on Local: checked
  control-openhands browser network-state offline
  control-openhands browser click 'testid=experimental-settings-screen >> text=Conversation reactions'   # queued: the server has not changed
  control-openhands api GET /api/settings/experimental              # still enabled: true
  control-openhands browser click 'testid=backend-selector'
  control-openhands browser click 'testid=backend-selector >> role=option[name="Connected QA_Second"]'
  control-openhands browser network-state online
  control-openhands api GET /api/settings/experimental              # Local: enabled false (the queued switch landed here)
  control-openhands api GET /api/settings/experimental --run "$OH_VERIFY_RUN_2"   # QA_Second: still enabled true
  control-openhands browser goto /settings/experimental
  control-openhands browser eval "document.querySelector('[data-testid=experimental-switch-conversation_reactions]').checked"   # true (QA_Second's)
  ```
  Switch back to Local: the page reads `false`. Remove QA_Second and stop the second stack afterwards (F25).
- **Docker conversations (`F31.docker`).**
  Needs a Docker daemon and a conversation image built from the same SDK as the server, with this sandbox's proxy CA (F26's CA layer). Until software-agent-sdk#6609 is released, build it from the checkout: `FROM ghcr.io/openhands/agent-server:latest-python`, copy the checkout's virtualenv `site-packages` and the four package directories at the same paths, and make `/usr/local/bin/openhands-agent-server` run `python3 -m openhands.agent_server` after `site.addsitedir` on the copied `site-packages`. Add `"OH_CONVERSATION_RUNTIME_SELECTABLE":"1"`, `"OH_CONVERSATION_IMAGE":"<that image>"` and `"OH_CONVERSATION_IMAGE_HAS_BROWSER":"false"` to `$OH_VERIFY_RUN/private/launcher-env.json`, then `control-openhands restart` and `doctor` (`api GET /server_info --pick available_conversation_runtimes` is `["local","docker"]`). The flag is on.
  ```sh
  control-openhands browser goto /
  control-openhands browser click 'testid=workspace-mode-selector'
  control-openhands browser click 'testid=workspace-mode-selector-option-docker_container'
  control-openhands conversation start --prompt "QA_F31_DOCKER Put a 👀 reaction on this message first, then reply with the single word ok." --wait --timeout 300 --stay
  docker ps --format '{{.Names}} {{.Image}}'                       # agent-server-conversation-<uuid> on that image
  control-openhands browser wait 'testid=reaction-eyes'
  control-openhands browser click 'testid=add-reaction'
  control-openhands browser click 'testid=reaction-picker-heart'
  control-openhands browser reload
  control-openhands browser wait 'testid=reaction-heart'
  ```
  Then turn the flag off in Settings → Experimental: on the Docker conversation `browser count 'testid=add-reaction'` is `0`, its chips still show, and `api POST /api/conversations/<id>/reactions --write --data '{"emoji":"tada"}'` answers 409 `Conversation reactions are turned off`. Pick **Docker** again on Home (the mode returns to the Runtime default) and start a conversation: `api GET /api/conversations/<new-id>` lists no `react`/`unreact`, and `api GET /api/conversations/<new-id>/runtime` answers 200 (it is a Docker conversation). Restore `launcher-env.json`, restart, and remove the containers with the conversations.

## Gotchas

- The flag is read when a conversation starts. Turning it on does not give a running conversation the tools, and turning it off does not take them away.
- Headless Chromium on Linux needs a colour emoji font (`fonts-noto-color-emoji`) for the chips to read as emoji in screenshots.
- A direct tool call (`execute_tool`) during a step is recorded by the run loop when the step returns; no UI path makes such calls, so two concurrent ones are covered by the SDK's unit tests, not by this family.
- `F31.react-while-running`: a person's reaction (like any message) waits for the agent's current step, which holds the state lock, so on a long command the chip comes when the command ends. That is expected today, not a hang.
- `F31.empty` needs an older Agent Server (`launch --new --sdk-version 1.60.1`); it is not part of the default pass.
