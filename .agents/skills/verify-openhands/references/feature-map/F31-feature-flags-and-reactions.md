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

## Gotchas

- The flag is read when a conversation starts. Turning it on does not give a running conversation the tools, and turning it off does not take them away.
- Headless Chromium on Linux needs a colour emoji font (`fonts-noto-color-emoji`) for the chips to read as emoji in screenshots.
- `F31.empty` needs an older Agent Server (`launch --new --sdk-version 1.60.1`); it is not part of the default pass.
