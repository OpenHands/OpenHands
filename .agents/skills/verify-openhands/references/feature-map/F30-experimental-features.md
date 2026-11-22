# F30 — Experimental features and reactions

Settings → Experimental lists the features a user can turn on or off for
themselves. The list, labels and descriptions come from the Agent Server, and
a change applies to conversations started afterwards. A notice under the list
says experimental features can change and are not guaranteed to stay
backwards compatible. The page and its navigation entries exist only for local
backends; Cloud does not show them. With **Conversation reactions** on, the
agent can put an emoji reaction on the user's messages, and the user can react
to the agent's messages from a small picker under each one. The agent sees the
user's reactions on its next turn. With **Conversation mentions** on, typing `#`
in the composer lists the user's other conversations and inserts a mention,
and mentions in messages open the conversation they name.

Source: `src/routes/experimental-settings.tsx`, `src/api/experimental-features-service/`, `src/api/conversation-reactions-service/`, `src/hooks/query/use-experimental-features.ts`, `src/hooks/mutation/use-update-experimental-feature.ts`, `src/components/features/chat/message-reactions.tsx`, `src/hooks/use-settings-nav-items.ts`.

Server: F18, F37.

## Sub-features

- `F30.open`: `/settings/experimental` shows the heading **Experimental**, one switch per server feature (`experimental-switch-<key>`) with its label and description, in the order `GET /api/settings/experimental` returns them, and the notice `Experimental features can change significantly and are not guaranteed to stay backwards compatible.` (`experimental-notice`) below the list.
- `F30.toggle`: clicking a switch saves at once (no Save button): `GET /api/settings/experimental` reports the new `enabled` value, and the switch keeps it after `browser reload`. Expected: a switch that fails to save shows one error toast. Known failure (reproduced 2026-11-21, [#19933](https://github.com/OpenHands/OpenHands/issues/19933)): with the Agent Server stopped, one click returns `PATCH /api/settings/experimental/<key>` `502` and shows two toasts, `An error occurred` at once and `Could not change the feature. Please try again.` about 5 s later.
- `F30.agent-reacts`: with `conversation_reactions` on, a conversation started afterwards whose prompt asks the agent to react with 👀 shows a 👀 chip (`reaction-eyes`) under the user's message, and no tool card for the `react` call.
- `F30.user-reacts`: hovering the agent's last message shows **Add reaction** (`add-reaction`); picking 👍 in the picker (`reaction-picker-+1`) shows the chip at once, and `conversation events <id> --kinds ReactionEvent --from-start` lists a user `ReactionEvent`. The agent's next turn sees it: its reply mentions the 👍 (whether it takes a reaction as consent is the model's choice).
- `F30.picker-placement`: the picker opens beside its button when it fits and above it when it does not (`reaction-picker` `data-placement` is `beside` or `above`); either way it stays inside the viewport, also at `browser viewport phone`.
- `F30.conversation-mentions`: the list includes **Conversation mentions** (`experimental-switch-conversation_mentions`, `The agent can find your other conversations by title, read one you point it at, and link to them in its answers. Applies to conversations started after you turn it on.`). Its state applies at once in the browser, also to conversations started earlier: on, the composer's `#` list (F05.conversation-mention-picker) and mention links (F06.conversation-mention-link) work; off, `#` opens nothing and a mention reads as its plain title. What the agent does with mentions needs a model.
- `F30.flag-off`: with `conversation_reactions` off, a new conversation shows no `add-reaction` button and the agent has no `react` tool.

## How to get to it (user POV)

- Sidebar settings gear (`backend-selector-settings-link`), then **Experimental** in the settings navigation (`sidebar-settings-/settings/experimental`).
- Direct URL `/settings/experimental`.
- Command menu (`Control+k` or `Meta+k`): search "Experimental" and choose **Experimental**.
- Reactions: under the agent's messages in any conversation started while `conversation_reactions` is on.

## Driving it with control-openhands

Preconditions:

- Baseline state and an LLM: `control-openhands llm preset deepseek --api-key-file <key file>`.
- Read the flags once: `control-openhands api GET /api/settings/experimental --pick body.0.enabled` (the first entry is `conversation_reactions`; `body.2` is `conversation_mentions`).

Bullets:

- **Open (`F30.open`).** `control-openhands browser goto /settings/experimental`, `control-openhands browser wait 'testid=experimental-switch-conversation_reactions' --state attached`, `control-openhands browser text 'testid=experimental-settings-screen'` (lists every server feature's label and description), `control-openhands browser text 'testid=experimental-notice'`, `control-openhands browser screenshot --feature F30.open --name desktop`.
- **Toggle (`F30.toggle`).** If `body.0.enabled` is `false`, `control-openhands browser click 'testid=experimental-switch-conversation_reactions' --force`; after a second, `control-openhands api GET /api/settings/experimental --pick body.0.enabled` is `true`. `control-openhands browser reload` and the switch is still on.
- **Agent reacts (`F30.agent-reacts`).** `control-openhands conversation start --prompt "React with the 👀 emoji on this message using your react tool. Then ask me one yes/no question: should you create the file qa-sync.txt containing the word approved? Do not create it yet; end your turn after asking." --wait --timeout 300`, then `control-openhands browser wait 'testid=agent-message'` and `control-openhands browser count 'testid=reaction-eyes'` (`1`).
- **User reacts (`F30.user-reacts`).** `control-openhands browser hover 'testid=agent-message >> nth=-1'`, `control-openhands browser click 'testid=agent-message >> nth=-1 >> testid=add-reaction' --force`, `control-openhands browser click 'testid=reaction-picker-+1'`, `control-openhands browser screenshot --feature F30.user-reacts --name reacted`. `control-openhands conversation events <id> --kinds ReactionEvent --from-start --full` lists an agent and a user `ReactionEvent`. `control-openhands conversation send <id> --prompt "Status?" --wait`: the agent's reply (`conversation events <id> --last 4 --full`) mentions the 👍 on its question.
- **Picker placement (`F30.picker-placement`).** `control-openhands browser viewport phone`, hover the last agent message and click `add-reaction` as above, then `control-openhands browser attr 'testid=reaction-picker' data-placement` (`beside` or `above`) and `control-openhands browser bbox 'testid=reaction-picker'` (`insideViewport` `true`). Press Escape and `control-openhands browser viewport desktop`.
- **Conversation mentions (`F30.conversation-mentions`).** `control-openhands browser text 'testid=experimental-settings-screen'` lists `Conversation mentions` with its description (third of five on Agent Server 1.63.1: Conversation reactions, Decision cards, Conversation mentions, Message edits, Factories). `control-openhands browser click 'testid=experimental-switch-conversation_mentions' --force`, then `control-openhands api GET /api/settings/experimental --pick body.2.enabled` is `true` and after `control-openhands browser reload` `control-openhands browser eval "document.querySelector('[data-testid=experimental-switch-conversation_mentions]').checked"` is `true`. Then run the F05.conversation-mention-picker and F06.conversation-mention-link bullets; a mention sent while the switch was off becomes a link once it is on, without starting a new conversation.
- **Failed switch (`F30.toggle`, #19933).** `control-openhands browser toasts --history --clear`, `control-openhands service stop agent-server`, `control-openhands browser click 'testid=experimental-switch-conversation_reactions' --force`, wait 10 s, `control-openhands browser toasts --history` and `control-openhands browser network --filter experimental` (`PATCH` `502`); the page then shows `Could not load experimental features from the agent server.` Bring the server back with `control-openhands restart`.
- **Flag off (`F30.flag-off`).** Turn the switch off as in the toggle bullet, start a new conversation with any prompt, and `control-openhands browser count 'testid=add-reaction'` is `0`. Turn it back on afterwards.

## Gotchas

- A flag change applies to conversations started afterwards; a conversation started while the flag was on keeps its `react` tool after the flag is turned off.
- The switch is a visually hidden checkbox: click it with `--force`, and read the state from the API rather than the DOM.
- The add-reaction button appears on hover: hover the message first, or click with `--force`.
- Cloud backends never show this page or its navigation entries; driving them needs a Cloud account and is blocked here.
