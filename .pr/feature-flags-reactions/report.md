# Live verification: feature flags and conversation reactions

Both servers were real: Canvas from this branch, against the Agent Server from
OpenHands/software-agent-sdk#6609 at `6343639` (`--sdk-path`), with
deepseek-flash, in the Paper theme.

## What changed since the first run

The first run's `heart-after-reload.png` showed no 👀 on the person's message,
even though the prompt asked for it. The agent had reacted and then taken the
reaction off just before answering, because the `react` tool told it to
"unreact once you have answered". #6609 now says a reaction stays on the
message, and both verify skills ask in plain words ("react … first, then …")
and check that the agent's reaction is still on after the answer. The chip
emoji are also bigger (`text-base`), so 👀 reads at a glance.

## Agent Server: verify-agent-server, family F37

`control-agent-server map run --fresh --file F37-reactions.md --record`:
6 pass, 0 fail (`F37.flag`, `F37.flag-gates`, `F37.rest`,
`F37.rest-validation`, `F37.tool`, `F37.rest-default`). `F37.tool` sends
"Please react to this message with the eyes emoji first, then reply with the
single word ok." and checks exactly one agent `ReactionEvent` for `eyes` on
that message, with `removed: false`.

## Canvas: verify-openhands, control-openhands

The 👀 run used Engel's own prompt: "Please react to this message with the eyes
emoji first, then tell me in one short sentence what 17 times 23 is." Three
conversations across two runs each ended with one agent 👀, still on.

Checks: 5 — pass 5, fail 0, blocked 0, not-run 0, xfail 0, xpass 0. Families: 1.

## Families

| Family | pass | fail | blocked | not-run | xfail | xpass |
|---|---|---|---|---|---|---|
| F30 | 5 | 0 | 0 | 0 | 0 | 0 |

## Checks

| Feature/check | Entry point | Expected → actual | Result | Note | Evidence |
|---|---|---|---|---|---|
| F30.agent-reaction | New conversation, flag on, Engel's prompt | agent has react/unreact; 👀 chip from agent stays on my message after the answer; no tool card → tools end react,unreact; one agent ReactionEvent eyes removed=false; chip sources=agent; Reacted cards=0 | pass | plain prompt 'react with the eyes emoji first, then …' (the one that left no chip before the guidance fix); same result in 2 more conversations on the previous run | `evidence/F30.agent-reaction/eyes.png` |
| F30.off | Settings > Features > off | REST 409; no add-reaction; existing chips stay; new conversation without react/unreact → REST 409 'Conversation reactions are turned off'; add-reaction 0; 👀 and ❤️ chips shown; new conversation has no react | pass |  | `evidence/F30.off/old-conversation.png` |
| F30.open | Command menu > Features | /settings/features lists Conversation reactions, off → url /settings/features; switch off | pass |  | `evidence/F30.open/off.png` |
| F30.toggle | Settings > Features > Conversation reactions | on at once, API agrees, still on after reload → API "enabled": true; after reload checked=true | pass |  | `evidence/F30.toggle/on.png` |
| F30.user-reaction | Agent answer > add reaction > ❤️ | pressed user chip, kept after reload; click removes → after reload: user true; 👀 still on; after click count=0 | pass |  | `evidence/F30.user-reaction/picker.png` `evidence/F30.user-reaction/eyes-and-heart.png` |

## Screenshots

- `features-off.png`, `features-on.png`: Settings → Features, off then on.
- `agent-eyes.png`: the agent's 👀 on the person's message after its answer.
- `picker.png`: the reaction picker under the agent's answer.
- `eyes-and-heart.png`: 👀 from the agent and ❤️ from the person, after a reload.
- `flag-off.png`: flag off: chips stay, no picker.
