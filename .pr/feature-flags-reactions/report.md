# Verification ledger — run 2026-11-07T2032-9b1b9c

Revision `ea66b6da8d91287724c98b38fdf33de3e3a32df5`, local mode, Agent Server 1.60.1, base http://127.0.0.1:18800.

Checks: 5 — pass 5, fail 0, blocked 0, not-run 0, xfail 0, xpass 0. Families: 1.

## Families

| Family | pass | fail | blocked | not-run | xfail | xpass |
|---|---|---|---|---|---|---|
| F30 | 5 | 0 | 0 | 0 | 0 | 0 |

## Checks

| Feature/check | Entry point | Expected → actual | Result | Note | Evidence |
|---|---|---|---|---|---|
| F30.agent-reaction | New conversation, flag on | agent has react/unreact; 👀 chip from agent on my message; no tool card → tools=react,unreact; chip sources=agent; Reacted cards=0 | pass | prompt asks to leave the reaction on; with a plain ask the agent follows the tool guidance and unreacts just before answering | `evidence/F30.agent-reaction/eyes-2.png` |
| F30.off | Settings > Features > off | REST 409; no add-reaction; existing chips stay; new conversation without react/unreact → REST 409; add-reaction count 0; eyes chip shown; new tools: none | pass |  | `evidence/F30.off/old-conversation.png` |
| F30.open | Command menu > Features | /settings/features lists Conversation reactions, off → url /settings/features; switch off | pass |  | `evidence/F30.open/off.png` |
| F30.toggle | Settings > Features > Conversation reactions | on at once, API agrees, still on after reload → API "enabled": true; after reload checked=true true | pass |  | `evidence/F30.toggle/on.png` |
| F30.user-reaction | Agent answer > add reaction > ❤️ | pressed user chip, kept after reload; click removes → after reload: user true; after click count=0 | pass |  | `evidence/F30.user-reaction/heart-after-reload.png` |
