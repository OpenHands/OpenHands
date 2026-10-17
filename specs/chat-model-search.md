# Chat model search

The chat composer model selector filters the models already available to the
user. It does not discover new models, change profiles, or modify the selected
model until the user picks a result.

| ID      | Behavior                                                                                                                                                                                                                                                                          |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MDS-001 | OpenHands LLM profiles can be filtered by profile name, raw model ID (including provider), or displayed model label. Matching ignores case and surrounding whitespace.                                                                                                            |
| MDS-002 | ACP models can be filtered by label or model ID. The agent-default choice participates in filtering when available.                                                                                                                                                               |
| MDS-003 | No matches shows localized feedback without hiding the existing settings link. Clearing restores all choices; closing and reopening the picker starts with an empty search.                                                                                                       |
| MDS-004 | Opening an inline picker focuses search. Arrow Down focuses the first matching choice; Enter on that choice retains existing selection behavior. Typing or pressing Enter inside search never submits a chat message. Escape closes the inline picker and restores trigger focus. |
| MDS-005 | Read-only model surfaces do not offer search. Permissions, available models, and model-switching APIs remain unchanged.                                                                                                                                                           |

The shared menu contents cover the inline selector and the narrow-screen Model
overflow submenu, in standalone and embedded Canvas. They use the same data for
Local and Cloud backends, and do not change standard, planning, or delegated
conversation configuration. Overflow search does not autofocus on mount because
hover submenus can be mounted while hidden.
