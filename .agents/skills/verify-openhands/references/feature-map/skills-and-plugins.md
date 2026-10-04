# F07 — Skills, plugins and conversation capability dialogs

Discover reusable capabilities, select what is enabled, and launch a reviewed
plugin workflow. Source: [skills route](../../../../src/routes/skills-settings.tsx),
[plugins route](../../../../src/routes/skills-plugins.tsx),
[launch route](../../../../src/routes/launch.tsx),
[conversation UI](../../../../src/components/features/conversation).

## Sub-features

- `F07.skills`: catalog/tags/search/detail; reviewed user/project skill add and enablement.
- `F07.plugins`: catalog/categories/search/detail/add; enabled-state persistence.
- `F07.launch`: valid plugin launch and missing/invalid payload error states.
- `F07.dialogs`: conversation Skills, Hooks and Tools contents and close behavior.

## How to get to it (user POV)

Use Customize (`/customize`) → Skills (`/skills`) or Plugins (`/plugins`). Open
plugin launch from a reviewed detail entry and `/launch` directly for its missing
payload state. Open conversation Skills/Hooks/Tools from the conversation menu.

## Driving it with Playwright

Preconditions: disposable workspace with a reviewed, harmless local skill/plugin
fixture. Execution checks need a real model; catalog checks do not.

- **Discover:** navigate to each route; use visible search/category controls to find
  an existing named card, then a no-match query, then clear it. Open its details and
  Add skill/Add plugin dialog using scoped roles and accessible names from the UI.
- **Enable:** add only the reviewed fixture through the actual dialog. Toggle it,
  reload and inspect effective enabled state; create a new bounded conversation
  and inspect which capability it receives when execution is authorized.
- **Launch:** start the reviewed plugin through its launch action; verify intended
  parameters arrive in the new conversation. Visit bare `/launch` separately and
  verify a useful invalid/missing-plugin state, not a blank page.
- **Conversation dialogs:** open Skills, Hooks and Tools one at a time at 390×844.
  Measure modal bounds, scroll long contents, use keyboard and close controls;
  repeat at desktop size. Inspect actual values privately before selecting evidence.

## Gotchas

A 640px dialog centered in 390px clips at x = −125px; recheck
[#17562](https://github.com/OpenHands/OpenHands/issues/17562), not just CSS names.
Recheck the plugin search width beside tabs ([#17565](https://github.com/OpenHands/OpenHands/issues/17565)).
Tool/system-message content can contain private instructions or service metadata;
inspect it without automatically publishing a full screenshot. Do not confuse
public catalog defaults with user/project overrides.
