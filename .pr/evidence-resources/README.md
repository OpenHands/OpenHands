# Conversation Resources verification

Code: `4903d251b0104500b8b3580500f571f168586f49`; base:
`cc99322a1efd0f4be8d6934ae93a4d1d4bc33e05`. Frontend build: Node 24.22.1,
npm 11.12.1 on Windows. Production launcher: Node 24.21.0 in existing Ubuntu WSL,
real Agent Server/SDK 1.60.1 and automation 1.30.1. Chrome was the system browser.

## Live setup

Launch this checkout with its unmodified
`control-openhands launch --new --no-browser --build auto --port-from 3270 --timeout 240`.
Run `control-openhands doctor --skip-ui`. Use the run's own session key for fixture
API calls; never publish it.

Create a local LLM profile pointing to an unreachable loopback provider
(`openai/gpt-4o-mini`, validation-only key, `http://127.0.0.1:9`), an Agent Profile
referencing it, and a custom secret `RESOURCES_TOKEN` containing a dummy value.
Allow-list that secret in the Agent Profile. Disable public skills for this fixture.
Put a `resources-project/SKILL.md` fixture under the run workspace's `.agents/skills`.
Create a conversation with that Agent Profile and LocalWorkspace, with no initial
message. No inference starts.

Open the conversation's Resources tab. Check profile revision 0, model, project
skill and the allowed secret name. The server's project skills are lazy; the
panel reads the existing SkillsClient API using this conversation's workspace.

Use the profile row's Settings link, rename the Agent Profile through the real
editor to `resources-renamed`, save, and return. Current revision is now 1;
the conversation still reports launch revision 0 and its original model.
The UI shows the changed-profile notice. No dummy secret value appears.

At phone width 390 × 844, open `/conversations/<id>/panel`, inspect Resources,
reload and repeat. Both attempts retain the Resources selection and displayed
launch data. The drawer-open preference itself is deliberately session-only.

## Artifacts

- `desktop-before.png`, `desktop-after.png`: before/after the profile rename.
- `desktop-configuration.png`: condenser, verification and MCP category display.
- `desktop.webm`: real UI transition through the existing Agent Profile editor.
- `phone.png`, `phone.webm`: phone route and two reload attempts.

All captures are unmodified running-Canvas output. Cloud and ACP component gates
are tested with underlying-service mocks; no live Cloud account or ACP provider
was used. MCP health is the existing store's last configuration check, not a new
runtime monitor or probe.

## Local checks

- `npm run lint`: exit 0, 177 pre-existing warnings, no new errors; typecheck and
  formatting pass on final code.
- `npm run build:app`: exit 0 through the canonical production launcher.
- `npm run build:lib`: exit 0 before the final secret-scope clarification.
- Five focused suites (resources API/component, conversation tabs/content,
  translation completeness): 70 passed on final code.
- Complete Windows `npm test -- --maxWorkers=4 --reporter=json`: 10,198 passed,
  60 failed, 45 pending, 7 todo. It started before the last secret-scope clarification.
- Re-running every failed file on untouched cc99322 main reproduces all 60 exact
  failing test names. `full-suite-comparison.json` records the comparison.

The complete Windows run is not reported as green. The final secret-scope change
has focused coverage and a rebuilt, live-verified application.
