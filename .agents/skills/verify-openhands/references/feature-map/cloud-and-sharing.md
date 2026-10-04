# F11 — Cloud authorization, organizations and sharing

Reach the selected Cloud organization, resume its sandbox-backed work and view an
intentionally shared conversation. Source:
[device route](../../../../src/routes/device-verify.tsx),
[shared route](../../../../src/routes/shared-conversation.tsx),
[Cloud services](../../../../src/api/cloud),
[backend registry](../../../../src/api/backend-registry).

## Sub-features

- `F11.device`: valid device authorization, reject/cancel/expired-code paths.
- `F11.organization`: backend/org switch, correct scoped lists and auth/logout behavior.
- `F11.sandbox`: real Cloud create/pause/resume with correct conversation ownership.
- `F11.sharing`: create/revoke a harmless share and view it as the intended recipient.
- `F11.entitlements`: permitted balance/subscription and capability-dependent UI.

## How to get to it (user POV)

Backend selector → Add backend → Cloud; select the authorized test organization.
Use `/oauth/device/verify` for a code supplied by an actual test login. Open sharing
through the conversation menu and `/shared/conversations/:conversationId` in a
separate viewer context.

## Driving it with Playwright

Preconditions: explicit authorization for a test Cloud account, organization,
resource/billing scope and device flow. Local Agent Server is not a substitute.

- **Authorize:** initiate the real test login, follow its device flow and approve only
  its known code. Verify the resulting backend/account, not just a success message.
  Test an expired/invalid code without guessing another user's active code.
- **Organization:** switch among authorized test orgs and verify correct settings and
  conversation scope after navigation/reload. Confirm logout does not retain usable
  credentials in the wrong context. Do not expose tokens or personal records.
- **Sandbox:** create harmless test work, pause and reopen; verify the sandbox resumes
  before events reconnect and the conversation retains its identity/history.
- **Share:** share only an approved non-sensitive fixture, open its link in a separate
  context and verify actual messages. Revoke it and verify access is removed.
  Capture missing/forbidden states separately from successful sharing.
- **Entitlements:** inspect the authorized account's available balance/subscription
  views without purchasing, changing billing or altering production entitlements.

## Gotchas

Local `/api/shared-conversations` or shared-event endpoints may return 404. That
proves neither Cloud failure nor sharing success. URLs, auth codes, session state,
conversation exports and account details require privacy review. Never authorize
an unknown device or infer permission to spend from access to a credential.
