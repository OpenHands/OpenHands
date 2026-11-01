# Running-Canvas evidence for PR #18913 (#18883)

Before/after evidence from the REAL production Canvas build driven against a
mock of the cloud app-server surface. The only scripted part is the cloud
backend (`mock-cloud-server.mjs`, ~10 endpoints, in-memory state seeded as an
account right after an ACP onboarding: `agent_settings.agent_kind = "acp"`,
the ACP profile `codex` active, the OpenHands profile `default` available).
The Canvas build, its agent-kind sync and the UI are the real product — the
driver registers the mock as a `kind: "cloud"` backend in localStorage (the
same shape the device flow produces) and clicks through the real UI.

## Layout

- `baseline/` — unpatched `main` @ `fee10e3d2` (this PR's base)
- `pr/` — this PR's HEAD
- Each dir: screenshots, `evidence.json` (extracted state per step) and
  `cloud-requests.jsonl` (every request the Canvas sent the mock, bodies included)
- Tooling: `../mock-cloud-server.mjs`, `../evidence-driver.mjs`, `../static-server.mjs`

## Recipe (reproducible)

```sh
npm run build:app                      # the checkout under test
node .pr/mock-cloud-server.mjs &       # mock cloud app-server on :18901
node .pr/static-server.mjs &           # serves build/ on :18300
node .pr/evidence-driver.mjs --label <baseline|pr> --out .pr/evidence/<label>
```

## Result

| Step | baseline (`fee10e3d2`) | this PR |
|---|---|---|
| `POST /api/v1/settings` after activating the OpenHands profile | **none** (0 in the whole run) | `{"agent_settings_diff":{"agent_kind":"openhands"}}`, 16 ms after the activate call |
| `GET /api/v1/settings` right after activation (`settings-after-activate`) | `agent_kind: "acp"` | `agent_kind: "openhands"` |
| After deleting the active profile (`after-delete`) | `agent_kind: "acp"` | `agent_kind: "openhands"` |
| Home composer with no active profile (`home-2-no-active-profile`) | stale ACP model `gpt-5.2` from `agent_settings` | OpenHands LLM affordance (`mock-llm` pill) |

The launch step (step 6) did not reach the mock in either build (the composer
submit never fired a request under the driver), so the launch-time kind is not
part of this evidence; the settings state a profile-less launch composes from
is captured above.
