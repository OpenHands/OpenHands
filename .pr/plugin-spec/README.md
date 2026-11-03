# Plugin-spec detail verification

Implementation: `29f5a1056b0e77922305e3443ac49286e443511b`.
Baseline: `1e89305416a0e090ff97bf6161e9f37fac67e610`.

Both versions were exercised against real isolated Agent Server 1.58.1 and
Automation 1.25.1 services on the same remote Linux workstation. An inactive
plugin-spec fixture with a 300-second timeout and a long configuration string
was created through `/api/automation/v1/plugin-spec`. No automation was run
and no model credentials were configured.

The repository's `control-openhands` CLI performed browser navigation,
observations, expansion and screenshots. The UI health probes passed after
allowing the real Google Fonts stylesheets through a temporary network proxy;
no response was mocked or replaced. Application APIs stayed on localhost.

## Before and after

- [Before](before.png): the script bundle is reported as unavailable, and the
  plugin definition, configuration and timeout are absent from the detail page.
- [After](after.png): the runtime-generated-bundle explanation, plugin source,
  configuration and timeout are visible.
- [Phone](after-phone.png): the same details are visible at a 390px viewport.
  Clicking Show more exposes the complete configuration string and max_results.

The fixed page's Run now button remained disabled. The recorded browser network
log contains no tarball or secret API calls; JavaScript asset names containing
"secret" are not API calls. No current end-to-end model/automation execution is
claimed.

## Reproduction route

1. Launch an isolated checkout with `control-openhands launch --new` and verify
   `control-openhands doctor`. Skip onboarding without adding an LLM key.
2. Create a local plugin fixture and arrange an **inactive** plugin-spec
   automation via `control-openhands api POST /api/automation/v1/plugin-spec
   --write --data @request.json`, with a source path, long config, cron trigger,
   `enabled: false`, `state: INACTIVE`, and `timeout: 300`.
3. Navigate to `/automations/<id>`. Observe the source/configuration/timeout and
   script section. Expand the long configuration, then repeat at phone width.
4. Capture before/after screenshots and delete the owned inactive fixture.
   Stop the isolated stack. Do not click Run now or activate the automation.

These artifacts were produced by Codex. They do not attest to human review.
