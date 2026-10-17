# Chat model search: running-Canvas evidence

These files were captured from the actual Agent Canvas app, not a mockup or an
MSW session. They demonstrate code commit
[`59c25112a8a4624002179980b0d01bad45eeb2fb`](https://github.com/jose-pki/OpenHands/commit/59c25112a8a4624002179980b0d01bad45eeb2fb).

## Environment

- Node 24, this checkout's production `agent-canvas` launcher and static build.
- Real local Agent Server / SDK 1.55.1 and automation 1.20.1.
- Isolated state and browser; ingress bound to `127.0.0.1:18810`.
- Profiles: `Fast` (`openai/gpt-4o-mini`) and `Smart`
  (`anthropic/claude-sonnet-4-6`), with no API keys configured.
- No LLM execution, mocked backend, or network interception. Search operates on
  saved profile summaries and does not need inference. The setup warning visible
  in the captures is expected for these keyless test profiles.
- Telemetry disabled and onboarding skipped before capture.

## Artifacts

- `filtered-model.png`: desktop (1440 × 1000), searching by underlying model ID.
  The `CLAUDE` query leaves only the `Smart` profile; `Fast` remains active until
  the user selects a result.
- `phone-search.png`: phone (390 × 844), filtering for `gpt` while `Smart` is
  active. The matching `Fast` profile and settings link remain readable.
- `search-and-select.webm`: actual browser recording. Opens the selector,
  filters by model ID, shows no-results feedback, clears search, filters by
  profile name, selects with Arrow Down + Enter, reopens with an empty query,
  then closes with Escape. The profile changes from `Fast` to `Smart` through
  the existing real activation endpoint.

## Reproduction

The verification CLI is used read-only with respect to its drivers and maps.
With Node 24 on `PATH` and npm dependencies installed:

```sh
export PATH="$PWD/.agents/skills/verify-openhands/scripts:$PATH"
export OH_VERIFY_HOME="$PWD/.tmp/pr-verify"
control-openhands launch --new --run-id chat-model-search-evidence --build auto
export OH_VERIFY_RUN="$OH_VERIFY_HOME/chat-model-search-evidence"
control-openhands doctor
control-openhands api POST /api/profiles/Fast --data '{"llm":{"model":"openai/gpt-4o-mini"}}' --write
control-openhands api POST /api/profiles/Smart --data '{"llm":{"model":"anthropic/claude-sonnet-4-6"}}' --write
control-openhands onboard --skip
control-openhands api POST /api/profiles/Fast/activate --write
control-openhands browser reload
control-openhands browser record start --feature model-search --name search-and-select
control-openhands browser click testid=chat-input-llm-profile
control-openhands browser fill testid=chat-input-model-search CLAUDE
control-openhands browser screenshot --feature model-search --name filtered-model
control-openhands browser fill testid=chat-input-model-search missing-model
control-openhands browser click 'role=button[name="Clear search"]'
control-openhands browser fill testid=chat-input-model-search Smart
control-openhands browser press ArrowDown
control-openhands browser press Enter
control-openhands browser click testid=chat-input-llm-profile
control-openhands browser press Escape
control-openhands browser record stop
control-openhands browser viewport phone
control-openhands browser click testid=chat-input-llm-profile
control-openhands browser fill testid=chat-input-model-search gpt
control-openhands browser screenshot --feature model-search --name phone-search
```

The API commands arrange preconditions; the browser interactions are the UI
proof. The CLI's doctor confirmed the served build matches the checkout's build
inputs (`787ad5572868e48d`), the pinned backend is healthy, and the UI loads
without page errors.

## Scope and limitations

These live captures prove the OpenHands profile picker, not ACP CLI execution or
Cloud backend integration. ACP filtering and unchanged selection are covered by
behavioral tests through the real mutation hook with a mocked underlying
service. Existing permission gates and backend transport are not changed.

The separate narrow-screen overflow positioning defect (#18063) remains out of
scope; this evidence does not claim it is fixed.
