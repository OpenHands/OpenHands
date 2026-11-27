# Real Local history projection check

Use each checkout's unchanged `control-openhands` CLI. Baseline is current main
`754abee4f0869dfc85dbd0acb6457f2e0ad17d4c`; the fixed revision is recorded in
`verification.json`. Both use Agent Server/SDK 1.62.0 and automation 1.32.1.
Run once for each checkout with a unique run ID and the same two helpers here.
Set `RUN` to the exact directory returned by launch; every command takes it.

```sh
control-openhands launch --new --run-id <unique-id> --build auto
control-openhands doctor --run "$RUN"
control-openhands onboard --skip --run "$RUN"
control-openhands fixture folder --name projection-proof --run "$RUN"
OH_PROJECTION_FIXTURE_KEY=fixture-unused control-openhands llm set \
  --profile projection-fixture --model openai/gpt-4o-mini \
  --api-key-env OH_PROJECTION_FIXTURE_KEY --base-url http://127.0.0.1:9 \
  --no-validate --run "$RUN"
```

Create two empty disposable conversations through `control-openhands api POST
/api/conversations --write --data @<private-request.json> --pick id --run "$RUN"`.
Use this body with the owned run's workspace. Do not send an initial message or
call a conversation run endpoint. Save the returned IDs as `PROJECTION` and `TIED`.

```json
{
  "agent": {
    "kind": "Agent",
    "llm": {
      "model": "openai/gpt-4o-mini",
      "api_key": "fixture-unused",
      "base_url": "http://127.0.0.1:9"
    },
    "tools": []
  },
  "workspace": {"working_dir": "<owned-run>/workspace/projection-proof"},
  "autotitle": false
}
```

```sh
control-openhands service stop agent-server --run "$RUN"
OPENHANDS_SUPPRESS_BANNER=1 uv run --no-project \
  --with openhands-agent-server==1.62.0 python seed-projection-history.py \
  --run "$RUN" --projection "$PROJECTION" --tied "$TIED"
control-openhands restart --run "$RUN"
control-openhands doctor --run "$RUN"
python observe-projection-history.py --checkout "$PWD" --run "$RUN" \
  --projection "$PROJECTION" --tied "$TIED" \
  --label <before-or-after> --output <observations.json>
control-openhands doctor --run "$RUN"
control-openhands stop --run "$RUN"
```

The seed helper refuses a running Agent Server and existing event history. It
uses canonical `ConversationState.append_event` to persist 100 events: a user
message, 49 action/observation pairs, and a final agent message. The 50-event
page boundary splits action025 (sequence49) from its observation (sequence50).
The run-owned encryption key is used internally and is never printed or copied.

The unchanged production server serves both pages. Desktop's collapsed group
is short enough to load the older page automatically; phone initially shows
25 calls and loads the older page when the transcript is scrolled up. Expand the
group: the baseline incorrectly shows 50 rows with command025 twice; the fix
should show exactly49, one per call. Open command025 and verify its output, and
type a draft without submitting. Repeat from navigation twice per viewport.

The second conversation has100 user messages. Messages050/051 share a timestamp.
Each attempt starts with50 messages; scrolling loads all100 exactly once in
order. This independently protects the cursor/order fix from #19649.

This is synthetic, canonical persisted history served through a real Local
backend. No model/tool executes, no browser store is injected, and no request
is intercepted. It proves the render/history path, not live model generation,
Cloud, ACP execution, delegated/planning conversations, or an embedded host.
The verification skill and feature map remain read-only. Their #16947/#17559
notes have no executable known-bug block; this packet therefore records
ordinary Control evidence, not an automated `bug proof` verdict.
