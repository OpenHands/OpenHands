# Real Local history-boundary reproduction

Use the repository's existing `control-openhands` CLI. The two Python helpers
here arrange canonical SDK-persisted fixture data and call that unchanged CLI;
they do not replace application APIs, intercept network requests, or modify the
verification feature map. No model is called. The dummy profile cannot reach a
model provider.

Run the following from each checkout, first baseline and then fix. Set
`OH_VERIFY_RUN` to the exact run directory returned by launch, and `TIED_ID` /
`CONTROL_ID` to the two IDs returned by the inert create requests.

```sh
control-openhands launch --new --run-id <unique-run-name> --build auto
control-openhands doctor --run "$OH_VERIFY_RUN"
control-openhands onboard --skip --run "$OH_VERIFY_RUN"
control-openhands fixture folder --name tied-boundary --run "$OH_VERIFY_RUN"
OH_TIED_FIXTURE_KEY=fixture-unused control-openhands llm set \
  --profile tied-fixture --model openai/gpt-4o-mini \
  --api-key-env OH_TIED_FIXTURE_KEY --base-url http://127.0.0.1:9 \
  --no-validate --run "$OH_VERIFY_RUN"
```

Create **two empty conversations** with `control-openhands api POST
/api/conversations --write --data @<run-private-request.json> --pick id --run
"$OH_VERIFY_RUN"`. The request has this structure, with `working_dir` replaced
by this run's `workspace/tied-boundary` folder. Do not supply an initial message
or invoke the conversation's run endpoint.

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
  "workspace": {"working_dir": "<owned-run>/workspace/tied-boundary"},
  "autotitle": false
}
```

```sh
control-openhands service stop agent-server --run "$OH_VERIFY_RUN"
OPENHANDS_SUPPRESS_BANNER=1 uv run --no-project \
  --with openhands-agent-server==1.61.0 \
  python .pr/tied-history/seed-boundary-history.py \
  --run "$OH_VERIFY_RUN" --tied "$TIED_ID" --control "$CONTROL_ID"
control-openhands restart --run "$OH_VERIFY_RUN"
control-openhands doctor --run "$OH_VERIFY_RUN"
python .pr/tied-history/observe-boundary-history.py \
  --checkout "$PWD" --run "$OH_VERIFY_RUN" \
  --tied "$TIED_ID" --control "$CONTROL_ID" \
  --label <before-or-after> --output <observations.json>
control-openhands doctor --run "$OH_VERIFY_RUN"
control-openhands stop --run "$OH_VERIFY_RUN"
```

The observation recipe reloads each conversation, checks its initial 50-row
tail, scrolls the actual chat container upward, and reads rendered message
markers and real event-search request queries. It repeats twice at 1440×1000
and 390×844. Each conversation contains 100 numbered user messages. Only the
TIED conversation gives messages050 and051 the same timestamp; CONTROL has
distinct timestamps. Expected: all100 appear once, with049,050,051 in sequence.

The seed helper refuses an active Agent Server or existing event history. It
uses only the selected run's generated encryption key internally to preserve
its own conversation state; that key is never printed or copied into evidence.
The recordings/screenshots contain only the disposable fixtures. Source/build
bindings and limitations are in verification.json.
