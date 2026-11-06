# Long-conversation verification

The Control OpenHands check exercises production Canvas with a real Agent Server and seeded tool history. It measures typing while new persisted user events arrive through the server and WebSocket. It does **not** execute a model or tools. The before capture reproduces repeated input stalls while retaining the draft; the after capture checks the same history and interactions.

| Run | Canvas revision | Build identity | Backend |
|---|---|---|---|
| Before, `chat-performance-before-20261103` | `dce9d017abe9ed3d362bc77e0cd3fff2e4931f22` | `c55e638e36a44cbb` | Agent Server/SDK 1.59.0; automation 1.26.1 |
| After, `chat-performance-after-20261103` | `794ec7c6898f28b6f358df93140cc829d6f2fa42` | `d2829f5551b8f6fb` | Same pins |

Both are isolated Control runs on macOS with Node 24.22.1 and the project's configured Chromium 153.0.8010.12. Doctor checks the actual served build and backend pins. Baseline source edits happened after its build was saved; its served build remained the original baseline. Test timing overlapped other work: the baseline had a two-worker unit suite and the comparison had a four-worker suite. These are diagnostic browser measurements, not a controlled hardware benchmark or a universal speedup claim.

## Results

| Viewport | Input → second frame p95, before → after | Maximum, before → after | Long tasks, before → after | DOM nodes, both builds |
|---|---:|---:|---:|---:|
| Desktop | 219.9 → 39.8 ms | 261.9 → 59.2 ms | 12 → 0 | 1,535 |
| Phone | 185.1 → 31.7 ms | 201.7 → 43.9 ms | 12 → 0 | 1,629 |


Each 6.5-second observation types the same 124-character draft with the existing Control `type` command (10 ms between keys) while six `run:false` messages arrive. The probe observes long tasks, frame intervals, and input to the **second animation frame**. This last measure includes normal frame scheduling; it is not the browser's INP metric. No React stores, fetch responses, event payloads, or app globals are modified in the browser.

The recorded behavior checks are exact draft retention, composer focus, oldest loaded history still present, and the sixth incoming message visible. All four checks pass at both desktop (1440×1000) and phone (390×844) in both builds. The input stalls, not message loss, are the baseline failure. The fixture has 2,000 SDK-serialized terminal action/observation pairs, 20 tool groups, checkpoint messages, and 4,045 persisted events before measurement. It is loaded through real lazy history pagination. The coarse heap snapshots are not retained-heap or leak measurements. The fixed code keeps the same rendered content; it does not cap DOM size or retained raw-event memory.

A separate manual check in the existing Google Chrome session opened the fixed app, typed an exact unsent draft, and expanded a tool group. The draft remained intact; no message was submitted.

Files: [before desktop](baseline-desktop.mp4), [after desktop](after-desktop.mp4), [before phone](baseline-phone.mp4), [after phone](after-phone.mp4). PNGs and sanitized JSON measurements have the same names. No live LLM generation, Cloud backend, or live ACP session is claimed by these captures.

## Reproduction

From each revision's checkout, use its own Control script. Launch with `--new --build auto`, run `doctor`, then `onboard --skip`. `OH_VERIFY_RUN` below is that explicitly selected run; `QA_ID` is the returned conversation ID.

```sh
export PATH="$PWD/.agents/skills/verify-openhands/scripts:$PATH"
control-openhands fixture folder --name perf-history --run "$OH_VERIFY_RUN"
control-openhands api POST /api/conversations --write --run "$OH_VERIFY_RUN" \
  --data "{\"agent\":{\"kind\":\"Agent\",\"llm\":{\"model\":\"openai/gpt-4o-mini\",\"api_key\":\"fixture-unused\",\"base_url\":\"http://127.0.0.1:9\"},\"tools\":[]},\"workspace\":{\"working_dir\":\"$OH_VERIFY_RUN/workspace/perf-history\"}}" --pick id
OH_PERF_FIXTURE_KEY=fixture-unused control-openhands llm set --profile perf-fixture \
  --model openai/gpt-4o-mini --api-key-env OH_PERF_FIXTURE_KEY \
  --base-url http://127.0.0.1:9 --no-validate --run "$OH_VERIFY_RUN"
control-openhands service stop agent-server --run "$OH_VERIFY_RUN"
OPENHANDS_SUPPRESS_BANNER=1 uv run --no-project --with openhands-agent-server==1.59.0 \
  python .pr/chat-performance/seed-tool-history.py --run "$OH_VERIFY_RUN" \
  --conversation "$QA_ID" --pairs 2000
control-openhands restart --run "$OH_VERIFY_RUN"
control-openhands doctor --run "$OH_VERIFY_RUN"
control-openhands api POST "/api/conversations/$QA_ID/events" --write --run "$OH_VERIFY_RUN" \
  --data '{"role":"user","content":[{"type":"text","text":"Fixture append readiness check."}],"run":false}'
control-openhands browser goto "/conversations/$QA_ID" --run "$OH_VERIFY_RUN"
control-openhands browser viewport phone --run "$OH_VERIFY_RUN"
node .pr/chat-performance/load-history.mjs "$OH_VERIFY_RUN"
node .pr/chat-performance/measure-live.mjs "$OH_VERIFY_RUN" "$QA_ID" baseline desktop
node .pr/chat-performance/measure-live.mjs "$OH_VERIFY_RUN" "$QA_ID" baseline phone
```

For an identical comparison, stop the baseline Agent Server and use `reset-fixture-tail.py` to remove only this recipe's measurement messages. Create an inert conversation with the same `conversation_id` in the after run, stop its Agent Server, and use `copy-fixture.py` to copy only event history and branch pointers; it copies no credentials. Restart and repeat the backfill and measurement commands with label `after`. The default model endpoint deliberately cannot answer, and every append uses `run:false`.

The existing Control CLI, verification scripts, and feature map remain unchanged. Reproduction helpers live only in this temporary PR evidence directory. Evidence rows use `F05.composer-resize` with separate desktop and phone entry points. Both owned runs were stopped with `control-openhands stop --run ...`; Control confirms `alive: false` for each. Evidence remains available.

This report was generated by an AI agent on behalf of the requester.
