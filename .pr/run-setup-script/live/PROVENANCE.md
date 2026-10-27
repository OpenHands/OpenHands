# Live-model evidence — provenance

Everything under `live/` was captured in one session on a real local Canvas, driven through
the browser daemon. No frame was recreated or edited.

## Environment

| Item | Value |
| --- | --- |
| Canvas checkout | PR head `d355e74c7` (`feat/run-setup-script-at-conversation-start`) |
| Agent Server / SDK | `1.58.2`, built from `software-agent-sdk` branch `feat/conversation-execute-tool-endpoint` (PR #6099), launched with `control-openhands launch --sdk-path /tmp/sdk6099` |
| Advertised capability | `direct_tool_execution_v1` present in `GET /` `capabilities` |
| TypeScript client | pinned `@openhands/typescript-client` `dist/` replaced with the same SDK branch build (adds `ConversationClient.executeTool`); frontend rebuilt so the bundle carries it |
| LLM (live) | model `deepseek-v4.1-flash` via `https://llm-proxy.app.all-hands.dev`, profile `live-doc` |
| Browser | Chromium, viewport 1440x1000, `control-openhands browser record` at 12 fps |
| Workspace fixture | `/tmp/ohlive2/repo` — `.openhands/setup.sh` sleeps 6 s, `export LIVE_SETUP_TOKEN=live-setup-ok`, writes `.live-setup-marker.txt` |

## Artifacts

| File | What it shows |
| --- | --- |
| `setup-before-first-turn.mp4` | Continuous take: `Creating conversation…` → `Setting up workspace…` → first turn → `TOKEN_IS=live-setup-ok` |
| `setup-before-first-turn.gif` | Same, GIF fallback (960 px) |
| `frame-01-setting-up-workspace.png` | Video frame at 0:06 — the setup toast |
| `frame-02-first-turn-result.png` | Video frame at 0:18 — the first turn's answer |
| `conversation-result.png` | Full-page still of the finished conversation |

## Ordering at the wire level

Browser request log (app-origin), same conversation `7e9fdee3-4bd2-43f4-83ad-22c1a9d28077`:

```
03:11:05.279Z  seq 2057  POST 200  /api/conversations/7e9fdee3…/execute_tool   ← .openhands/setup.sh
03:11:12.417Z  seq 2058  POST 200  /api/conversations/7e9fdee3…/events         ← first user turn
```

Event stream:

```
03:11:05.494  SystemPromptEvent
03:11:12.419  MessageEvent      user  Run exactly: echo TOKEN_IS=$LIVE_SETUP_TOKEN; cat …
03:11:13.943  ObservationEvent  env   TOKEN_IS=live-setup-ok
                                      live setup finished at Tue Oct 27 03:11:11 UTC 2026
```

`execute_tool` completes ~7 s before the first `MessageEvent`; the `export` in the setup
script is visible in the first turn's observation.

## Reproduce

```sh
control-openhands launch --new --sdk-path /tmp/sdk6099 --build auto
control-openhands llm set --profile live-doc --model openai/deepseek-v4.1-flash \
    --base-url https://llm-proxy.app.all-hands.dev --api-key-file <key>
control-openhands browser record start --feature F03.setup-script \
    --name live-setup-before-first-turn-video --fps 12
control-openhands conversation start --workspace /tmp/ohlive2/repo \
    --prompt "Run exactly: echo TOKEN_IS=\$LIVE_SETUP_TOKEN; cat .live-setup-marker.txt" --wait
control-openhands browser record stop --gif
```

The key is a short-lived managed proxy key; it is not stored in this repository.
