# Current-head virtualization review

PR [#17820](https://github.com/OpenHands/OpenHands/pull/17820) is useful for limiting mounted message rows, but it is not ready to approve: loading older messages with different heights moves the reader's position. This was reproduced in the current production build through Control OpenHands, with a paired clean-main control.

| Build | Revision | Build fingerprint | Agent Server |
|---|---|---|---|
| PR #17820 | `734bafb4164f28b01f607446bca29d8caf5a3778` | `dfe21b4a39ed478f` | 1.59.0 |
| Main control | `dce9d017abe9ed3d362bc77e0cd3fff2e4931f22` | `c55e638e36a44cbb` | 1.59.0 |

Both isolated stacks passed `control-openhands doctor` before driving and after the measurements. The warning about an unconfigured model is expected for this rendering-only fixture. Chrome was used at 1440×1000 and 390×844. No route interception, mock responses, model calls or tool execution were used.

## Observed result

| Viewport | Main anchor drift | PR anchor drift | PR mounted rows during virtualized page loads |
|---|---:|---:|---:|
| Desktop | 0 px on every page | −109 to +83 px | 21–23 |
| Phone | 0 px on every page | −310 to +360 px | 18–20 |

The PR keeps the DOM small: around 823–858 total elements during the desktop virtualized loads, compared with main growing to 11,225 elements with all 600 messages loaded. This confirms the virtualization's benefit as well as its anchoring regression. These are DOM counts, not heap-memory measurements.

Each attempt starts with the newest 50 messages. The driver scrolls to 200 px, remembers the first fully visible message, then scrolls to 50 px to trigger the next real history page. It measures that same message before and after loading and subtracts the deliberate scrolling: `drift = after.top - before.top + scrollBy`. Eleven older pages are loaded; a twelfth sample at the beginning of history loads nothing. The first plain-list prepend has zero drift in both builds. The PR starts virtualizing at 150 rendered rows; later virtualized pages continue to drift, so this is not only a threshold transition.

The phone MP4 pair is included because it shows the largest difference with little artifact size. JSON files cover both viewports, preserving every sampled frame's anchor geometry and scroll/DOM metrics. Unrelated row geometry was omitted to keep the copies small; each JSON includes the SHA-256 of its complete original. Screenshots were visually checked and contain only synthetic messages and normal application chrome.

## Likely cause and narrow follow-up

In the reviewed PR, `messages.tsx:569–590` freezes an average height for unmeasured rows. `chat-interface.tsx:544–556` restores older-history scrolling using the total scroll-height difference. A virtual shell's estimated total is not the actual inserted height when rows vary; measuring visible rows changes it again. The paired geometry result supports this diagnosis.

Keep the virtualizer, but preserve the visible row's stable key and offset within that row through a prepend. Give one component ownership of restoring that anchor after rows are inserted and measured. Add a real-browser regression with mixed row heights, including the 150-row transition and several subsequent pages on desktop and phone. A uniform-height fixture or jsdom test cannot establish this geometry contract.

The separate minimal performance PR removes repeated action-history scans, reasoning deduplication scans and composer-triggered history work. It complements virtualization. It does not claim to bound retained history or expanded tool-group children. PR #17820 also virtualizes outer grouped rows: one very large expanded group is still a large rendered row.

## Reproduce

Use each checkout's own Control CLI and production bundle. Launch an isolated run with `control-openhands launch --new --print-run`, set `OH_VERIFY_RUN` explicitly on every command, run doctor, and skip onboarding. Chrome can be selected with `CONTROL_OPENHANDS_BROWSER=/Applications/Google Chrome.app/Contents/MacOS/Google Chrome` before launch. Keep both builds on Agent Server 1.59.0.

1. Create an inert conversation using the Control API arrangement verb. Supply an OpenHands Agent with `tools: []` and an LLM object containing `model: "openai/gpt-4o-mini"`, `api_key: "fixture-unused"`, and `base_url: "http://127.0.0.1:9"`. Use an owned folder under the run workspace. Do not start the conversation or send messages.
2. Stop this run's Agent Server with `control-openhands service stop agent-server`.
3. Seed the empty conversation using official SDK serialization:

   ```sh
   OPENHANDS_SUPPRESS_BANNER=1 uv run --no-project --with openhands-agent-server==1.59.0 \
     python PATH/seed-variable-history.py --run "$OH_VERIFY_RUN" --conversation "$CONVERSATION_ID"
   ```

4. Run `control-openhands restart`, doctor, and `browser goto /conversations/$CONVERSATION_ID`. Set `browser viewport desktop` and wait for `testid=chat-scroll-container`.
5. From that checkout's root, run `node PATH/measure-anchors.mjs "$OH_VERIFY_RUN" desktop-label`. All browser operations use the public Control CLI. The JavaScript eval expressions only read DOM geometry.
6. Set `browser viewport phone`, reload, wait for the chat container, then rerun the same driver with a new label.
7. Record evidence under `F06.load-older-history` and stop only the owned run with `control-openhands stop`.

The seed contains 600 synthetic assistant MessageEvents with 0–8 additional paragraphs. They use genuine SDK persistence and are served by the real history API. This is a rendering/pagination fixture, not a complete runnable agent conversation: do not use it to test subsequent model turns. This review does not claim full Cloud, ACP, streaming, or every expansion-state pathway coverage.

Original complete raw JSON, desktop MP4s and GIFs remain under `evidence/F06.virtualization-review/` in these Control runs:

- `virtualization-pr-live-20261103`
- `virtualization-baseline-live-20261103`

Focused existing-PR tests also passed: 5 files, 83 tests. They do not replace the live geometry result above.
