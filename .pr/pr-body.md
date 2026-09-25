HUMAN:

Verified the change locally by providing structured RCA input containing a summary, evidence, and suspected component. Confirmed that the RCA context is correctly passed to the coding agent and used during repository analysis. Also verified that existing non-RCA task behavior remains unchanged.

---

AGENT:

Implemented the feature end-to-end in Agent Canvas. Reviewer artifacts under `.pr/` on this branch: `rca-import-modal.png` (import modal with a pasted HolmesGPT-style payload) and `rca-attached-chip.png` (attached chip on the home launcher), captured by `.pr/capture-rca-evidence.mjs` against `npm run dev:mock`. These screenshots show the UI in MSW mock mode, not a full live agent run; the composed first-message payload is verified by unit tests on both the local and Cloud service paths.

## Why

External incident-analysis tools (e.g. HolmesGPT) produce structured root-cause findings — summary, evidence, suspected components/files, recommended action. Previously the only way to feed that into OpenHands was pasting raw JSON as free-form task text, which loses structure and gives the agent no instruction to verify the analysis before editing.

## Summary

- Add generic `RcaContext` parsing/prompt utilities and a `/launch` deeplink `rca` param (base64 JSON; plugins still required), plus a `buildRcaLaunchPath` helper for external producers.
- Add an "Add RCA context" import control on the home launcher with a paste-JSON modal, attach/remove chip, and RCA-only submission support via `pendingRcaContext` in the conversation store.
- Compose the RCA block into the first user message (`initial_message`) for both local and Cloud conversation creation; behavior is unchanged when no RCA is supplied.

## Issue Number

Fixes #17711

## How to Test

- `npm ci && npm run dev:mock`, open `http://localhost:3001/conversations`, click **Add RCA context**, paste a payload such as `{"summary":"Pool exhausted","suspected_files":["src/db/pool.ts"],"source":"holmesgpt"}`, and confirm the chip attaches; submitting creates a conversation whose first message contains the formatted "Root Cause Analysis Context" block.
- Or replay the capture: start `npm run dev:mock`, then `node .pr/capture-rca-evidence.mjs`.
- Deeplink: `/launch?plugins=<base64-json>&rca=<base64-json>`; `?rca=` alone still errors since plugins are required.
- Unit tests: `npx vitest run __tests__/utils/rca-context.test.ts __tests__/components/features/home/rca-import-control.test.tsx __tests__/routes/launch.test.tsx __tests__/api/agent-server-conversation-service.test.ts __tests__/components/features/home/home-chat-launcher.test.tsx`.

## Video/Screenshots

- Import modal: `https://github.com/sid288791/OpenHands/blob/feature/rca-context/.pr/rca-import-modal.png`
- Attached chip on home launcher: `https://github.com/sid288791/OpenHands/blob/feature/rca-context/.pr/rca-attached-chip.png`

## Type

- [ ] Bug fix
- [x] Feature
- [ ] Refactor
- [ ] Breaking change
- [ ] Docs / chore

## Notes

- The RCA schema is tool-agnostic (`summary`, `evidence`, `suspected_components`, `suspected_files`, `recommended_action`, `source`) and accepts snake_case or camelCase keys.
- No `compatibility.minimumAgentServer` bump: the RCA block reuses the existing `initial_message` contract on both backends.
- Verification run: `npm run lint` (typecheck + eslint + prettier) clean; 212 focused unit tests pass across the files listed above.
