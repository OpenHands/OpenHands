# Focused Paper verification maintenance

This follows the basic Paper mapping already merged in #19480. It strengthens the local-font and keyboard checks, adds a credential-free user-message syntax recipe, covers Paper's Old/New editor consumers and Neutral restoration, and records the terminal/embedded prerequisites. It does not change Paper or the verification harness.

**Scope verdict:** six targeted checks across five families: **3 pass, 1 existing failure, 2 not-run**. These are not full-family passes. The existing [#17567](https://github.com/OpenHands/OpenHands/issues/17567) Monaco disposal error remains; no product fix or paired baseline/fix proof is claimed. The every-offered-theme wording corrections do not add new contrast measurements or rewrite historical receipts.

| Check | Result | Evidence |
|---|---|---|
| F16 Paper selection, actual local font, reload/navigation persistence, phone keyboard, Neutral restoration | Pass; repeated with a fresh browser profile | [Phone](settings-phone.png), [round trip recording](settings-roundtrip.mp4) |
| F06 real user-message syntax and heading/body/code typography at desktop and phone width | Pass; exact documented block executed twice | [Desktop](message-paper.png), [phone](message-paper-phone.png) |
| F08 Paper Diff/Old/New palette, distinct old/new contents, Neutral editor restoration | Pass for these palette observations; exact new mode block executed twice | [Old](diff-old-paper.png), [New](diff-new-paper.png), [Neutral](diff-neutral.png), [mode recording](diff-modes.mp4) |
| F08 existing mode-switch error check | Fail: one `TextModel got disposed before DiffEditorWidget model got reset` pageerror in the final bounded replay | [Observed error](observations.json) |
| F27 Paper ANSI/selection palette and restoration | Not-run: no authorized model execution or genuine agent terminal events | [Ledger prerequisite](ledger.json) |
| F26 embedded Paper font, license and host-style isolation | Not-run: no built embedded Paper host in this focused run | [Ledger prerequisite](ledger.json) |

[Sanitized ledger](ledger.json) preserves the per-check verdicts. [Observations](observations.json) retain the loaded `Canvas Newsreader` FontFace, completed same-origin font responses, computed renderer values, phone dimensions, known error and shutdown result. The font check no longer treats `document.fonts.check()` as proof of a local load or claims the whole app makes no external font requests.

## Run identity and authorization

- Run: `paper-firstpass`, 2026-11-07 **19:18:37–19:31:03 UTC**. An initial launch/doctor/screenshot/ledger/stop cycle also proved evidence survives shutdown; the same run was restarted for the focused checks.
- Repository: OpenHands/OpenHands. Product BASE and TARGET: `c157abf3bdc1d795aadedb20c9862a2518e08806`; only verification-map prose changed during the run. No product comparison or maintenance-baseline advance.
- Build: `npm run build:app`, successful; build identity `d6ba345dda550ba8`. Dependencies replayed from the committed lock with the explicitly selected npm 11.12.1 binary (`ci --ignore-scripts --no-audit --no-fund`). The launcher build shell used active npm 11.19.1 (repository pin: 11.12.1). No dependency files changed.
- Platform: macOS 27.0.1 (26A434), Node 24.22.1, Google Chrome 157.0.8103.38 through `control-openhands` with its own browser profile. Desktop 1440×1000; phone 390×844.
- Real Local Agent Server/SDK **1.60.1**. Automation pin **1.30.1**, health endpoint 200. Owned ports: ingress 18800, Agent Server 18801, automation 18802, frontend 18803, reserved VS Code 19801.
- Scope allowed no paid model execution. **Model calls/spend: zero.** No mock backend, mock LLM, route interception or fabricated agent/tool history was used. External font/icon/version requests made by the app are not model requests.
- Newsreader is the bundled Paper face at this product SHA. Paper and OpenHands-Neutral were exercised; other theme contrast sweeps, model-dependent agent replies/tool output, Cloud, ACP, Electron and embedded Canvas were not run.

## Reproduction and limits

Use the revised F16 Paper bullet, F06 Paper user-message block and F08 Paper mode block with the existing `control-openhands` CLI. Every new command was executed. The new F06/F08 code blocks were replayed literally twice; the focused F16 flow was repeated after `browser reset`. The unchanged model-dependent/full-family recipes were not rerun.

F06 used a genuine idle conversation: no initial message, `autotitle:false`, no tools, a dummy key and a deliberately unreachable `http://127.0.0.1:1/v1` endpoint. The real API accepted a user event with `run:false`; no Send or Play action occurred. The conversation remained `idle`. This proves user-message rendering, not agent output or composer submission.

F08 used an actual run-owned repository from `fixture git-repo --name qa-paper` and an idle conversation with the same safe configuration. The committed `src/calc.py` contained `add`; the working tree changed its return to `a + b + 1` and added `sub(a, b)` returning `a - b`. UI Commits → Uncommitted → `src/calc.py` loaded real Git content. Old excluded `sub`, New included it, both retained Paper's editor background/ink, and Neutral restored the dark editor background. Direct Bash API output was not substituted for the Terminal event consumer.

The recordings are authentic CLI captures. The settings recording includes desktop/phone viewport changes and returns to Neutral; the mode recording shows Diff → Old → New → Diff. No recording pauses or edited frames were used. Screenshots and sampled video frames were reviewed before publication; local session URLs, keys, raw private logs and raw traces are omitted.

## Validation and cleanup

Map checks passed: 29 families / 842 IDs; coverage found zero unmapped routes/component directories; 1,240 referenced test IDs resolved. Palette-only path mapping now selects F06 for `src/themes/syntax-highlighter-themes.ts` and F27 for `src/themes/terminal-themes.ts`. `git diff --check` passed. These static checks complement the targeted live evidence; they do not establish rendering by themselves.

All three owned idle conversations were deleted. The final theme was OpenHands-Neutral, final doctor passed, and `control-openhands stop` confirmed the browser/launcher stopped and all five owned ports closed. Retained screenshots were checked after shutdown. Both run-owned scratch repositories and the remaining dummy request files were removed. Private browser/service state and logs remain outside the PR artifacts. Other processes, tabs, automation status/schedule and the accepted maintenance baseline were untouched.

This report was generated by an AI agent (Codex) on behalf of Engel Nyst.
