# F16 agent summary (verbatim)

The final message of the OpenHands agent (Agent Canvas, DeepSeek V4.1 Flash) that ran F16 cold. Its first attempt stopped at the last recipe when the model account ran out of credit; after a top-up the same conversation was resumed with one message asking it to finish, and this is its closing reply.

---

All F16 recipes are run, recorded, and the stack is stopped cleanly. No repository files were modified (`git status` clean).

## Result

**Run** `2026-10-05T0154-f1e371` · revision `d6bd50bb633b` · local mode, Agent Server 1.50.1. Ledger: **24 checks — 19 pass, 4 fail, 1 blocked, 0 not-run**.

### Pass / fail / blocked per F16 sub-feature

| Sub-feature | Result | Note |
|---|---|---|
| F16.open | **pass** | gear → `/settings/agents` → Application; command menu; direct URL; Save disabled |
| F16.language | **fail** (mixed) | UI/API switch + persist pass; `<html lang>` stays `en` |
| F16.color-theme | **pass** | instant, localStorage, other pages, absent from backend |
| F16.analytics | **pass** | switch + API + telemetry localStorage; restored off |
| F16.analytics-cloud | **blocked** | no Cloud backend/account in an isolated local run |
| F16.sound | **pass** | saves; `/assets/notification-*.mp3` played on finish; negative (off) check `plays: []` |
| F16.checklist | **pass** | instant hide/show, no Save, survives reload |
| F16.title-model | **pass** | options correct, saved `deepseek-pro`, conversation titled, Automatic clears key |
| F16.title-model-fallback | **pass** | UI delete clears pref; stale outside-UI key shown as Automatic, Save disabled |
| F16.manage-profiles-link | **pass** | → `/settings/llm`, rows listed |
| F16.voice-endpoint | **pass** | saves per keystroke, key `type=password`/session-storage, absent from API; composer mic not driven (per recipe) |
| F16.git-identity | **fail** (mixed) | save/persist pass; **agent commits still `Author identity unknown`** |
| F16.git-identity-validation | **pass** | native `@` validation; empty → defaults |
| F16.save-states | **fail** (mixed) | enable/disable + skeleton pending pass; **failed save discards edits** |
| F16.keyboard | **fail** | first Tab skips all three switches; no `switch`/`checkbox` role |
| F16.phone | **pass** | no horizontal overflow (348 @ x16), Save reachable |
| F16.title-model-clear | **pass** | hover + × resets to Automatic, Save disabled |
| F16.dropdown-filter | **pass** | `Deu` filters to Deutsch; Escape restores, Save disabled |
| F16.browser-scope | **pass** | backend `de` follows a fresh profile; theme/checklist reset |

All 4 fails reproduce the **known product bugs already documented in the F16 entry** (git identity never reaches the agent; switches not keyboard-reachable/announced; `<html lang>` hard-coded `en`; failed save discards edits). Family errors sweep: `appErrors: 0`, `pageErrors: 0`.

## Feedback — where the skill / README / recipes were unclear, wrong or hard to follow

**Skill (`SKILL.md`)**
- The "Control-openhands evidence add" example and the "Define features" loop never state the reporting expectation when a single recipe bullet proves two sub-features at once (I folded F16.open's nav + command-menu/URL into one row). A line on "one ledger row per recipe bullet" would remove the guesswork.
- No verb list mentions `browser media --clear`; I only found it via `browser --help`. The skill's "Verbs that are easy to miss" would benefit from it (it is the arrange step for the F16.sound negative check).

**README**
- "Feature entry contract" says the H1 is followed by *one paragraph* then `Source:`; F16 in fact has a two-sentence paragraph. Harmless, but the wording implies a stricter shape than the map enforces.
- The full-page-screenshot note ("`--full-page` … capture only the viewport") reads as a contradiction; it means a `--full-page` shot is still viewport-bounded inside scroll containers. Phrasing is confusing on first read.

**F16 recipes**
- **F16.save-states sampler is timing-fragile.** The recipe's expected string is exactly two states (`Save Changes skeleton=false -> - skeleton=true`), but following it literally (wait for the `HTTP request failed` toast, *then* read) captures three, because the form has already returned: I got `Save Changes skeleton=false -> - skeleton=true -> Save Changes skeleton=false`. Either state "at least this prefix" or read `window.__seen` right after the click, before waiting on the toast.
- **F16.title-model "the start output's own title is often still null"** did not hold — `conversation start` already returned the generated title (`🔧 Acknowledge with ok only`). The claim is drift, not a bug; a note that it may already be populated would be more accurate. The example title `✅ Reply with only ok` also differs from the `🔧` prefix actually used.
- **F16.open's "snapshot shows comboboxes … and the sidebar 'Nouvelle discussion'"** conflates the scoped `testid=app-settings-screen` snapshot (which has no sidebar) with the screenshot. The heading assertions match; the sidebar names only appear in the screenshot.
- **F16.keyboard** labels the same behavior both as "Expected" (switches before the title input) and "Known failure" (first Tab lands on the title input). It's internally consistent once you read both, but the file-order "Expected" line reads as the acceptance criterion; a `Known current result:` label would avoid a false-fail reading.
- **F16.title-model-clear's selector** `'testid=title-llm-profile-input >> xpath=../..//button[not(@aria-label)]'` is brittle: `Show suggestions` is also an unnamed-attribute button, so the xpath relies on the DOM nesting to disambiguate. It worked here (Playwright strict mode would have failed otherwise), but a stable test id would be better.
- **F16.browser-scope's final "Restore English and save"** is the only recipe bullet with no explicit commands (every other bullet gives them). After the reset the UI is German, so the followable steps (open Language, pick English, save) have to be inferred.
- Minor: `F16.title-model-fallback`'s cleanup (Sound on/save, off/save) is correct but subtle — it only removes the stale key because a *different* saved field is toggled; the reason is implied rather than stated.

**Environment (not a defect, but worth flagging):** the mid-run stack restart from my side left the browser at `about:blank` while backend state and the evidence ledger survived; I recovered with `browser reset` + `onboard --skip` + `goto`. The skill could note that a launcher restart detaches the browser page while backend-saved state persists, so a family can resume from the page that was being driven.
