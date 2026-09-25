# Unknown-class lint adoption

Base: `a3cfe98267c52937dc4e91a55644a6131c6c01bc`. Pinned linter: `@shadcn/lint@0.1.0`.

The rule is enabled at warning level. Four inert class usages are also removed; generated styling and the DOM structure are preserved.

## Inventory

`npx eslint src --rule 'shadcn/no-unknown-classes:warn' --format json` initially reported **6** unknown-class warnings, alongside **347** existing arbitrary-value warnings. No theme-loading or worker-fallback warnings were emitted.

Two exact exceptions are supported by code:

- `environment-switch-overlay`: `src/index.css:134` uses it to animate child elements; that stylesheet is outside the Tailwind theme import graph.
- `conversation-overview-diffs-git-action`: a selector target in `src/components/features/conversation/conversation-overview-diffs-row.tsx:40` for clearing row hover while the action is hovered.

All **4 genuine findings are fixed** by removing inert classes:

| File | Finding and correction |
| --- | --- |
| `src/components/features/chat/drag-over.tsx` | Remove `drag-over-content`, which has no stylesheet or selector consumer; retain the wrapper element. |
| `src/components/features/home/repo-selection-form.tsx` | Remove `max-w-auto`, which emits no CSS. |
| `src/components/features/home/workspace-selection-form.tsx` | Remove the same unsupported width class. |
| `src/utils/form-control-classes.ts` | Remove unsupported `ease`; retain the existing transform transition, duration, reduced-motion handling, and default timing. |

The dropdowns pass the optional class to `cn("relative", className)`; removing it does not reveal a previously overridden maximum width. Do not substitute `min-w-auto`: minimum width is a different property. Likewise, connecting the drag overlay to the similarly named `.drag-over-ui-content` rule would newly force white text and fixed typography. That is a separate visual decision, not necessary to remove an unused marker.

`node .pr/verify-inert-classes.mjs` verifies identical compiled CSS before/after removal with the actual Tailwind theme/plugins and checks that class merging preserves all valid utilities. A repository search found no CSS/JS/test consumers of the removed marker or utilities. The transition helper is shared by caret controls in both standalone and embedded Canvas, independent of backend/agent type; it retains its existing behavior in all consumers.

## Reproduction

Run `node .pr/verify-unknown-classes.mjs` from the repository root. It uses the actual ESLint config and virtual TSX under `src/ui`, verifying that the rule remains active inside component implementations.

Observed output:

```text
"hovr:flex" ... Did you mean "hover:flex"?
"flex-cols" ... Did you mean "flex-col"?
"environment-switch-overla" ... no CSS is generated for it.
PASS: real theme/plugins accepted; typos reported; exceptions stay exact.
```

Accepted: Canvas surface/border/focus tokens, `prose`, `scrollbar-hide`, a HeroUI color with a data variant, and the two documented markers. This also detects the documented fallback mode that incorrectly accepts `hovr:flex`.

## Original adoption checks

Executed with Node 22.23.2, npm 10.9.8 on macOS (CI's supported Node environment should also run its normal checks):

- `npm run lint`: passed, **0 errors / 351 warnings** (347 existing + 4 new genuine findings).
- `npm run build`: passed.
- `npm run build:lib`: passed.
- `npm test -- --maxWorkers=2`: **733 files passed / 1 failed; 7,718 tests passed / 1 failed / 7 todo**. The existing `DeleteProfileModal > calls deleteProfile and shows success toast on successful delete` assertion expected a success-toast call but observed zero. No runtime source changed in this branch. Rerunning that file alone passed all 12 tests; the full-suite failure remains disclosed.
- Initial sandboxed test attempt was stopped because launcher tests could not bind local ports. Results above are from the rerun with port access.

The pinned rule delegates some missing color-token diagnostics to `no-raw-colors`; it is not a complete missing-token guard, nor does it detect the semantic `text-base` collision. Other rules retain their original settings.

This directory is PR-only evidence. Fork PRs require manual `.pr/` cleanup before merge.

## Warning cleanup checks

- `node .pr/verify-inert-classes.mjs`: passed; merged valid classes and compiled CSS are identical for all removals.
- `node .pr/verify-unknown-classes.mjs`: passed; valid theme/plugin utilities and exact selector exceptions accepted while typos remain reported.
- `npm run lint`: passed, **0 unknown-class warnings / 347 existing arbitrary-value warnings / 0 errors**; formatting and TypeScript checks passed.
- `npx vitest run __tests__/components/features/home/repo-selection-form.test.tsx __tests__/components/features/home/workspace-selection-form.test.tsx __tests__/components/features/home/git-repo-dropdown.test.tsx __tests__/components/features/home/workspace-dropdown.test.tsx __tests__/utils/form-control-classes.test.ts --maxWorkers=2`: **5 files / 69 tests passed**.
- `npm run build` and `npm run build:lib`: passed.
- No visual change is claimed: the removed classes have no styling or selector consumers, and the compiled CSS comparison above is identical. The PR also includes real-app workspace screenshots to satisfy frontend evidence CI; they are a smoke comparison, not pixel-diff coverage of every component.
- The original PR head passed Ubuntu and Windows test/build CI. The checks above disclose the earlier local full-suite flake; no new tests or weakened assertions were introduced for inert class removal.

### Real-app screenshot setup

`PATH=/opt/homebrew/opt/node@22/bin:$PATH VITE_DO_NOT_TRACK=1 npm run dev:mock -- --host 127.0.0.1 --port 3196 --strictPort`

Open Workspace in the real Canvas UI using MSW data. `workspace-before.png` uses the four affected source files from `0bfdebb`; `workspace-after.png` uses `a35e74e`. The same running app applied the revisions through Vite HMR. No live LLM or credentials were used. The workspace field and its shared caret retain their appearance; compiler/merge parity covers all four removals.
