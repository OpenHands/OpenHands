# Unknown-class lint adoption

Base: `a3cfe98267c52937dc4e91a55644a6131c6c01bc`. Pinned linter: `@shadcn/lint@0.1.0`.

The only production change is ESLint configuration. Runtime code and generated CSS are unchanged.

## Inventory

`npx eslint src --rule 'shadcn/no-unknown-classes:warn' --format json` initially reported **6** unknown-class warnings, alongside **347** existing arbitrary-value warnings. No theme-loading or worker-fallback warnings were emitted.

Two exact exceptions are supported by code:

- `environment-switch-overlay`: `src/index.css:134` uses it to animate child elements; that stylesheet is outside the Tailwind theme import graph.
- `conversation-overview-diffs-git-action`: a selector target in `src/components/features/conversation/conversation-overview-diffs-row.tsx:40` for clearing row hover while the action is hovered.

The resulting **4 genuine findings remain warnings**, for a separate cleanup after checking intended behavior:

| File | Finding |
| --- | --- |
| `src/components/features/chat/drag-over.tsx:24` | `drag-over-content`, no stylesheet/selector consumer found |
| `src/components/features/home/repo-selection-form.tsx:154` | `max-w-auto`, no emitted CSS |
| `src/components/features/home/workspace-selection-form.tsx:202` | `max-w-auto`, no emitted CSS |
| `src/utils/form-control-classes.ts:39` | `ease`, no emitted CSS |

Do not blindly accept the linter's `min-w-auto` suggestion: minimum width is a different property. Adoption leaves existing UI behavior alone.

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

## Checks

Executed with Node 22.23.2, npm 10.9.8 on macOS (CI's supported Node environment should also run its normal checks):

- `npm run lint`: passed, **0 errors / 351 warnings** (347 existing + 4 new genuine findings).
- `npm run build`: passed.
- `npm run build:lib`: passed.
- `npm test -- --maxWorkers=2`: **733 files passed / 1 failed; 7,718 tests passed / 1 failed / 7 todo**. The existing `DeleteProfileModal > calls deleteProfile and shows success toast on successful delete` assertion expected a success-toast call but observed zero. No runtime source changed in this branch. Rerunning that file alone passed all 12 tests; the full-suite failure remains disclosed.
- Initial sandboxed test attempt was stopped because launcher tests could not bind local ports. Results above are from the rerun with port access.

The pinned rule delegates some missing color-token diagnostics to `no-raw-colors`; it is not a complete missing-token guard, nor does it detect the semantic `text-base` collision. Other rules retain their original settings.

This directory is PR-only evidence. Fork PRs require manual `.pr/` cleanup before merge.
