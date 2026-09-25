# Base-color / font-size migration verification

Base: `a3cfe98267c52937dc4e91a55644a6131c6c01bc`. Code head: `c5f86b2fbbb4efeeb850ce61206a8f3750c63bd4`.

## Reproduce the CSS bug and fix

`npx vitest run __tests__/themes/tailwind-text-base.test.ts` was run first with the test added to the original source: the emitted `text-base` rule contained `color: var(--oh-color-base)` and **no font-size**, so the regression assertion failed. After migration, both tests pass. The tests compile the real stylesheet, including the Tailwind config and plugins, and exercise the actual Canvas `cn` helper.

`node .pr/verify-theme.mjs` compares the original stylesheet (read from git) with this branch. Output:

```text
PASS: background/opacity/hover/ring/gradient/text-color CSS is identical after renaming.
text-base after: font-size: var(--text-base);
    line-height: var(--tw-leading, var(--text-base--line-height));
```

The intentional behavior change is that font-size consumers of `text-base` now receive its real 1rem size/default line height instead of a color. Reject-action text uses `text-canvas-base` so its separate `text-sm` size is no longer discarded by merging. Fixed 16px chat/onboarding inputs keep their explicit sizes and scoped suppressions.

## Browser checks

Started independent real Canvas frontends with mock backend data, telemetry disabled:

```sh
VITE_DO_NOT_TRACK=1 npm run dev:mock -- --host 127.0.0.1 --port 3196 # base UI
VITE_DO_NOT_TRACK=1 npm run dev:mock -- --host 127.0.0.1 --port 3195 # changed UI
```

Connected each app's suggested local mock backend, skipped LLM setup, opted out in the consent dialog, and opened the home screen. Captures use the default desktop viewport (1280×720) and a 390×844 mobile override. No actual credentials or live LLM were used.

| View | Before | After |
| --- | --- | --- |
| Home / chat input | [before](home-before.png) | [after](home-after.png) |
| Mobile home / chat input | [before](home-mobile-before.png) | [after](home-mobile-after.png) |

The input's computed styles were **font-size 16px, line-height 20px, color rgb(255,255,255)** before and after. A temporary browser root-font override to 20px produced **16px / 25px / rgb(255,255,255)** on both, confirming why replacing the fixed size with 1rem would be a behavior change. The override and mobile viewport were restored. See [root-font-evidence.json](root-font-evidence.json).

The screenshots are smoke evidence, not pixel-diff certification: mock activity timestamps and asynchronous sections can differ. All route/theme combinations, Cloud execution, and third-party embedding hosts were not exercised. Compiled-color parity and the unchanged `--oh-color-base` variable cover the mechanical color mapping; standard text sizing is deliberately restored.

## Checks

Node 22.23.2 / npm 10.9.8 on macOS. The repository declares Node >=24; CI must still verify its supported environment.

- `npm run lint`: passed with the existing 347 warnings, zero errors.
- `npm run build`: passed.
- `npm run build:lib`: passed.
- `npm test -- --maxWorkers=2`: **735 files passed; 7,721 tests passed; 7 todo**.
- Focused regression: two tests passed after the fix; original CSS failed the missing-font-size assertion.
- The first full test run on the separate lint-only branch had one intermittent existing profile-toast test failure (7,718 passed). That file passed individually; the complete theme run above also passed it. No test assertion was weakened.

The worktree initially reused the lint checkout's pinned dependencies. It subsequently received its own `npm ci` installation and the app/library builds and focused regression were rerun to verify normal package output paths.

## Compatibility / cleanup

Custom users of generated `bg-base` / color `text-base` need the migration in `docs/DEVELOPMENT.md`. The underlying `--oh-color-base` and `base-secondary` tokens remain unchanged. Keeping `--color-base` as a deprecated alias would preserve the collision, so it is intentionally removed.

This is a fork PR: remove `.pr/` manually before merge. The repository workflow does not automatically clean fork branches.
