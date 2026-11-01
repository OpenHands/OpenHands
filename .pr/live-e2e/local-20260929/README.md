# OpenHands PR 16319 — real settings UI evidence

Recorded on 2026-09-29 (Asia/Shanghai) for https://github.com/OpenHands/OpenHands/pull/16319.

These are unedited, continuous Playwright browser recordings of the actual local frontend connected to the real Agent Server, with no mocked local API responses. They were recorded by Codex automation, not claimed as human manual testing.

- Before: frontend commit fad39b8f0b0451f765b1ff64331cbfd2dfae1d1d.
- After: frontend commit 01f135dd0dabfdbf5ed159118eb8383ce97be32c.
- Both use Agent Server / SDK 1.49.5, Node 24, Playwright 1.63.0, Chromium build 1243 on Linux.
- Before: Max size has no min/step; saving 19 sends PATCH requests rejected with HTTP 422. Saving 20 succeeds with HTTP 200.
- After: min=20 and step=1; Save remains available for 19, then displays "Max size must be at least 20" with no condenser PATCH. Saving 20 succeeds with HTTP 200.
- Both verify the rejected value did not replace 240, then verify 20 by a real GET and page reload.
- Browser external fonts/telemetry/version checks were blocked. No model conversation was created and no LLM credentials were provided. Backend settings traffic was not mocked.

The PNG files are direct browser screenshots of invalid 19 and valid 20 during the corresponding run. Each WebM decoded fully with no errors. These are PR-only artifacts on the implementation branch, following the repository .pr convention. The directory name identifies a local recording date, not a GitHub Actions run. Remove these files before merging the implementation; immutable commit URLs preserve the review evidence.

## Files

- before.webm: 1171745 bytes; SHA256 773d6282da283ff408a0d99dcba69be09e03d03e4d5357344fa78daecb00d787
- before-invalid-19.png: 114977 bytes; SHA256 c5298bc02708533a47706c02d8247a7340186f37cc573ffd7d0efd911c0a5802
- before-valid-20.png: 124781 bytes; SHA256 17987ac3766a2eb800cf61155656e8ef42bf9ab40bcfc71d466c5499f0d28f8f
- after.webm: 936912 bytes; SHA256 01d6f78a2cc69e569f09c6ad5560290f1af3a1c4df0d6951db9479a27d0ed952
- after-invalid-19.png: 111577 bytes; SHA256 8cf2933ce7145b70288dcac136d9f56f274768c199afa0cc1b38a70e95b18a9c
- after-valid-20.png: 118323 bytes; SHA256 48aa6a0833813838d4f27736c5d30e2d9da7b625d3939d7ca1fe78cd8a02b066
