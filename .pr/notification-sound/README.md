# Notification sound on opening a conversation: running-app evidence

## Setup (same for before and after)

- Canvas: this checkout, launched with the repository's verification tool
  (`control-openhands launch --new`), which starts the shipped
  `bin/agent-canvas.mjs` CLI with a real local Agent Server/SDK 1.63.1 and
  Chromium at 1440x1000. `control-openhands doctor` passed on both runs.
- Before: `main` at `bb526a3bd9fa96518c00f4a426e2a4982dfbe6dc`.
  After: the fix commit `264816067185113ec9f190856958a97c67fc81cf`.
- LLM: the repository's own mock LLM server
  (`tests/e2e/mock-llm/scripts/mock-llm-server.py`, openhands-sdk 1.63.1),
  saved as an ordinary LLM profile. The bug does not depend on the model.
- Arrange: two conversations ("First conversation", "Second conversation")
  run to `finished`; then Settings > Application > Sound Notifications is
  turned on and saved (`enable_sound_notifications: true` in `/api/settings`).
- Sound is observed with `control-openhands browser media`, the verification
  skill's documented check for `F16.sound`: it records each
  `HTMLMediaElement.play()` call and lets playback happen.

## Steps

1. `browser goto /`, `browser media --clear`, start recording.
2. Click "First conversation" in the sidebar, wait 5 s, read `browser media`.
3. Click "Second conversation" in the sidebar, wait 5 s, read `browser media`.
4. Stop recording. Separately: `browser media --clear`, `browser reload`,
   wait 8 s, read `browser media`.

Neither conversation is running. Nothing happens in them during the steps.

## Before (`before.mp4`, `before.png`)

- Step 2: one `notification-*.mp3` play (`before-media-1.json`).
- Step 3: a second play (`before-media-2.json`).
- Step 4 (reload of an open finished conversation): one play.

The sound fires every time an already idle or finished conversation is
opened or reloaded, so it no longer means "the agent needs you".

## After (`after.mp4`, `after.png`)

- Steps 2 and 3: `plays: []` (`after-media-1.json`, `after-media-2.json`).
- Step 4: no play.
- Control (`after-live-finish.mp4`, `after-live-finish-media.json`): with the
  second conversation open, a follow-up message is sent through the composer
  (`conversation send ... --wait`); when the agent goes from running to
  finished, exactly one play is recorded. The intended notification still
  works.

## Limitations

The recordings are video only; the audio is shown through the play log
above, with timestamps matching the clicks in the videos.
