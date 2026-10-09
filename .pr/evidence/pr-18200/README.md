# Before/after video for OpenHands/OpenHands#18200

Compact section navigation on tablets (768–1023 px), shown at 820 × 1180.

| File | What it is |
| --- | --- |
| `pr-18200-tablet-navigation.mp4` | 51 s, 1920 × 1080, 30 fps, with quiet background music |
| `pr-18200-tablet-navigation.gif` | The same video, 960 px wide, 10 fps, silent (plays inline on GitHub) |
| `selector-clipped-on-secrets.png` | The selector's list clipped on `/settings/secrets` (found while recording) |
| `tools/` | The scripts that recorded and composed the video |

## Builds and stack

| | Revision | Notes |
| --- | --- | --- |
| Before | `4a8f77e17e56a4db676086b7ab4436665b49fb58` | `main`, the PR's base |
| After | `d84357bfa57384f6a39834f4f8d55e940f8969fe` | PR head (base merged in) |

Both were launched with `verify-openhands` (`control-openhands launch --new --build always`)
as isolated real stacks: Agent Server/SDK 1.53.0, automation 1.19.0, Node 24.21.0,
Playwright 1.63.0 Chromium. `doctor` passed on both; `onboard --skip`; fresh state,
English, OpenHands-Neutral theme. No LLM calls. Every "Getting started" card in the
video reads "0 complete", so both builds are in the same state.

## How it was made

1. `tools/record_scene.sh before|after settings|customize` drives each scene with
   `control-openhands browser viewport tablet`, `browser click` (with `--expect-url`
   on every navigation) and `browser record start|stop`. It logs each click's
   target box (`browser bbox`) and time to a JSONL file.
2. `tools/compose.py` builds the final cut. `browser record` captures no pointer,
   so the cursor, target outlines and click rings (red before, blue after) are
   drawn in post at the logged boxes. Click times are snapped to the frame where
   each click's effect first appears (`tools/changes.py`). Page content is the
   unedited recording.
3. `tools/make_music.py` synthesizes the music from scratch (pad, bass, bells,
   convolution reverb; no samples), mixed at −25 LUFS.

Settings is recorded as Application → LLM → Secrets in both builds. The selector
renders fully when opened on Application and LLM, but it is clipped when opened
on Secrets (see below), so the video does not open it there.

## Found while recording

At 820 px with fresh state, opening the Settings selector on `/settings/secrets`
or `/settings/agents` clips its list. The settings `<main>`
(`settingsLayoutMainScrollClassName`, `overflow-y-auto`) is only as tall as
these short pages: its bottom edge is at y = 370 (Secrets) and 354 (Agent),
while the list ends at 394. That hides the current item. `settings-screen`
(`min-h-0`, no height) does not give `SettingsLayout`'s `h-full` a definite
height. Application, LLM, Verification and all four Customize pages render
the list fully.

## Checksums (SHA-256)

```
3ad6b65d588e4e40a734618079b88d587063760eb3bc6915fc12b127ecc468b2  pr-18200-tablet-navigation.mp4
882abd080e0beda6b214798c7c36a954551e98c49fc52e116c5d8f78f1da2244  pr-18200-tablet-navigation.gif
8cf6e6847409828c513e33984c78b3d2f78dfb20047666efcd63472323aece86  selector-clipped-on-secrets.png
```
