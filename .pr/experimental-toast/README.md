# Failed experimental feature switch: running-app evidence

## Setup (same for before and after)

- Canvas: this checkout, launched with the repository's verification tool
  (`control-openhands launch --new`), which starts the shipped
  `bin/agent-canvas.mjs` CLI with a real local Agent Server/SDK 1.63.1 and
  Chromium at 1440x1000. `control-openhands doctor` passed on both runs.
- Before: `main` at `bb526a3bd9fa96518c00f4a426e2a4982dfbe6dc`.
  After: the fix commit `eb9dfef0a26939e7f0d8e505b5cf2aa239a00d56`.
- No LLM is involved: the Features page only talks to `/api/settings/experimental`.
- Toasts are read with `control-openhands browser toasts --history`, requests
  with `control-openhands browser network`.

## Steps

1. `onboard --skip`, `browser goto /settings/experimental`; the list loads (GET 200).
2. Start recording, then `service stop agent-server` (the verification
   skill's backend-down state). The page stays open.
3. Click "Conversation reactions" to turn it on. The PATCH fails with 502.

## Before (`before.mp4`, `before-log.json`)

Two toasts for the one failed switch:

- `11:48:29.262` "An error occurred" (`before-first-toast.png`), 3 ms after
  the PATCH 502. This is the global mutation error toast.
- `11:48:34.958` "Could not change the feature. Please try again."
  (`before-second-toast.png`), the page's own message, shown once the
  reconciling list refetch has given up.

## After (`after.mp4`, `after.png`, `after-log.json`)

Same steps, same requests (PATCH 502, then the list refetch). One toast:
`11:52:04.413` "Could not change the feature. Please try again."

Control on the after run: with the Agent Server back (`restart`), clicking the
switch turns the feature on (`GET /api/settings/experimental` reports
`enabled: true`) and shows no toast.

## Limitations

The page's message still waits for the reconciling refetch (about 5 s while
the server is down), as it did before; this change only removes the extra
generic toast.
