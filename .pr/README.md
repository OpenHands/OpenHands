# OSS-13667 — Canvas base path on links that leave the app

Reviewer artifacts for [OSS-13667](https://linear.app/all-hands-ai/issue/OSS-13667/cmdclick-new-chat-in-cloud-opens-conversations-not-found).

## What the bug is

In the cloud deployment both apps share one origin: the enterprise app owns
`/`, and Agent Canvas is mounted at `/canvas`. React Router's `<Link>` generates
hrefs that are correct for client-side navigation but omit the SPA base path.
That is invisible until the browser follows the link itself instead of letting
the router intercept it — cmd/ctrl+click, middle-click, "Copy link address", or
a toast `<a>`. The link then resolves against the origin root and lands on the
enterprise app, which is the "conversations not found" in the ticket.

## Reproduce

```bash
cd /workspace/repos/openhands
VITE_BASE_PATH=/canvas VITE_MOCK_API=true npm run build:mock

# One origin, two apps: enterprise at /, Canvas at /canvas.
# The server also serves the MSW worker at the origin root, because MSW
# registers its service worker with / scope regardless of the SPA base path.
node .pr/shared-origin-server.mjs --port 13001 --dir build-fixed
node .pr/shared-origin-server.mjs --port 13002 --dir build-prefix
```

`build-fixed` is the current branch; `build-prefix` is `HEAD~1` built the same
way. Both are produced with `npm run build:mock` and copied out of `build/`.

Then:

```bash
node .pr/verify-base-path.mjs   # every in-app href, pre-fix vs post-fix
node .pr/verify-cmd-click.mjs   # the literal cmd/ctrl+click from the ticket
node .pr/capture-screenshots.mjs
```

Results are in `evidence.txt`; screenshots are in `artifacts/`.

## Result

| | pre-fix | post-fix |
|---|---|---|
| in-app hrefs missing `/canvas` | 54 / 54 | 0 / 54 |
| cmd/ctrl+click on New Chat lands on | enterprise app | Canvas |

`verify-cmd-click.mjs` performs a real `ControlOrMeta`+click and follows the
popup, so it exercises the browser's own link handling rather than the router.