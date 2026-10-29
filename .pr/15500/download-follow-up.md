# Download review follow-up

Both download entry points now share `downloadRuntimeFile`, which releases the typed `FileClient` in `finally`. The workspace helper still resolves relative paths and snapshots the selected backend before asynchronous work. The older runtime helper keeps its existing signature and Cloud URL guard.

## Regression proof

Before the change, the runtime-service tests failed on successful and failed downloads because `close()` was never called. After the change, all 49 tests passed:

```sh
LANG=en_US.UTF-8 npx vitest run \
  __tests__/api/runtime-service/agent-server-runtime-service.test.ts \
  __tests__/api/conversation-file-download.test.ts \
  __tests__/routes/files-tab.test.tsx \
  src/api/no-direct-agent-server-calls.test.ts
```

## Real runtime check on 2026-10-07

The full unit suite passed with 8,078 tests and 7 todo across 765 files. Lint and both application/library builds passed.

The production Canvas build was served by `scripts/static-server.mjs` on port 3310, forwarding API and WebSocket requests to port 3302. The backend was the pinned `ghcr.io/openhands/agent-server:1.50.1-python` image, with port 8000 exposed only on loopback. Authentication used an isolated test key.

An idle conversation used `LocalWorkspace` at `/workspace/download-review` inside this Docker container. No LLM request was sent: file downloads do not require model execution. This checks a Docker-hosted Agent Server; the earlier DockerWorkspace dispatch check remains documented in the PR description.

The real Files toolbar downloaded both fixture files. An observer retained the Blobs handed to `URL.createObjectURL` and the anchor filenames without replacing the request or download behavior. Their hashes matched `sha256sum` inside the container:

| Filename | Bytes | SHA-256 |
| --- | ---: | --- |
| `café data.bin` | 1024 | `785b0751fc2c53dc14a4ce3d800e69ef9ce1009eb327ccf458afe09c242c26c9` |
| `download-example.txt` | 55 | `2cc74222d0f04530f7f950b3f0f06a16cb82ecf3e96fb50854ee6539c90bc1b8` |

Renaming the binary fixture inside the test container made the next click return 404. Canvas displayed "Could not download the file. Try again." and the observed download count remained at two. The fixture was restored. The browser host disconnected before a successful retry could be verified; no retry-success claim is made for this run. The check compares bytes at the browser download boundary, not an OS-saved file.

[Toolbar recording](05-docker-download.mp4) and [screenshot](06-docker-download.png).

## Remaining evidence gap

Live OpenHands Cloud remains unverified. No Cloud backend was configured in the available browser profile and no existing Cloud test conversation was available. Unit tests cover provisioned runtime URL/session-key routing and rejection when a Cloud runtime is unavailable; those tests do not replace live Cloud evidence.
