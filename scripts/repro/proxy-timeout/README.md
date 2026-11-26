# Reproduce the downstream timeout listener leak

These probes run the actual `scripts/ingress.mjs` child process and the actual
`createProxyHandlers` implementation. They make no LLM requests and bind only to
loopback. They are newly recorded reproductions, not the missing source of the
older composed screenshot in PR #16841.

With Node 24.22.1 and dependencies installed in each target directory:

```sh
node scripts/repro/proxy-timeout/repro.mjs /path/to/unpatched-checkout warning
node scripts/repro/proxy-timeout/repro.mjs /path/to/patched-checkout clean
node scripts/repro/proxy-timeout/timeout-repro.mjs /path/to/either-checkout
```

`repro.mjs` sends 20 sequential requests over one verified TCP connection. It
captures and forwards child stdout/stderr, waits for the child streams to close,
and asserts the expected presence or absence of `MaxListenersExceededWarning`.
A startup failure, unsuccessful response, connection replacement or unexpected
warning result fails the command.

`timeout-repro.mjs` uses a hanging HTTP upstream and the real proxy helper with
`timeout: 300` and `proxyTimeout: 0`. It asserts `ECONNRESET` after the downstream
deadline. This isolates the helper's deadline from the CLI's fixed 120-second
setting and Node's separate after-response keep-alive timeout.

## Recorded comparison

`results-20261107.txt` is raw output from the same two scripts on macOS arm64,
Node 24.22.1, in this order:

1. Current upstream main at `9979fa2591681d0801f6b5c1d66fd87ea7230de4`.
2. This branch's working tree after merging that main (scripts and fix as committed
   alongside this log).
3. The npm package `@openhands/agent-canvas@1.15.0` referenced in the original report.

Main and the published package emit the warning; this branch does not. All three
retain the downstream deadline. For the captured run, the unchanged `scripts/`, `config/`, and `package.json`
were copied to neutral temporary directories; `node_modules` links resolve to
the installed dependencies. The source checkouts use the same unchanged lockfile
and dependencies; the npm package uses its installed dependencies.
This is an ingress/runtime regression check, not a whole-app UI or live-agent test.

The dedicated `Proxy timeout reproduction` workflow repeats the main/head
comparison on Linux and uploads raw output as `proxy-timeout-raw-output`. The
unpatched control is deliberately pinned to the main commit above, so future
merges cannot silently turn the negative control into an already-fixed version.
