# Editor WebSocket origin checks — raw reproduction (before/after)

The real `scripts/static-server.mjs` is spawned as a child process with
`--route /vscode=<upstream> --vscode-base-path /vscode` (the form the Docker
entrypoint runs it in), against a mock editor upstream exercising the
agent-server editor proxy's `validOrigin` rule: when `Origin` is present,
its parsed host must match the first `X-Forwarded-Host` entry (or `Host`
if the forwarded header is absent). This mock does not test authentication
or run code-server. The handshakes use the same upstream on both sides:

| Case                   | Host                      | Origin                           | Other headers              |
| ---------------------- | ------------------------- | -------------------------------- | -------------------------- |
| b: direct HTTP         | `localhost:<static-port>` | `http://localhost:<static-port>` | None                       |
| c: mismatched Origin   | `localhost:<static-port>` | A different host                 | None                       |
| d: simulated TLS proxy | `public.example`          | `https://public.example`         | `X-Forwarded-Proto: https` |

The transcript was captured with `node /private/tmp/oh-repro-19945/repro-editor-ws.mjs <checkout-root>` (local harness, run against base `7eb4108` and the candidate). The equivalent checked-in regression cases run with `npx vitest run __tests__/scripts/static-server.test.ts -t 'proxied client identity'`.

## BASE (main @ 7eb4108, before the fix)

```
b) matching Origin (README docker run path): HTTP/1.1 401 Unauthorized
c) mismatched Origin (must stay rejected): HTTP/1.1 401 Unauthorized
d) TLS reverse proxy forwarding Host: HTTP/1.1 401 Unauthorized
static-server log: (no rejection line)
```

All three fixture requests receive 401. The reported browser symptom is close
code 1006; this command makes no browser connection.

## BRANCH (capture at `347a7a1`, with this fix)

```
b) matching Origin (README docker run path): HTTP/1.1 101 Switching Protocols
c) mismatched Origin (must stay rejected): HTTP/1.1 401 Unauthorized
d) TLS reverse proxy forwarding Host: HTTP/1.1 101 Switching Protocols
static-server log: [static:18716:editor] WebSocket upstream rejected /vscode/: HTTP 401
```

The two matching upgrades get 101. The mismatch is still rejected upstream;
the new rejection line includes the request path and status, not the query
or the configured target URL.
