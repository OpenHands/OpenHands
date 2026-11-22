# TLS-edge consumer check transcript (#19945)

Compares the stock and fixed packaged stacks behind a local TLS-terminating
edge. The editor proxy handles token exchange and rejects unauthenticated or
mismatched-Origin upgrades before forwarding to code-server. Successful
upgrades exercise both consumers in sequence.

## Stack and byte binding

Both variants run the same image:
`ghcr.io/openhands/agent-canvas:1.43.0`
(local Docker image ID `sha256:bcdbb3fb9610f14eda67f39fc609f162ad8c3c28697d6ff3f4adae55c15c3f1d`),
started with the documented README options (`AGENT_CANVAS_ALLOW_LAN_SESSION_KEY=true`,
a state volume, a projects directory). The packaged consumers are the pinned
Agent Server `1.63.1` editor proxy (`code_server_proxy.cjs`) and the packaged
code-server `4.145.0` behind it.

- **Fixed variant** — container `oh19945-tls`
  (`550f73ced1df…`, started 2026-11-22T11:44:35Z), mounting the two proxy
  scripts read-only under `/opt/agent-canvas/`. In-container SHA-256, captured
  after the runs:
  - `static-server.mjs` = `9bbc2450…a08cfe` (git blob `6d6f3afee88d9778`)
  - `proxy-utils.mjs` = `e2cb19ad…2b838e` (git blob `f7d18e7720fb85b5`)
- **Stock variant** — container `oh19945-stock`
  (`4f70d035e9bf…`, started 2026-11-22T06:24:04Z), no script mounts; it runs
  the image's own bytes (`static-server.mjs` = `4a09c7fc…34473a67`,
  `proxy-utils.mjs` = `e064a4b8…b71363c`).

Session credentials are ephemeral values minted by these local containers;
they are redacted below.

## Edge harness

A local Node TLS terminator (TLS 1.3, `TLS_AES_256_GCM_SHA384`, self-signed
`CN=localhost`, SAN `DNS:localhost, IP:127.0.0.1`) fronts each variant's
published port. The original matrix used three policies in `tls-edge.cjs`;
the separate `tls-edge-noxfp.cjs` variant added the fourth policy:

| Style         | `Host`            | `X-Forwarded-Host` | `X-Forwarded-Port` | `X-Forwarded-Proto` |
| ------------- | ----------------- | ------------------ | ------------------ | ------------------- |
| hostonly      | authority w/ port | _(absent)_         | edge port          | `https`             |
| authoritative | authority w/ port | authority w/ port  | edge port          | `https`             |
| hoststyle     | host, no port     | host, no port      | edge port          | `https`             |
| noxfp         | authority w/ port | _(absent)_         | _(absent)_         | `https`             |

Harness files (SHA-256): `tls-edge.cjs` `9433f43b…a2b6781`,
`tls-consumer-check.mjs` `daefb542…2d6213`, `cert.pem` `fb4f5255…a62a752f`.
Node `v22.23.2` on macOS. Commands for the original matrix:

```sh
node tls-edge.cjs <upstream-port> <edge-port> <style> <wirelog>.jsonl
node tls-consumer-check.mjs https://localhost:<edge-port> <session-key> <variant>-<style>
```

## Procedure (identical per run)

1. Exchange the advertised editor token over HTTPS:
   `GET /vscode/?tkn=<session-key>&folder=workspace` (navigation-like, no
   `Origin` header) — expect `303` plus the editor cookie.
2. `GET` the exchange target with that cookie — expect `200` (workbench HTML).
3. WebSocket upgrade `/vscode/?reconnectionToken=…` with the cookie and the
   matching `Origin: https://localhost:<port>` — the combined consumer
   decision (editor proxy **and** code-server must both accept).
4. Same upgrade with a mismatched `Origin: https://evil.example` — must stay
   rejected.
5. Same upgrade with the matching `Origin` and **no** cookie — must stay
   rejected.

## Results

| Variant \ Edge style     | hostonly | authoritative | hoststyle |
| ------------------------ | -------- | ------------- | --------- |
| fixed (captured scripts) | **101**  | **101**       | 401       |
| stock 1.43.0             | **401**  | 101           | 401       |

Every upgrade that returned 101 passed both consumers' checks; the client
connection to the edge used TLS 1.3. Every mismatched-Origin and every
missing-cookie attempt returned 401 at the editor proxy, before reaching
code-server, on both variants under all three edge styles. These requests
isolate the Origin and cookie checks only where the authenticated,
matching-Origin control returned 101. The harness records statuses; its
exit code alone does not assert that a case passed.

Cookie behavior (pinned Agent Server 1.63.1): every exchange returned `303`
with

```
Set-Cookie: openhands_editor_75c8c72e0ec6f1e8=<redacted>;
  Path=/vscode; HttpOnly; Secure; SameSite=None
```

—the `Secure` attribute is chosen from the forwarded protocol (`https`), the
cookie was accepted for the follow-up workbench request (200, 5787 bytes) and
for each successful upgrade. In those successful configurations, removing the
cookie from the matching-Origin upgrade changed the result to 401.

### Reading the matrix

- **hostonly** exercises the Host-based proxy path in #19945: the
  browser-facing authority travels in `Host` and `X-Forwarded-Host` is absent.
  In this run the edge also set `X-Forwarded-Port` to the edge port; the
  checked-in static-server fixture asserts the emitted port when this header
  is absent. The packaged `noxfp` run below checks that the real stack connects
  with that edge configuration.
  Only the fixed variant connects (101 vs 401); on stock,
  httpxy rewrites `Host` to the loopback upstream and nothing re-supplies the
  client authority, so the editor proxy's Origin comparison fails.
- **authoritative** is an edge that itself sets `X-Forwarded-Host` (common
  nginx best practice): both variants connect. The outer proxy supplies the
  header, which is exactly the masking the issue text describes ("no outer
  proxy supplying that header" is the failure precondition); the fix keeps
  this path working rather than changing it.
- **hoststyle** simulates `proxy_set_header Host $host` on a **non-default**
  port: the serialized authority loses the port, so no Origin with a port can
  match — both variants fail. For this setup the edge must forward the
  browser-facing authority including the port. A separate
  `X-Forwarded-Port` does not change the editor proxy's Origin comparison.

### Edge wire log (excerpts, fixed/hostonly)

```json
{"kind":"request","path":"/vscode/?tkn=<redacted>&folder=workspace",
 "host":"localhost:19449","origin":null,"cookie":null,"xfh":null,
 "xfp":"19449","xfproto":"https"}
{"kind":"upgrade","path":"/vscode/?reconnectionToken=t&reconnection=false&skipWebSocketFrames=false",
 "host":"localhost:19449","origin":"https://localhost:19449","cookie":"present",
 "xfh":null,"xfp":"19449","xfproto":"https"}
```

### static-server stderr (fixed variant)

Every 401 upgrade logged one redacted line (no token, no query, no upstream
URL); the stock variant logged nothing (the logging is part of this change):

```
[static:8000:editor] WebSocket upstream rejected /vscode/: HTTP 401
```

## No-X-Forwarded-Port run (scripts from `51f8fcd`)

After the matrix above was captured, the production scripts changed once more
(proxy-error label retention, reviewer-added regression tests); the outgoing
head at capture was `51f8fcd67838…`. A fresh fixed container `oh19945-r15`
(`67eea1c4…`, started 2026-11-22T16:59Z) mounted the outgoing scripts
read-only under `/opt/agent-canvas/`; in-container SHA-256 after the runs:

- `static-server.mjs` = `67e01367…cd78d2c` (outgoing head bytes)
- `proxy-utils.mjs` = `d1b06403…bccf00` (outgoing head bytes)

The stock control reused container `oh19945-stock` (image bytes, hashes
above). A variant of the TLS terminator and the same five-step consumer check ran
with a new edge policy `noxfp` — `Host` carries the browser authority with
port, `X-Forwarded-Proto: https`, and **both** `X-Forwarded-Host` and
`X-Forwarded-Port` absent. The checked-in fixture asserts that the static
server derives the forwarded port from `Host`. The packaged result cannot
isolate that derivation: code-server does not append `X-Forwarded-Port`
when the forwarded authority already has a port. Harness
variant `tls-edge-noxfp.cjs` SHA-256 `29058d43…a56e3c7`; edge ports 19460
(fixed) and 19461 (stock). Sanitized receipts: [fixed](https://raw.githubusercontent.com/Bensonluo/OpenHands/6b3032528792eafb8131ec8e4dabae4fdbaf18e9/.pr/editor-ws-origin/tls-edge-noxfp-receipt-fixed.redacted.json)
and [stock](https://raw.githubusercontent.com/Bensonluo/OpenHands/6b3032528792eafb8131ec8e4dabae4fdbaf18e9/.pr/editor-ws-origin/tls-edge-noxfp-receipt-stock.redacted.json).
Cookie and token values are redacted; statuses are unchanged. The receipts
record request starts at `2026-11-22T17:07:09Z` (fixed) and
`2026-11-22T17:07:30Z` (stock). Original wire logs and harness scripts are
local captures, not files in this checkout.

| Variant                     | exchange | follow-up HTTP | matching Origin + cookie | mismatched | no cookie |
| --------------------------- | -------- | -------------- | ------------------------ | ---------- | --------- |
| fixed (outgoing head bytes) | 303      | 200            | **101** (TLS 1.3)        | 401        | 401       |
| stock 1.43.0                | 303      | 200            | 401                      | 401        | 401       |

Both edge wire logs show no forwarded port or host: `"xfp":null`,
`"xfh":null`, with the corresponding edge port in `Host`. The fixed
container logged two upstream refusals, both without the query string;
the stock container logged no such lines. The conversation-level editor
launch on the same container (Canvas UI → panel → Open in VS Code) opened
the workbench with no error dialog. The browser capture used the direct HTTP
path, not the TLS edge.

## Boundaries

Self-signed local certificate on loopback; the harness did not verify the
certificate. This checks the proxied application behavior over TLS, not TLS
peer verification. The no-X-Forwarded-Port run used the same script bytes as
the outgoing candidate; the first three policies used the earlier script
hashes. Deployment-specific ingress policies, custom editor upstreams, and
external stderr parsers were not exercised. The excerpts omit session keys and
cookie values. The successful exchange redirects to a URL without the token
query parameter.
