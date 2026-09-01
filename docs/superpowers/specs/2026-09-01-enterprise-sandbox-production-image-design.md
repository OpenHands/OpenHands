# Enterprise Sandbox Production Image Design

**Status:** Approved design, pending implementation plan

**Date:** 2026-09-01

**Repository:** `OpenHands/OpenHands` (Agent Canvas)
**Scope:** Docker packaging, local container topology, build/release automation, and operator documentation

## Summary

Agent Canvas currently publishes an all-in-one Docker image that starts the Agent
Server, Automation backend, static Canvas frontend, and ingress proxy in one
container. OpenHands Enterprise runs those responsibilities in a different
topology: its control plane manages isolated sandbox pods, and each sandbox pod
runs the standard OpenHands Agent Server contract.

This change introduces a production-grade custom sandbox image for OpenHands
Enterprise and a separated local container topology. The Enterprise image will
extend the exact Agent Server image expected by the target Enterprise release,
add only sandbox-relevant tools and dependencies, and preserve the upstream
entrypoint and command without modification. Local container development will
run the Canvas/Automation/proxy control plane separately from a sandbox
container built from the same shared runtime layer.

The existing product behavior is unchanged. This is a deployment-boundary
change only: the local control plane reaches the Agent Server through a service
hostname and port instead of `127.0.0.1` in the same container.

## Goals

1. Produce an OpenHands Enterprise-compatible sandbox image from this
   repository.
2. Preserve the upstream Agent Server runtime contract, including its
   `ENTRYPOINT` and `CMD`.
3. Share the sandbox runtime layer between local container development and the
   Enterprise production image.
4. Separate the local containerized control plane from the sandbox so local
   networking, environment injection, and failure boundaries more closely
   resemble Enterprise.
5. Keep the existing one-command local workflow available through a launcher or
   Compose orchestration layer.
6. Keep the current all-in-one image available as a legacy compatibility target
   during the first migration PR.
7. Build, inspect, test, and publish the Enterprise image reproducibly for
   `linux/amd64`.

## Non-goals

- No new Canvas, Agent, Automation, API, or user-facing feature.
- No change to Agent Server API or WebSocket contracts.
- No attempt to reimplement the Enterprise runtime manager locally.
- No per-conversation Kubernetes pod lifecycle in the first local Compose
  topology; the local sandbox container may remain long-lived.
- No replacement of Enterprise application, Automation, ingress, database, or
  control-plane service images.
- No custom wrapper around the Enterprise sandbox entrypoint.
- No secrets, credentials, deployment URLs, or task-specific source changes
  baked into either image.
- No removal of the legacy all-in-one image in the first PR.

## Terminology

- **Control plane / server:** Canvas frontend, Automation backend, and ingress
  proxy. In Enterprise these are platform-managed services. In the local
  container topology they run in a repository-built control-plane container.
- **Pod / sandbox:** The isolated environment in which an agent edits files and
  runs commands. It runs the OpenHands Agent Server process.
- **Agent Server:** The process and API inside the sandbox. It is not the same as
  the Enterprise control-plane server.
- **Legacy all-in-one image:** The existing repository image whose custom
  entrypoint starts Agent Server, Automation, Canvas, and proxy together.

## Architecture

### Target topology

```text
Local container development

Browser
  |
  v
Control-plane container
  - Canvas static frontend
  - Automation backend
  - ingress/proxy
  |
  | HTTP + WebSocket on a private container network
  v
Sandbox container
  - upstream Agent Server entrypoint
  - shared tools and dependencies


OpenHands Enterprise

Enterprise-managed control plane
  |
  | Enterprise runtime protocol
  v
Sandbox pod
  - the same shared tools and dependencies
  - upstream Agent Server entrypoint
```

The external browser URL and public Canvas behavior remain unchanged. Only the
internal Agent Server address changes from an in-container loopback address to a
configured sandbox service address.

### Docker build graph

One multi-target Dockerfile is the source of truth:

```text
official pinned agent-server image
              |
              v
      agent-runtime-base
      - sandbox tools
      - sandbox dependencies
      - safe shared ENV defaults
      - OCI metadata
          |           |
          v           v
   dev-sandbox   enterprise-sandbox
                     - no ENTRYPOINT
                       or CMD override

frontend-build + automation dependencies
              |
              v
       control-plane image
       - Canvas
       - Automation
       - proxy entrypoint

shared/legacy stages
              |
              v
       legacy all-in-one target
```

The final stage may remain the legacy target initially so existing unqualified
`docker build` consumers do not change behavior. New build helpers and CI must
name their intended target explicitly.

### Enterprise sandbox contract

The `enterprise-sandbox` target must:

1. use a pinned `ghcr.io/openhands/agent-server:<version>-python` base image;
2. match the Agent Server major and minor version required by the deployed
   Enterprise release;
3. inherit the base image's `ENTRYPOINT` and `CMD` unchanged;
4. contain only sandbox-relevant additions from this repository;
5. run as the user expected by the upstream image;
6. build for `linux/amd64`;
7. expose no Canvas, Automation, SQLite, local-secret generation, or ingress
   runtime behavior; and
8. include OCI labels that identify the Canvas source revision, Canvas version,
   and Agent Server base version.

The Enterprise Admin Console's default Sandbox Image Tag is authoritative for
the required Agent Server version. The image must be rebuilt for each Enterprise
upgrade rather than relying on a floating tag.

### Entrypoint ownership

The pod/sandbox image has no repository-owned entrypoint. It inherits and runs
the official Agent Server image entrypoint.

The local control-plane image retains a repository-owned entrypoint only because
it starts multiple local services: Automation and the Canvas static
server/proxy. It must not start an Agent Server process. If those control-plane
processes are split into individual containers in a later change, that
orchestration entrypoint can also be removed.

The legacy all-in-one target continues using the existing entrypoint during the
migration period.

## Environment contract

Environment similarity is achieved through shared canonical variables and
explicit injection, not through a shared entrypoint.

### Common sandbox variables

Only non-secret, sandbox-relevant defaults may be declared in
`agent-runtime-base`. Examples include a shared tool import path or stable
workspace/tool configuration. A common default must be valid in both local and
Enterprise sandboxes without knowing the surrounding control-plane topology.

### Platform-owned sandbox variables

Runtime identity, Agent Server bind configuration, authentication, workspace
assignment, persistence, secrets, and other per-pod values are injected when the
sandbox starts. Compose owns these values locally; Enterprise owns them in
production. They are not generated by the image.

### Dev-only control-plane variables

The following concerns remain in the local control plane and must not leak into
the Enterprise sandbox image:

- Canvas and proxy ports/base paths;
- Automation ports, database, file storage, callbacks, and API keys;
- `LOCAL_BACKEND_API_KEY` convenience handling;
- public-mode test server configuration;
- local secret generation and persisted local state;
- frontend and Automation telemetry configuration; and
- local runtime-service discovery inputs.

Local convenience names may remain supported, but the control-plane entrypoint
must translate them once into canonical variables. Internal components should
not each invent their own aliases or fallback rules.

### Secret policy

Secrets must be injected at runtime. They must not appear in Docker build args,
image layers, labels, generated defaults, CI artifacts, Compose files committed
to the repository, or logs. Local generated secrets remain local control-plane
state and are never copied into the Enterprise sandbox image.

## Networking and data flow

The separated local topology uses a private container network:

1. The browser connects only to the control-plane ingress port.
2. The control plane receives an explicit internal Agent Server URL, such as
   `http://agent-server:8000`.
3. Proxy routes for `/api`, `/sockets`, Agent Server health/docs endpoints, and
   the editor path use that explicit URL.
4. Automation receives an explicit URL and credential for its calls to the
   Agent Server.
5. The sandbox receives the Automation/ingress URL that is reachable from the
   sandbox's own network perspective.
6. `runtime_services` is rendered from these explicit directional URLs. A
   control-plane-to-sandbox URL must not be reused as a
   sandbox-to-control-plane URL.

Separated mode has no silent `127.0.0.1` fallback for the Agent Server. Missing
or invalid URLs cause a clear startup error. The control plane waits for the
sandbox with a bounded readiness timeout. If the sandbox later fails, the
control plane remains alive and proxied routes return a clear unavailable/502
response; container restart policy may recover the sandbox independently.

These rules affect only internal host and port selection. Browser-visible routes,
payloads, authentication headers, APIs, WebSockets, and product behavior remain
unchanged.

## Local workflow and compatibility

A Compose file or equivalent launcher starts the control-plane and dev-sandbox
targets together. It creates the private network, injects explicit directional
URLs and runtime secrets, mounts the same persistent/project paths required by
the current local workflow, publishes only the control-plane port, and shuts
both containers down together.

The existing one-command experience remains available. The command may delegate
to Compose internally, but users must not need to manually coordinate two
containers.

The existing all-in-one image remains buildable and keeps its current default
behavior during this first migration. Existing release, CLI, and mock-LLM E2E
paths continue using it until a later, separately reviewed migration changes
their default topology.

## Build and release

### Local builds

Build tooling must expose explicit modes or targets for:

- legacy all-in-one image;
- local control-plane image;
- local dev-sandbox image; and
- Enterprise sandbox image.

The Agent Server image is resolved from `config/defaults.json` by default. An
explicit override is supported for an Enterprise release that expects a
different compatible tag. Build output must print the resolved base image,
target, platform, and destination tag without printing secrets.

### CI builds

CI continues building the legacy multi-architecture image. It additionally
builds the Enterprise sandbox for `linux/amd64` and verifies its image contract.
Same-repository PRs may receive PR/SHA image tags consistent with existing
repository policy.

The proposed repository is:

```text
ghcr.io/openhands/agent-canvas-enterprise-sandbox
```

Stable releases use immutable, compatibility-explicit tags:

```text
<canvas-version>-agent-server-<agent-server-version>
```

For example:

```text
1.16.0-agent-server-1.44.1
```

The workflow does not publish an unqualified `latest` tag for this image. A
manual workflow dispatch may override the Agent Server base image; the resolved
version must be reflected in the output tag, OCI labels, and build metadata.

### Enterprise configuration

Operator documentation covers:

1. selecting the Agent Server tag expected by the Enterprise Admin Console;
2. building and pushing the image for `linux/amd64`;
3. configuring repository, tag, and registry credentials under Sandbox Image;
4. deploying the updated Enterprise configuration; and
5. rebuilding the image for every Enterprise upgrade.

## Testing strategy

Implementation follows test-driven development.

### Static and unit contract tests

Tests must fail before implementation and then verify that:

- all required Docker targets exist;
- target relationships share the intended runtime base;
- the Enterprise target cannot acquire Canvas, Automation, proxy, SQLite, local
  secret, or repository-owned entrypoint content;
- build scripts resolve the target, Agent Server image, platform, and tag
  deterministically;
- separated-mode URL construction has no loopback fallback; and
- runtime-service URLs preserve their two network perspectives.

### Image inspection

After building the upstream base and Enterprise target, automated inspection
compares their effective `Config.Entrypoint` and `Config.Cmd`. It also verifies
the expected architecture, labels, user, and absence of control-plane artifacts.

### Compose smoke test

A bounded smoke test starts the separated stack and verifies:

- control plane and sandbox become healthy;
- Canvas is reachable on the existing external port/base path;
- HTTP and WebSocket proxy routes reach the separate sandbox;
- Automation can reach the same sandbox with the configured authentication;
- the generated runtime-service metadata contains the correct directional
  service URLs; and
- stopping or restarting the sandbox does not terminate the control plane.

### Existing quality gates

The final validation set includes, at minimum:

```text
npm run lint
npm test
npm run build
npm run build:lib
```

Relevant Docker builds, image inspections, and the separated-stack smoke test
are additional gates. Existing mock-LLM E2E coverage remains on the legacy
all-in-one path for this first migration unless implementation reveals a small,
non-disruptive way to exercise both targets without changing test semantics.

## Acceptance criteria

1. A documented command builds an Enterprise-compatible sandbox image for
   `linux/amd64`.
2. The image is based on an explicit Agent Server version and preserves its
   effective entrypoint and command exactly.
3. The Enterprise image contains no Control Plane, Automation, local storage,
   local secret generation, or custom orchestration runtime behavior.
4. Local control-plane and sandbox containers can run as a separated stack using
   the existing public Canvas port and routes.
5. The separated stack has no silent loopback fallback for cross-container
   communication.
6. Local and Enterprise sandbox targets share one sandbox runtime layer and the
   same resolved Agent Server base image.
7. The existing all-in-one build remains functional and remains the compatibility
   default during the first migration.
8. CI builds and inspects the Enterprise image and can publish immutable,
   version-explicit tags.
9. Operator documentation explains the Enterprise Admin Console configuration
   and upgrade/version compatibility requirement.
10. Existing product behavior and API contracts are unchanged; the authored
    change is limited to image, environment, network, process, test, and release
    boundaries.

## Risks and mitigations

### Enterprise version mismatch

An Enterprise release rejects incompatible Agent Server versions. Builds use an
explicit base image and compatibility-explicit output tag; documentation makes
the Admin Console's expected tag authoritative.

### Hidden loopback assumptions

Existing all-in-one code may assume `127.0.0.1`. Contract tests and the Compose
smoke test exercise explicit service URLs and reject separated-mode fallbacks.

### Entrypoint drift

A future refactor could accidentally add a wrapper or inherit the local
entrypoint. Image inspection compares the effective Enterprise entrypoint and
command with the pinned upstream image.

### Oversized sandbox image

Only sandbox-relevant tools and dependencies enter the shared runtime layer.
Frontend, Automation, proxy, and their dependencies remain outside the
Enterprise target.

### Migration regression

The legacy all-in-one target remains the default during the first PR. Existing
tests and release paths continue to validate it while the separated topology
receives focused new coverage.

## Follow-up candidates

The following are intentionally deferred:

- switch the default Docker/CLI distribution from all-in-one to the separated
  topology;
- migrate mock-LLM production-fidelity E2E fully to the separated topology;
- emulate one disposable sandbox per conversation locally;
- split Automation and Canvas/proxy into separate control-plane containers; and
- remove the legacy all-in-one target and entrypoint.
