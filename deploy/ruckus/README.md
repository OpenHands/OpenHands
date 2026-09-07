# Ruckus deployment preparation

**Status: local preparation only. Not deployed or safe for employee access.**

Target project: `cohesive-mark-500400-k3` (number `799294352663`).
Administrator: `jake@sixtyoneeighty.com`.
Intended hostname: `ruckus.sixtyoneeighty.com`.

The hostname already points to a Vercel deployment. These files do not alter
Cloudflare DNS or Vercel. Do not cut over the live hostname until authentication,
the application, and rollback have been tested.

## What is prepared

`main.tf` describes a separate private staging deployment: one e2-standard-2 VM,
20 GB boot disk, protected 30 GB state disk, a dedicated network, an HTTPS load
balancer with IAP, and a DNS-authorized certificate. It grants only the named
administrator access to the IAP backend. These resources incur ongoing charges
when provisioned; no resources have been provisioned by this preparation.

Inbound application traffic is limited to Google load-balancer source ranges.
SSH is limited to IAP source ranges. The VM service account receives no project
roles or OAuth scopes. `startup.sh` prepares Docker and the state disk, but does
not start an application or copy credentials. Container access to link-local
metadata addresses is blocked.

The backend gateway is in the sibling `software-agent-sdk` checkout on branch
`codex/ruckus-managed-access`:

- `openhands-agent-server/openhands/agent_server/iap_gateway.py`
- `tests/agent_server/test_iap_gateway.py`
- `deploy/iap-gateway.Dockerfile`

It verifies signed Google IAP assertions with the exact backend audience and
administrator email for HTTP and WebSockets. Unsigned headers, backend session
keys alone, and other identities cannot grant access. Only a minimal health
endpoint is public. This is an administrator-only staging gate, not employee
authorization. The gateway image contains no company configuration or secrets.

## Required before provisioning

1. Approve enabling the IAP and Certificate Manager APIs, provisioning the
   described billable resources, and granting the named administrator IAP access.
2. Configure an external Google OAuth web client for IAP. The project is owned
   by a Google organization different from the configured administrator's email
   domain; Google's default organization-only client must not be assumed to work.
   Follow https://docs.cloud.google.com/iap/docs/custom-oauth-configuration.
3. Supply `iap_oauth_client_id` and sensitive `iap_oauth_client_secret` locally.
   Do not paste the secret in chat or commit it. Terraform/OpenTofu state contains
   the OAuth secret even though CLI output redacts it. Keep state and saved plans
   private, with restricted file permissions and protected backups.
4. Run `tofu plan` and review actual changes before applying. `tofu validate`
   checks configuration syntax and provider schema, not account permissions,
   domain access, quotas, cost, or runtime behavior. No authenticated plan or
   apply has been completed.

The certificate validation CNAME output is separate from the existing app's DNS
record. The `future_dns_cutover_address` output must not be published yet.

## Required before starting the staging application

Build the current customized Canvas for Linux amd64 with base path `/` and the
repository's pinned Agent Server/automation versions. Build the sibling gateway
with `deploy/iap-gateway.Dockerfile` for Linux amd64. Install both local images on
the private VM; `compose.yaml` expects `ruckus-canvas:staging` and
`ruckus-iap-gateway:staging`.

Import the current settings privately only after checking the running app's
latest configuration. Include the matching encryption material for encrypted
credentials. Do not indiscriminately copy the Mac home directory, conversation
history, or unrelated authentication files. Set the state directory ownership
to match the Canvas image's runtime user. This migration has not been performed.

Create a root-readable `gateway.env` using the `gateway_environment` output.
Start Compose only after the external sign-in client, IAP audience, certificate,
and settings import are ready. Test administrator login, non-administrator
denial, direct-origin denial, conversation streaming, and restart persistence.
Keep a verified backup and document rollback before any hostname cutover.

## Employee lockdown work still required

The existing all-in-one Agent Server is a single trust domain. Agents execute
terminal/file tools under an account that can read the stored provider and MCP
credentials. Read-only settings buttons or API write restrictions do not prevent
an employee from asking an agent to read those files.

Employee rollout therefore requires isolated employee runtimes and a separate
credential broker outside those runtimes. Server-side policy must choose the
approved LLM profile, skills, tools, MCP destinations, and workspace; reject
configuration overrides and secret export; and scope conversations, files,
events, sockets, and automation actions to the authenticated employee. Broker
credentials need constrained scope and revocation, and the broker must not expose
upstream keys in responses or logs. Administrator settings changes must remain
restricted to the administrator, including changes attempted through agent tools.

Those components are not implemented by this staging gateway. Do not broaden
its email check or IAP membership to employees as a shortcut. Employee onboarding
and bypass/isolation tests are required before calling the application locked.

## Local verification completed

- SDK `make build` completed.
- Gateway suite: 19 passing tests, including real ES256/RS256 signature checks,
  forged-signature and wrong-audience rejection, and an authenticated WebSocket
  relay to a real local socket server.
- Ruff and SDK `git diff --check` passed.
- Linux amd64 gateway container built and started; its health endpoint responded
  and four unauthenticated application/API routes returned 401.
- OpenTofu provider initialization and `tofu validate` passed; startup script
  passed `bash -n`.

These checks do not establish real Google sign-in, employee isolation, provider
generation, settings migration, or deployed availability. The existing frontend
was not changed or rebuilt during this preparation.
