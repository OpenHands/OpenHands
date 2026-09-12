# Team Access (Clerk)

Put a Clerk sign-in in front of Agent Canvas so only your team can open it.

The gate is **opt-in**. With no `VITE_CLERK_PUBLISHABLE_KEY` set, nothing
changes — `npx @openhands/agent-canvas`, the test suites, and every existing
deployment behave exactly as before.

## What this does and does not protect

> [!IMPORTANT]
> This gates the **canvas UI**, not the agent server.

Agent Canvas is a client-only SPA (`ssr: false`). It has no server of its own,
so the gate runs entirely in the browser. The agent server still authenticates
on the session API key alone (`X-Session-API-Key`, or a bearer token for
cloud — see `src/api/backend-registry/auth.ts`). Anyone who obtains that key
can talk to the agent server directly and will never meet Clerk.

| Goal | Covered? |
| --- | --- |
| Only my team can open the canvas at this URL | ✅ Yes |
| A stranger who finds the URL sees a sign-in, not the app | ✅ Yes |
| Every UI entry point is gated (onboarding, API-key entry, deep links) | ✅ Yes — the gate wraps the whole router |
| Only my team can reach the **agent server** | ❌ No — requires the agent server to validate Clerk JWTs |
| The session API key is hidden from a signed-in user | ❌ No — it lives in browser `localStorage` |

If you need the agent server itself locked down, keep it on a private network
or behind a reverse proxy, and treat the session API key as the real secret.
`docs/SELF_HOSTING.md` covers that side.

## Setup

### 1. Create a Clerk application

In the [Clerk Dashboard](https://dashboard.clerk.com), create an application and
enable whichever sign-in methods you want (Google / GitHub SSO, email codes,
passwords).

### 2. Restrict who can sign up — this is the real control

Under **Configure → Restrictions**, set sign-up mode to **Restricted**, then
invite your team under **Users → Invitations**. Without this step Clerk will
happily create an account for anyone who finds the URL, and the gate becomes
decoration.

Clerk also offers **Allowlist** (specific addresses or `@yourdomain.com`
patterns) and **Blocklist** if that fits your team better.

### 3. Configure Agent Canvas

Copy the publishable key from **Configure → API Keys**:

```bash
VITE_CLERK_PUBLISHABLE_KEY="pk_live_..."
```

The publishable key is designed to ship in the client bundle — it identifies the
Clerk instance and is not a secret. Agent Canvas never needs Clerk's **secret**
key, because it never verifies a token server-side.

Optionally add a client-side email-domain allowlist as a second layer:

```bash
VITE_CLERK_ALLOWED_EMAIL_DOMAINS="acme.com,acme.dev"
```

A signed-in user whose *verified* primary email is outside that list gets a
"this account doesn't have access" screen with a sign-out button. Leave it unset
to rely on the Clerk Dashboard alone — which is the stronger control, since this
check runs in the browser and only catches configuration drift.

### 4. Add your domain to Clerk

Under **Configure → Domains**, add the origin the canvas is served from (your
Vercel URL, or your own domain). Clerk rejects requests from unknown origins.

### Injecting config into a prebuilt bundle

Like `VITE_SESSION_API_KEY` and `VITE_LOCK_TO_CLOUD`, both settings can be
injected at serve time instead of build time, via window globals:

```html
<script>
  window.__AGENT_CANVAS_CLERK_PUBLISHABLE_KEY__ = "pk_live_...";
  window.__AGENT_CANVAS_CLERK_ALLOWED_EMAIL_DOMAINS__ = "acme.com";
</script>
```

## How it fits together

`ClerkGate` (`src/components/features/auth/clerk-gate.tsx`) wraps the entire
app in `src/root.tsx`, *above* first-run onboarding and the API-key entry
screen — so an unauthenticated visitor never reaches a screen that can register
a backend.

```
App
└── ClerkGate                      ← no publishable key? renders straight through
    ├── <ClerkLoading>             → spinner
    └── <ClerkLoaded>
        ├── <SignedOut>            → <SignIn routing="virtual" />
        └── <SignedIn>
            └── TeamMemberGate     → email-domain allowlist (opt-in)
                └── CanvasApp      ← onboarding, backends, conversations, …
```

The sign-in flow uses `routing="virtual"`, which keeps it inside the component.
Agent Canvas has no `/sign-in` route and puts conversation IDs in the URL, so
neither `path` nor `hash` routing would work here.

Signed-in users get a "Logout" row at the top of **Settings → Application**
(`ClerkAccountSection`).

## Relationship to the existing auth in this repo

Agent Canvas already had three credential mechanisms. None of them are user
identity, and the Clerk gate replaces none of them:

| Mechanism | What it is | Changed? |
| --- | --- | --- |
| Session API key | Machine credential for the agent server | No |
| OAuth device flow | Mints an agent-server API key from OpenHands Cloud | No |
| Cookie mode (`VITE_LOCK_TO_CLOUD`) | Delegates login to a separate "main app" | No |

The cookie mode and the Clerk gate are two answers to the same question and
should not both be on. If you set `VITE_CLERK_PUBLISHABLE_KEY`, do not also
point the build at a cookie-auth cloud host.

## Styling

The sign-in card uses Clerk's default appearance, which is light while the
canvas is dark. Restyle it in the Clerk Dashboard under **Configure →
Appearance**, or pass an `appearance` prop to `ClerkProvider` in
`clerk-gate.tsx`.
