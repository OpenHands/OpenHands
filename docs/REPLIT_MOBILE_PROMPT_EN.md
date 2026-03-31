Build a production-grade **web + mobile companion platform** for the existing live OpenHands deployment at **https://hands.gantor.ir**.

Name the product:
- **Gantor OpenHands Companion**

Important positioning:
- This project must **integrate with the live QADR OpenHands system**, not replace it.
- Treat `hands.gantor.ir` as the primary live agent backend and operator interface.
- Treat the QADR infrastructure as the execution environment and FreeGPT.ir as the LLM/API backbone.
- The result should feel comparable to, and in several operator workflows better than, Devin, ChatGPT Agent, Codex, and similar agentic development systems.

## Language and quality requirements

- Write all code, comments, documentation, and architecture notes in English.
- The UI must support:
  - English
  - Persian
- Persian must be first-class and fully polished.
- Support proper RTL/LTR behavior, especially for:
  - file paths
  - commands
  - shell output
  - tokens
  - code
  - logs
  - diffs
- Mobile UX quality must be extremely high.

## Platform targets

Build for:
- responsive web
- installable PWA
- Android app
- iOS app

Preferred mobile approach:
- React Native + Expo + TypeScript
- Expo Router
- Zustand or equivalent state management
- TanStack Query for data fetching
- WebSocket / SSE support
- Capacities for future native modules

If you choose a different mobile stack, it must still be production-grade and easy to build in Replit.

## Device-specific requirement

Optimize layout behavior for:
- Samsung Galaxy Z Fold
- foldable / resizable layouts
- compact, medium, and expanded breakpoints

The app must intelligently adapt between:
- phone portrait
- phone landscape
- foldable cover screen
- foldable tablet-like expanded mode

## Existing live environment you must integrate with

Primary OpenHands target:
- `https://hands.gantor.ir`

Related QADR infrastructure:
- `https://chat.freegpt.ir`
- `https://freegpt.ir`
- `https://api.freegpt.ir`
- `https://watch.alefba.dev`
- QADR host runs Dockerized services and AI infrastructure

Important existing QADR integration assumptions:
- OpenHands is hosted on QADR behind Caddy
- OpenHands uses Docker runtime / sandbox containers
- OpenHands can use FreeGPT/LiteLLM as its model backend
- FreeGPT API is OpenAI-compatible
- QADR also hosts monitoring, VPN, Enigma, MiroFish, and other internal services

## Product goal

Create a companion application and operator console for OpenHands that enables:
- mobile-first and web-based access to OpenHands sessions
- project and repo management
- code task execution
- session control
- terminal access workflows
- web browsing task visibility
- secure SSH task orchestration abstractions
- task planning and execution visibility
- notifications
- approvals
- incident awareness
- AI-enhanced operator controls

This must not be a shallow wrapper. It should become a serious agent workspace on top of the existing OpenHands deployment.

## Key capabilities

### 1. Session management
- list active and historical OpenHands sessions
- create new session
- resume existing session
- view status, duration, model, budget, current task
- show whether a session is:
  - planning
  - browsing
  - coding
  - executing commands
  - waiting for approval
  - blocked
  - failed
  - completed

### 2. Project and repo workspace integration
- connect sessions to repositories and workspaces
- support repo selection
- show branch, commit, dirty state, ahead/behind, PR status
- support repository notes and operational tags
- show what repo and environment a session is acting on
- allow operators to pin important repos

### 3. Agent execution console
- live activity feed
- step timeline
- thought summary cards
- command execution feed
- file changes summary
- diff preview
- approval requests
- warnings and policy blocks

This should feel stronger than a simple chat transcript.

### 4. Code and terminal workflows
- show terminal execution summaries
- show command history for the session
- allow safe operator approvals for sensitive actions
- support structured command requests
- support command templates
- support dry-run where possible
- clearly differentiate:
  - read-only actions
  - repo write actions
  - server write actions
  - privileged actions

### 5. Web and browser workflows
- show browser-driven task state
- show visited pages and extracted findings
- support mobile-friendly browsing history summaries
- support action previews before high-impact changes

### 6. SSH and infrastructure abstractions

Do not expose raw unsafe SSH by default.

Instead, design a secure operator abstraction layer for:
- host command execution
- service checks
- Docker actions
- repo sync
- deployment actions
- log collection
- health probes
- controlled restarts

Support future integration with:
- QADR direct SSH
- controlled bastion or tunnel access
- audited remote command execution

### 7. FreeGPT API integration

The app must integrate with the FreeGPT/OpenAI-compatible API layer.

Model integration requirements:
- support selecting available agent models
- support service-specific model profiles
- show model health and cost tier
- show whether the model is:
  - free
  - coding-focused
  - reasoning-focused
  - premium
  - internal
- support role-aware defaults

Assume these types of model lanes may exist:
- Gantor engine
- free models
- coding models
- reasoning models
- local/internal fallback models

### 8. Notifications and mobile operations
- push notifications
- in-app notifications
- task completion notifications
- approval needed notifications
- failure notifications
- budget warnings
- infrastructure warnings
- reminder scheduling

### 9. Admin and governance
- role-based access control
- audit logs
- session ownership
- approval policies
- emergency stop
- operator notes
- per-session risk flags
- policy-based command restrictions

### 10. Observability and AI insights
- session analytics
- failure clustering
- model usage trends
- agent effectiveness insights
- cost tracking
- time-to-completion metrics
- repo and environment impact summaries
- AI-generated recommendations
- AI-generated post-run summaries

## Required architecture

Build a clean architecture with these domains:
- auth
- users and roles
- environments
- repositories
- sessions
- runs
- commands
- approvals
- notifications
- models
- providers
- metrics
- audit logs
- settings

## Security requirements

This is an operations-grade system. Security matters.

Mandatory:
- RBAC
- audit logging
- scoped API credentials
- secure token storage
- no plaintext secret exposure in the UI
- approval workflows for dangerous actions
- explicit separation between observation and mutation
- support for policy gates before privileged actions
- session timeout and re-auth flows
- safe mobile persistence rules

## UX requirements

This must feel premium and operationally serious.

Requirements:
- elegant design
- clear typography
- excellent dark mode
- touch-friendly mobile layout
- resizable desktop layout
- split view on larger screens
- timeline + terminal + diff + chat coexistence on expanded/foldable layouts
- excellent loading and empty states
- polished Persian UX
- no generic admin-dashboard look

## Replit compatibility requirements

Make the project realistic for Replit development:
- clean setup
- `.env.example`
- mock mode for disconnected development
- production connector mode for real QADR integration
- modular API client layer
- documented deployment strategy

## API and integration layer

Design a typed connector layer for:
- OpenHands live backend at `hands.gantor.ir`
- FreeGPT API at `api.freegpt.ir`
- future QADR Watch monitoring APIs
- future SSH/action APIs
- future repository intelligence APIs

If some live endpoints are not yet documented, create:
- a connector abstraction
- mock adapters
- production adapters
- clear placeholders for real endpoint binding

## Data model expectations

Track and present:
- session id
- task title
- repo
- branch
- environment
- model
- provider
- runtime type
- created time
- updated time
- status
- token/cost estimate
- operator ownership
- approval state
- risk score
- summaries
- outputs

## Suggested navigation

- Overview
- Sessions
- Repositories
- Tasks
- Terminal
- Browser
- Approvals
- Models
- Notifications
- Audit
- Settings

## Deliverables

Produce:
1. full project scaffold
2. production-grade architecture
3. responsive web UI
4. Android-first mobile app
5. iOS-ready mobile support
6. PWA support
7. typed integration layer
8. mock + production modes
9. clear README
10. environment configuration docs
11. deployment notes
12. explanation of how it integrates with:
   - `hands.gantor.ir`
   - `api.freegpt.ir`
   - QADR infrastructure

## Build order

1. define architecture and data model
2. scaffold UI shell and mobile shell
3. implement i18n and RTL/LTR
4. implement auth and roles
5. implement session list and detail screens
6. implement repo/project layer
7. implement activity timeline and command feed
8. implement approvals and notifications
9. implement model/provider layer
10. implement QADR/FreeGPT integrations
11. optimize foldable/responsive layouts
12. polish UX and docs

## Final quality bar

The result should feel like a serious operator-grade AI development environment:
- stronger than a thin chat wrapper
- deeply integrated with QADR
- aware of repositories, sessions, commands, models, and approvals
- comfortable on desktop and excellent on mobile
- especially strong on Samsung Z Fold-style layouts

The name everywhere should be:
**Gantor OpenHands Companion**

The primary live backend to integrate with is:
**https://hands.gantor.ir**
