# Feature Landing Checklist — central artifacts

This is the canonical home for the engineering "landing checklist" process:
the design that takes a `(feat)` PR beyond just merging, through docs, an E2E
test, self-hosted (Helm / embedded-cluster) availability, production release,
a 3+ person bug bash, tech-council approval, public launch communication, and
eventual feature-flag removal.

Start with [`PLAN.md`](PLAN.md) for the full architecture and rollout plan.

## What's canonical here vs. reference-only

Per the plan, this repo is the single source of truth for the two pieces
every production repo shares:

- **`../workflows/landing-checklist-reusable.yml`** (at
  `.github/workflows/landing-checklist-reusable.yml`) — the reusable required
  check. Every production repo's caller workflow points at this file `@main`,
  so editing checklist copy or validation logic here updates all repos at
  once.
- **[`repos.yml`](repos.yml)** — the expandable allowlist of production
  repositories. Automations resolve this file at run time instead of
  hard-coding repo names.

This repo is itself production repo #1, so it also carries its own caller at
`.github/workflows/landing-checklist.yml` (pointing at the reusable workflow
in this same repo, `@main`).

Everything else in this directory — the Linear state machine and issue
template, the seven OpenHands automation specs, the deterministic
council-approval gate script and its tests, `docs-visibility.md`, and
`tracker-format.md` — is the full reference design, kept here for
discoverability alongside the workflow it describes.

## Related work

- `OpenHands/enterprise#91` installs the caller workflow and PR-template
  checklist for the Enterprise pilot repo, pointing at the reusable workflow
  added here.
