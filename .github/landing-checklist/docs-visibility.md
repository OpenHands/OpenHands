# Docs visibility — hidden until verified flag-on

Item 1 of the landing checklist ("documentation added for the feature on
the website") needs to be true in two different senses at two different
times:
- **At merge time:** the docs page exists, is reviewed, is technically
  accurate — but the feature isn't public yet (could still be behind a
  flag, mid bug-bash, awaiting tech-council sign-off).
- **At independently verified flag-on:** the same page becomes discoverable to
  the world; final GA still waits for flag removal and social-post evidence.

Mintlify (which powers `OpenHands/docs`) has a built-in mechanism for
exactly this split: **frontmatter `hidden: true`**.

## The mechanism (confirmed from Mintlify's own docs)

Source: [Mintlify page metadata — `hidden`](https://www.mintlify.com/docs/organize/pages#param-hidden).

Add to the page's YAML frontmatter:

```yaml
---
title: "My New Feature"
description: "..."
hidden: true
---
```

Effect: the page is excluded from sidebar navigation, site search,
`sitemap.xml`, search-engine indexing, and AI-assistant context — but the
file still exists in the repo, builds normally, and deploys with everything
else. To reveal it, **remove the `hidden` field entirely** (Mintlify's docs
explicitly warn not to set `hidden: false` — that's undefined behavior).

**Important limitation, stated directly in Mintlify's own docs:** hidden
pages are not access-controlled. "Anyone who knows the URL can still access
them." This hides *discovery* (nav, search, Google, AI assistants) — it
does not prevent someone with the direct link from viewing the page. For
"don't let this leak to the world before we're ready," that's almost
certainly sufficient (nobody stumbles onto it, it's not indexed, it's not
in nav). If a URL genuinely needs to be access-controlled (not just
undiscoverable), Mintlify supports group-based auth, but that's a
meaningfully bigger lift than a frontmatter flag — flag it if you actually
need that stronger guarantee for a specific feature.

## One-time pilot validation

As of June 16, 2026, GitHub code search finds no existing `hidden: true`
frontmatter in `OpenHands/docs`, so the Enterprise pilot will introduce this
convention. Before treating the flag as a control, use the docs deployment
preview to confirm the page is absent from navigation, search, sitemap, and AI
context while still building successfully. This validates the current Mintlify
configuration rather than relying only on vendor documentation.

## Who sets/clears the flag, and when

| Stage | `hidden` state | Set/cleared by |
|---|---|---|
| PR opened, docs page authored | `hidden: true` | The PR author, as part of doing checklist item 1 — instructed via the PR template note |
| Merged → In Prod → Bug Bash → Council Review | stays `hidden: true` | n/a (no automation touches it) |
| Two council approvals (`stage:council-approved`) | stays `hidden: true` | Council approval alone is not proof the feature is enabled |
| Flag independently verified in production (`stage:flag-on`) | **cleared** — `hidden` field removed | **Automation 7**, on its next cron pass, opens the reveal PR and enables auto-merge after required checks pass |
| GA / flag removed from code | already cleared, no further action needed | n/a |

This piggybacks on Automation 7 (`07-flag-cleanup-reminder.md`): it already
runs against `stage:flag-on` tickets, so it can reveal docs and separately
manage the three-month cleanup timer. For same-day reveal, move that step
into the production reconciler that advances `stage:council-approved` to
`stage:flag-on`; that reconciler is still to be implemented.

## Real (not honor-system) verification for checklist item 1

Per your "yes" on closing the honor-system gap: **Automation 3** (prod
tracker) is the one that now actually checks this instead of trusting a
PR-body checkbox. When it confirms a feature has reached `stage:in-prod`,
it also searches `OpenHands/docs` for a merged PR referencing the same
feature PR URL or Linear ticket, and confirms:
1. a matching `.mdx` page exists under `openhands/usage/` (or wherever the
   feature naturally lives), and
2. that page currently has `hidden: true` in its frontmatter (expected —
   it shouldn't be public yet at this stage).

If no matching docs PR is found at all, Automation 3 does **not** silently
tick the checklist item — it leaves it unchecked and calls it out
explicitly in both the Linear comment and the tracker comment, the same
way it already handles ambiguous prod-detection cases for other repos. See
`automations/03-prod-tracker.md` for the exact prompt wording.

## Reveal repository permissions and merge policy

`OpenHands/docs` is not part of the production-repo allowlist. It is an
additional integration repository, and Automation 7 needs pull-request write
access to it. The reveal automation opens a normal PR that removes
`hidden: true`, enables auto-merge, and waits for the repository's required
checks. It must not bypass branch protection or merge a failing docs PR.
