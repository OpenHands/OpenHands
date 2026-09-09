---
name: pr-design-doc
description: >
  For a non-trivial pull request, write a self-contained HTML design doc under the
  temporary `.pr/` directory and link it in the PR description via htmlpreview, so
  maintainers grasp the proposal at a glance - code/API design, and the before/after of
  the change, grounded to real code. Use when opening or updating a non-trivial PR, or
  when the user says "add a design doc", "document this PR for reviewers", "show the
  before/after", "make the design reviewable", or "write the .pr/ page".
triggers:
- /pr-design-doc
- /design-doc
license: MIT
metadata:
  tags: pull-request, design-doc, html, review, before-after, htmlpreview
---

# pr-design-doc - a reviewable design doc for a non-trivial PR

A diff shows *what changed line by line*. It does not show *the design*: the shape of the
change, the API before and after, and why this approach. Reviewers reconstruct that by
hand, slowly. The scarce resource is the maintainer's attention and trust budget - not the
agent's effort. Spend extra effort to hand them **one self-contained HTML page** that
conveys the **big picture** and the **before → after core difference**, with every claim
**clickable back to the real code**, then link it from the PR description.

This is the same craft as a "show me this change" explainer, aimed at one job: making a
non-trivial PR easy to review.

## When to use it

- Opening or updating a **non-trivial** PR: new/changed public API, a new module or
  subsystem, a behavior change in core logic, a migration, or anything a reviewer can't
  fully judge from the diff in a couple of minutes.
- **Skip it** for trivial PRs - a typo, a one-line guard, a dependency bump, a docs tweak, a simple bug fix.
  A design doc adds more to review. Use judgment; if the diff *is* the explanation, don't add a
  page.

## The `.pr/` workflow

Use the temporary **`.pr/`** directory for PR-only artifacts. Before relying on automatic
cleanup, verify that the target repository has an enabled
`.github/workflows/pr-artifacts.yml` workflow that removes `.pr/` after approval.

- Same-repository PR with verified cleanup workflow: the workflow removes `.pr/` after
  approval.
- Fork PR, or repository without a verified cleanup workflow: remove `.pr/` manually before
  merge.

The design doc is a review aid that lives with the branch while the PR is open. It must not
ship in the merged tree.

## Workflow

1. **Check out and verify the PR head.** Do not write or commit the design doc from the base
   branch or an unrelated checkout. Start with a clean worktree, then inspect and check out
   the PR:
   ```bash
   gh pr view <n> --json title,body,baseRefName,baseRefOid,headRefName,headRefOid,headRepository,headRepositoryOwner,isCrossRepository,files,additions,deletions
   gh pr checkout <n>
   git rev-parse HEAD
   gh pr view <n> --json headRefOid --jq .headRefOid
   ```
   The final two SHAs must match before you continue. If they do not, stop and fix the
   checkout. Compute the merge-base SHA with
   `git merge-base <baseRefOid> <headRefOid>`. Group changed files by area and keep both the
   merge-base SHA and head SHA for source links.

2. **Read both sides of each logical file.** Compare
   `git show <merge-base-sha>:<path>` with the verified head. Capture the
   **function-level** behavioral difference - what the code *did* vs *does now*.
   - new file → no "before"; one "after" diagram + a line on the role it adds.
   - deleted file → "before" diagram + who/what takes over.
   - edited file → a before/after pair, with the delta highlighted.

3. **Classify each file.** *Logic* change (behavior moved) → draw before/after. *Mechanical*
   change (rename, constant, config, import move) → a one-line `before → after` row, no
   diagram. Don't dilute the signal by drawing mechanical edits.

4. **If the change is an API change, lead with the API.** Show the signature/schema/type
   **before and after** side by side (function signature, endpoint + payload, config field,
   event shape). Name the compatibility impact plainly: additive, breaking, or behind a flag.

5. **Find the cross-file story.** If one call chain threads several files, draw a single
   **overview** before/after at the top; per-file cards drill in.

6. **Build the page** per [`references/html-craft.md`](references/html-craft.md) - one
   self-contained, offline, editorial HTML file with hand-drawn SVG figures. Save it to
   the repo's `.pr/` directory, e.g. `.pr/design.html` (or `.pr/<topic>.html`).

7. **Commit under `.pr/`, push to the verified PR head, and link it.** Confirm that the push
   remote resolves to `headRepository.nameWithOwner`; never push the artifact to the base
   repository's default branch.
   ```bash
   git add .pr/design.html
   git commit -m "docs(.pr): design doc for <PR topic>"
   git push <head-repo-remote> HEAD:<headRefName>
   ```
   Then add the htmlpreview link near the top of the PR description, pointing at the **fork
   and branch the PR is opened from** (it renders before merge):
   ```
   📄 Design doc: https://htmlpreview.github.io/?https://github.com/<fork-owner>/<repo>/blob/<pr-branch>/.pr/design.html
   ```

## What the page contains

1. **What changed (decision first)** - one paragraph: the intent, net effect, and why the
   reviewer should care. Put the highest-impact conclusion, risk, or API-compat note in a
   `★` callout, with the most important changed `path:line` nearby. Stats (`N files ·
   +A / −D`) are context, not the lead. If there's a cross-file flow, the **overview
   before/after SVG** goes here.
2. **API before → after** (when the PR changes an interface) - signatures/schemas/types side
   by side, with the compatibility verdict stated.
3. **Left rail / index** - changed files grouped by area, each tagged (🟢 added · 🔴 removed ·
   ✏️ changed · ⚙️ mechanical) with +/− counts; click to jump.
4. **Per-file cards** - for each logical file: a claim-carrying title, a one-line summary of
   how its behavior changed, **before/after** diagrams with real symbol names + `file:line`
   (changed nodes in orange), and the diff in a collapsed `<details>`. Mechanical files get a
   small `before → after` table, no diagram.
5. **(optional) Risk / follow-ups** - only if grounded in what you read.

## Non-negotiable principles

1. **Optimize for scarce reviewer attention.** The first screen answers, in ~15 seconds:
   what this PR does, whether it's risky, where to look first, and what evidence backs the
   claim. Lead with the conclusion, not your process.
2. **Show the difference, not just the after.** For any logic or API change, draw **before**
   and **after** and make the *delta* visually loud (color + line style). The contrast is
   the product.
3. **Ground everything to code, beside the claim.** Every box, node, and sentence names a
   real symbol + `path:line`, and links to the correct source revision where possible: the
   merge-base SHA for before-state evidence and the verified head SHA for after-state
   evidence. One click from "this changed" to the exact code.
4. **Hand-draw the carrying diagrams.** Prefer bespoke inline SVG for the before/after that
   makes the argument; Mermaid is fine only for quick auxiliary graphs.
5. **Self-contained & offline.** One HTML file, inline CSS/SVG, opens by double-click,
   survives being copied to another machine (htmlpreview needs this).
6. **`.pr/` only, and temporary.** The doc is a review aid, not project docs. Keep it in
   `.pr/` and ensure it is removed before merge. Rely on automatic cleanup only when the
   repository's workflow has been verified; otherwise remove it manually. Do not move design
   HTML into `docs/` or ship it in the merged tree.

## Anti-patterns

- ❌ Dumping the raw diff / file tree and calling it a "design doc" - adds nothing over the
  PR page.
- ❌ Empty nodes ("process data", "handle request") - every node is a real symbol +
  location.
- ❌ Only the after-state when something changed - reviewers want the *contrast*.
- ❌ A design doc on a trivial PR - noise. Skip it.
- ❌ Committing the HTML outside `.pr/` (e.g. `docs/`), where it would merge into `main`.
- ❌ Publishing a private-repository design doc through htmlpreview, GitHub Pages, or another
  public host. Use the private/local preview path in the craft reference. Use GitHub Pages
  only with explicit user authorization after verifying private Pages access control.
