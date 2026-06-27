# Issue Triage
These are the procedures and guidelines on how issues are triaged in this repo by the maintainers.

## General
* All issues must be tagged with **enhancement**, **bug** or **troubleshooting/help**.
* Issues may be tagged with what it relates to (**llm**, **app tab**, **UI/UX**, etc.).

## Severity
* **High**: High visibility issues or affecting many users.
* **Critical**: Affecting all users or potential security issues.

## Difficulty
* Issues good for newcomers may be tagged with **good first issue** by maintainers or by the automated triage workflow when the issue clearly meets the criteria below.
* The `welcome-good-first-issue` workflow only posts a welcome comment after the label is present; it does not decide whether the label should be applied.
* The automated triage workflow should be conservative and should not auto-apply **good first issue** to issues that look like duplicates or overlapping-scope reports.
* Use **good first issue** only when all of the following are true:
  * The work is narrow in scope and should be solved as a single issue rather than a multi-step project.
  * The bug, request, or expected outcome is already clear enough that a contributor should not need major discovery work before starting.
  * The likely fix stays within a well-bounded area of the repo and does not require deep architecture knowledge, cross-repo coordination, enterprise-only context, migrations, or infrastructure work. Exclude enterprise/ directory from these.
  * The validation path is straightforward: the PR author should be able to give clear **How to Test** steps, and if the change is user-facing, screenshots or video evidence should be practical to provide.
* Avoid **good first issue** for broad design discussions, umbrella tracking issues, items missing reproduction steps or acceptance criteria, and changes that are likely to require significant maintainer guidance.

## Not Enough Information
* User is asked to provide more information (logs, how to reproduce, etc.) when the issue is not clear.
* If an issue is unclear and the author does not provide more information or respond to a request,
the issue may be closed as **not planned** (Usually after a week).

## Multiple Requests/Fixes in One Issue
* These issues will be narrowed down to one request/fix so the issue is more easily tracked and fixed.
* Issues may be broken down into multiple issues if required.

## Stale and Auto Closures
* In order to keep a maintainable backlog, issues that have no activity within 40 days are automatically marked as **Stale**.
* If issues marked as **Stale** continue to have no activity for 10 more days, they will automatically be closed as not planned.
* Issues may be reopened by maintainers if deemed important.
