---
name: upcoming-release
description: This skill should be used when the user asks to "generate release notes", "list upcoming release PRs", "summarize upcoming release", "/upcoming-release", or needs to know what changes are part of an upcoming release.
---

# Upcoming Release Summary

Generate a concise summary of PRs included in the upcoming release.

## Prerequisites

Two commit SHAs are required:
- **First SHA**: The older commit (current release)
- **Second SHA**: The newer commit (what's being released)

If the user does not provide both SHAs, ask for them before proceeding.

## Workflow

1. Run the script from the repository root with the `--json` flag:
   ```bash
   .github/scripts/find_prs_between_commits.py <older-sha> <newer-sha> --json
   ```

2. Filter out PRs that are:
   - Chores
   - Dependency updates
   - Adding logs
   - Refactors

3. Categorize the remaining PRs:
   - **Features** - New functionality
   - **Bug fixes** - Corrections to existing behavior
   - **Security/CVE fixes** - Security-related changes
   - **Other** - Everything else

4. Format the output with PRs listed under their category, including the PR number and a brief description.
