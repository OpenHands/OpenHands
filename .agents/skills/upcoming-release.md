---
name: upcoming-release
description: Generate a concise summary of PRs included in the upcoming release.
triggers:
- /upcoming-release
---

We want to know what is part of the upcoming release.

To do this, you need two commit SHAs. One SHA is what is currently running. The second SHA is what is going to be
released. The user must provide these. If the user does not provide these, ask the user to provide them before doing
anything.

Once you have received the two SHAs:
1. Run the `.github/scripts/find_prs_between_commits.py` script from the repository root directory with the `--json` flag. The **first SHA** should be the older commit (current release), and the **second SHA** should be the newer commit (what's being released).
2. Do not show PRs that are chores, dependency updates, adding logs, refactors.
3. From the remaining PRs, split them into these categories:
   - Features
   - Bug fixes
   - Security/CVE fixes
   - Other
4. The output should list the PRs under their category, including the PR number with a brief description of the PR.
