name: upcoming-release
description: Generate a summary of PRs that are including in the upcoming release.
triggers:
- /upcoming-release
---

We want to know what is part of the upcoming release.

To do this, you need two commit SHAs. One SHA is what is currently running. The second SHA is what is going to be
released. The user must provide these. If the user does not provide these, ask the user to provide them before doing
anything.

Once you have received the two SHAs:
1. Run the `.github/scripts/find_prs_between_commits.py` script from the repository root directory. The **first SHA** should be the older commit (current release), and the **second SHA** should be the newer commit (what's being released).
2. Summarize what is part of these PRs. Present the results in markdown format, grouped by: Added Features, Fixed Bugs, and Other. Include the PR numbers in your summary.
