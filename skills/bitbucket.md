---
name: bitbucket
type: knowledge
version: 1.0.0
agent: CodeActAgent
triggers:
- bitbucket
- git
---

You have access to an environment variable, `BITBUCKET_TOKEN`, which allows you to interact with
the Bitbucket API.

<IMPORTANT>
You can use `curl` with the `BITBUCKET_TOKEN` to interact with Bitbucket's API.
ALWAYS use the Bitbucket API for operations instead of a web browser.
ALWAYS use the `create_bitbucket_pr` tool to open a pull request
</IMPORTANT>

If you encounter authentication issues when pushing to Bitbucket (such as password prompts or permission errors), the old token may have expired. In such case, update the remote URL to include the current token: `git remote set-url origin https://x-token-auth:${BITBUCKET_TOKEN}@bitbucket.org/username/repo.git`

Here are some instructions for pushing, but ONLY do this if the user asks you to:
* NEVER push directly to the `main` or `master` branch
* Git config (username and email) is pre-set. Do not modify.
* You may already be on a branch starting with `openhands-workspace`. Create a new branch with a better name before pushing.
* Use the `create_bitbucket_pr` tool to create a pull request, if you haven't already
* Once you've created your own branch or a pull request, continue to update it. Do NOT create a new one unless you are explicitly asked to. Update the PR title and description as necessary, but don't change the branch name.
* Use the main branch as the base branch, unless the user requests otherwise
* After opening or updating a pull request, send the user a short message with a link to the pull request.
* Do NOT mark a pull request as ready to review unless the user explicitly says so
* Do all of the above in as few steps as possible. E.g. you could push changes with one step by running the following bash commands:
```bash
git remote -v && git branch # to find the current org, repo and branch
git checkout -b create-widget && git add . && git commit -m "Create widget" && git push -u origin create-widget
```
