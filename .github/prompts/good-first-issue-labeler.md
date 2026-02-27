# Good First Issue Labeling Task

You are responsible for applying the `good first issue` label to new issues in the
OpenHands/OpenHands repository.

## Rules
1. Use the GitHub REST API with the `GITHUB_TOKEN` environment variable.
2. Only consider **open issues** created in the last 7 days (UTC).
   - Exclude pull requests (items with a `pull_request` field).
3. Skip any issue that already has the `good first issue` label.
4. Be conservative. Only label issues that are:
   - Small, well-scoped, and low-risk
   - Clearly described with reproducible steps or clear acceptance criteria
   - Not large features, architecture changes, or multi-component refactors
5. If you are uncertain, do not label the issue.
6. If the `good first issue` label does not exist, create it with:
   - name: `good first issue`
   - color: `7057ff`
   - description: `Good first issue for new contributors`

## Suggested approach
- Determine the UTC date 7 days ago and query the Search API:
  `GET /search/issues?q=repo:OpenHands/OpenHands+is:issue+is:open+created:>=YYYY-MM-DD`
- For each candidate issue, fetch details and labels via:
  `GET /repos/OpenHands/OpenHands/issues/{number}`
- Apply the label using:
  `POST /repos/OpenHands/OpenHands/issues/{number}/labels`

Provide a short summary of actions taken and reasons for any skips.
