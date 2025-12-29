#!/bin/bash

set -euxo pipefail

# This script updates the PR description with commands to run the PR locally
# It adds both Docker and uvx commands

# Get the branch name for the PR
BRANCH_NAME=$(gh pr view "$PR_NUMBER" --json headRefName --jq .headRefName)

# Define the Docker command
DOCKER_RUN_COMMAND="docker run -it --rm \
  -p 3000:3000 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --add-host host.docker.internal:host-gateway \
  -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.openhands.dev/openhands/runtime:${SHORT_SHA}-nikolaik \
  --name openhands-app-${SHORT_SHA} \
  docker.openhands.dev/openhands/openhands:${SHORT_SHA}"

# Get the current PR body
PR_BODY=$(gh pr view "$PR_NUMBER" --json body --jq .body)

# Prepare the new PR body with both commands
if echo "$PR_BODY" | grep -q "To run this PR locally, use the following command:"; then
  # For existing PR descriptions, use a more robust approach
  # Split the PR body at the "To run this PR locally" section and replace everything after it
  BEFORE_SECTION=$(echo "$PR_BODY" | sed '/To run this PR locally, use the following command:/,$d')
  NEW_PR_BODY=$(cat <<EOF
${BEFORE_SECTION}

To run this PR locally, use the following command:

GUI with Docker:
\`\`\`
${DOCKER_RUN_COMMAND}
\`\`\`
EOF
)
else
  # For new PR descriptions: use heredoc safely without indentation
  NEW_PR_BODY=$(cat <<EOF
$PR_BODY

---

To run this PR locally, use the following command:

GUI with Docker:
\`\`\`
${DOCKER_RUN_COMMAND}
\`\`\`
EOF
)
fi

# Update the PR description
echo "Updating PR description with Docker and uvx commands"
gh pr edit "$PR_NUMBER" --body "$NEW_PR_BODY"
