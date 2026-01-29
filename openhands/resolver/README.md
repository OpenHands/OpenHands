# OpenHands GitHub, GitLab, Bitbucket & Azure DevOps Issue Resolver 🙌

Need help resolving a GitHub, GitLab, Bitbucket, or Azure DevOps issue but don't have the time to do it yourself? Let an AI agent help you out!

This tool allows you to use open-source AI agents based on [OpenHands](https://github.com/openhands/openhands)
to attempt to resolve GitHub, GitLab, Bitbucket, and Azure DevOps issues automatically. While it can handle multiple issues, it's primarily designed
to help you resolve one issue at a time with high quality.

Getting started is simple - just follow the instructions below.

## Using the GitHub Actions Workflow

This repository includes a GitHub Actions workflow that can automatically attempt to fix individual issues labeled with 'fix-me'.
Follow these steps to use this workflow in your own repository:

1. [Create a personal access token](https://github.com/settings/tokens?type=beta) with read/write scope for "contents", "issues", "pull requests", and "workflows"

   Note: If you're working with an organizational repository, you may need to configure the organization's personal access token policy first. See [Setting a personal access token policy for your organization](https://docs.github.com/en/organizations/managing-programmatic-access-to-your-organization/setting-a-personal-access-token-policy-for-your-organization) for details.

2. Create an API key for the [Claude API](https://www.anthropic.com/api) (recommended) or another supported LLM service

3. Copy `examples/openhands-resolver.yml` to your repository's `.github/workflows/` directory

4. Configure repository permissions:
    - Go to `Settings -> Actions -> General -> Workflow permissions`
    - Select "Read and write permissions"
    - Enable "Allow Github Actions to create and approve pull requests"

    Note: If the "Read and write permissions" option is greyed out:
    - First check if permissions need to be set at the organization level
    - If still greyed out at the organization level, permissions need to be set in the [Enterprise policy settings](https://docs.github.com/en/enterprise-cloud@latest/admin/enforcing-policies/enforcing-policies-for-your-enterprise/enforcing-policies-for-github-actions-in-your-enterprise#enforcing-a-policy-for-workflow-permissions-in-your-enterprise)

5. Set up [GitHub secrets](https://docs.github.com/en/actions/security-for-github-actions/security-guides/using-secrets-in-github-actions):
   - Required:
    - `LLM_API_KEY`: Your LLM API key
   - Optional:
     - `PAT_USERNAME`: GitHub username for the personal access token
     - `PAT_TOKEN`: The personal access token
     - `LLM_BASE_URL`: Base URL for LLM API (only if using a proxy)

   Note: You can set these secrets at the organization level to use across multiple repositories.

6. Set up any [custom configurations required](https://docs.openhands.dev/usage/how-to/github-action#custom-configurations)

7. Usage:
   There are two ways to trigger the OpenHands agent:

   a. Using the 'fix-me' label:
      - Add the 'fix-me' label to any issue you want the AI to resolve
      - The agent will consider all comments in the issue thread when resolving
      - The workflow will:
        1. Attempt to resolve the issue using OpenHands
        2. Create a draft PR if successful, or push a branch if unsuccessful
        3. Comment on the issue with the results
        4. Remove the 'fix-me' label once processed

   b. Using `@openhands-agent` mention:
      - Create a new comment containing `@openhands-agent` in any issue
      - The agent will only consider the comment where it's mentioned
      - The workflow will:
        1. Attempt to resolve the issue based on the specific comment
        2. Create a draft PR if successful, or push a branch if unsuccessful
        3. Comment on the issue with the results

Need help? Feel free to [open an issue](https://github.com/openhands/openhands/issues).

## Manual Installation

If you prefer to run the resolver programmatically instead of using GitHub Actions, follow these steps:

1. Install the package:

```bash
pip install openhands-ai
```

2. Create a GitHub, GitLab, Bitbucket, or Azure DevOps access token:
   - Create a GitHub access token
      - Visit [GitHub's token settings](https://github.com/settings/personal-access-tokens/new)
      - Create a fine-grained token with these scopes:
      - "Content"
      - "Pull requests"
      - "Issues"
      - "Workflows"
      - If you don't have push access to the target repo, you can fork it first

   - Create a GitLab access token
      - Visit [GitLab's token settings](https://gitlab.com/-/user_settings/personal_access_tokens)
      - Create a fine-grained token with these scopes:
      - 'api'
      - 'read_api'
      - 'read_user'
      - 'read_repository'
      - 'write_repository'

   - Create a Bitbucket access token
      - Visit [Bitbucket's app passwords settings](https://bitbucket.org/account/settings/app-passwords/)
      - Create an app password with these scopes:
      - 'Repositories: Read'
      - 'Repositories: Write'
      - 'Pull requests: Read'
      - 'Pull requests: Write'
      - 'Issues: Read'
      - 'Issues: Write'

   - Create an Azure DevOps access token
      - Visit [Azure DevOps token settings](https://dev.azure.com/{organization}/_usersSettings/tokens)
      - Create a personal access token with these scopes:
      - 'Code: Read & Write'
      - 'Work Items: Read & Write'
      - 'Pull Request: Read & Write'

3. Set up environment variables:

```bash

# GitHub credentials

export GITHUB_TOKEN="your-github-token"
export GIT_USERNAME="your-github-username"  # Optional, defaults to token owner

# GitLab credentials if you're using GitLab repo

export GITLAB_TOKEN="your-gitlab-token"
export GIT_USERNAME="your-gitlab-username"  # Optional, defaults to token owner

# Bitbucket credentials if you're using Bitbucket repo

export BITBUCKET_TOKEN="your-bitbucket-token"
export GIT_USERNAME="your-bitbucket-username"  # Optional, defaults to token owner

# Azure DevOps credentials if you're using Azure DevOps repo

export AZURE_DEVOPS_TOKEN="your-azure-devops-token"
export GIT_USERNAME="your-azure-devops-username"  # Optional, defaults to token owner

# LLM configuration

export LLM_MODEL="anthropic/claude-sonnet-4-20250514"  # Recommended
export LLM_API_KEY="your-llm-api-key"
export LLM_BASE_URL="your-api-url"  # Optional, for API proxies
```

Note: OpenHands works best with powerful models like Anthropic's Claude or OpenAI's GPT-4. While other models are supported, they may not perform as well for complex issue resolution.

## Resolving Issues

The resolver can automatically attempt to fix a single issue in your repository using the following command:

```bash
python -m openhands.resolver.resolve_issue --selected-repo [OWNER]/[REPO] --issue-number [NUMBER]
```

For instance, if you want to resolve issue #100 in this repo, you would run:

```bash
python -m openhands.resolver.resolve_issue --selected-repo openhands/openhands --issue-number 100
```

The output will be written to the `output/` directory.

If you've installed the package from source using poetry, you can use:

```bash
poetry run python openhands/resolver/resolve_issue.py --selected-repo openhands/openhands --issue-number 100
```

## Responding to PR Comments

The resolver can also respond to comments on pull requests using:

```bash
python -m openhands.resolver.send_pull_request --issue-number PR_NUMBER --issue-type pr
```

This functionality is available both through the GitHub Actions workflow and when running the resolver locally.

## Visualizing successful PRs

To find successful PRs, you can run the following command:

```bash
grep '"success":true' output/output.jsonl | sed 's/.*\("number":[0-9]*\).*/\1/g'
```

Then you can go through and visualize the ones you'd like.

```bash
python -m openhands.resolver.visualize_resolver_output --issue-number ISSUE_NUMBER --vis-method json
```

## Uploading PRs

If you find any PRs that were successful, you can upload them.
There are three ways you can upload:

1. `branch` - upload a branch without creating a PR
2. `draft` - create a draft PR
3. `ready` - create a non-draft PR that's ready for review

```bash
python -m openhands.resolver.send_pull_request --issue-number ISSUE_NUMBER --username YOUR_GIT_USERNAME --pr-type draft
```

If you want to upload to a fork, you can do so by specifying the `fork-owner`:

```bash
python -m openhands.resolver.send_pull_request --issue-number ISSUE_NUMBER --username YOUR_GIT_USERNAME --pr-type draft --fork-owner YOUR_GIT_USERNAME
```

## Providing Custom Instructions

You can customize how the AI agent approaches issue resolution by adding a repository microagent file at `.openhands/microagents/repo.md` in your repository. This file's contents will be automatically loaded in the prompt when working with your repository. For more information about repository microagents, see [Repository Instructions](https://github.com/OpenHands/OpenHands/tree/main/microagents#2-repository-instructions-private).

## Troubleshooting

If you have any issues, please open an issue on this GitHub, GitLab, or Bitbucket repo, we're happy to help!
Alternatively, you can [email us](mailto:contact@openhands.dev) or join the OpenHands Slack workspace (see [the README](/README.md) for an invite link).
