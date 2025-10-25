# OpenHands GitHub Actions Workflow Templates

This directory contains workflow templates that make it easy to integrate OpenHands AI agent into your GitHub CI/CD workflows.

## Available Templates

### 1. OpenHands Code Review (`openhands-code-review.yml`)
Automatically reviews pull requests for code quality, security, and best practices.

**Triggers:** Pull request opened or updated
**Permissions:** `contents: read`, `pull-requests: write`, `issues: write`

### 2. OpenHands Bug Fix (`openhands-bug-fix.yml`)
Automatically investigates and fixes bugs when issues are labeled with 'bug'.

**Triggers:** Issue labeled with 'bug'
**Permissions:** `contents: write`, `pull-requests: write`, `issues: write`

### 3. OpenHands Documentation (`openhands-documentation.yml`)
Keeps project documentation up-to-date by reviewing code changes and updating docs.

**Triggers:** Weekly schedule, manual dispatch, or code changes
**Permissions:** `contents: write`, `pull-requests: write`

### 4. OpenHands Custom Task (`openhands-custom-task.yml`)
Run any custom development task with a manual trigger and custom description.

**Triggers:** Manual dispatch with task description input
**Permissions:** `contents: write`, `pull-requests: write`, `issues: write`

## Setup Instructions

### Prerequisites

1. **OpenHands API Key**: Get your API key from [OpenHands](https://app.all-hands.dev)
2. **GitHub Repository**: The templates work with any GitHub repository

### Installation

1. **Add API Key Secret**:
   - Go to your repository's Settings → Secrets and variables → Actions
   - Add a new repository secret named `OPENHANDS_API_KEY`
   - Set the value to your OpenHands API key

2. **Use Templates**:
   - Go to your repository's Actions tab
   - Click "New workflow"
   - Look for "OpenHands" templates in the template gallery
   - Choose the template that fits your needs
   - Customize as needed and commit

### Customization

All templates can be customized:

- **Prompts**: Modify the task descriptions to fit your specific needs
- **Triggers**: Change when workflows run (schedule, events, manual)
- **Timeouts**: Adjust polling timeouts based on task complexity
- **Permissions**: Modify based on what actions OpenHands needs to perform

### Example Customizations

#### Custom Review Criteria
```yaml
prompt = '''Please review this pull request focusing on:
- Performance optimization opportunities
- Database query efficiency
- API design consistency
- Error handling completeness
'''
```

#### Specific Documentation Updates
```yaml
prompt = '''Update the following documentation:
1. API reference in docs/api.md
2. Installation guide in README.md
3. Code examples in docs/examples/
'''
```

## How It Works

1. **Workflow Trigger**: GitHub event triggers the workflow
2. **Setup**: Installs Python and downloads the OpenHands API helper
3. **API Call**: Creates a conversation with OpenHands using your prompt
4. **Execution**: OpenHands performs the requested task in your repository
5. **Results**: OpenHands may create PRs, comments, or other outputs

## Security Considerations

- **API Key**: Keep your `OPENHANDS_API_KEY` secret secure
- **Permissions**: Templates request minimal required permissions
- **Repository Access**: OpenHands will have access to your repository during task execution
- **Review Changes**: Always review any PRs or changes made by OpenHands

## Troubleshooting

### Common Issues

1. **Missing API Key**: Ensure `OPENHANDS_API_KEY` is set in repository secrets
2. **Permission Errors**: Check that workflow permissions match template requirements
3. **Timeout Issues**: Increase timeout values for complex tasks
4. **Rate Limits**: OpenHands API has rate limits; space out workflow runs if needed

### Getting Help

- Check the [OpenHands Documentation](https://docs.all-hands.dev)
- Review workflow run logs for detailed error messages
- Ensure your repository is accessible and has the necessary permissions

## Contributing

To improve these templates:

1. Test changes thoroughly
2. Update documentation
3. Follow GitHub Actions best practices
4. Consider security implications

## License

These templates are provided under the same license as the OpenHands project.
