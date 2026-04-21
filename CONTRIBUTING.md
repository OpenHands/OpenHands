# Contributing

Thanks for your interest in contributing to OpenHands! We're building the future of AI-powered software development, and we'd love for you to be part of this journey.

## Our Vision

The OpenHands community is built around the belief that AI and AI agents are going to fundamentally change the way we build software. If this is true, we should do everything we can to make sure that the benefits provided by such powerful technology are accessible to everyone.

We believe in the power of open source to democratize access to cutting-edge AI technology. Just as the internet transformed how we share information, we envision a world where AI-powered development tools are available to every developer, regardless of their background or resources.

## Getting Started

### Quick Ways to Contribute

- **Use OpenHands** and [report issues](https://github.com/OpenHands/OpenHands/issues) you encounter
- **Give feedback** using the thumbs-up/thumbs-down buttons after each session
- **Star our repository** on [GitHub](https://github.com/OpenHands/OpenHands)
- **Share OpenHands** with other developers

### Set Up Your Development Environment

- **Requirements**: Linux/Mac/WSL, Docker, Python 3.12, Node.js 22+, Poetry 1.8+
- **Quick setup**: `make build`
- **Run locally**: `make run`
- **LLM setup (V1 web app)**: configure your model and API key in the Settings UI after the app starts

Full details in our [Development Guide](./Development.md).

### Find Your First Issue

- Browse [good first issues](https://github.com/OpenHands/OpenHands/labels/good%20first%20issue)
- Check our [project boards](https://github.com/OpenHands/OpenHands/projects) for organized tasks
- Join our [Slack community](https://openhands.dev/joinslack) to ask what needs help

## Understanding the Codebase

- **[Frontend](./frontend/README.md)** - React application
- **[App Server (V1)](./openhands/app_server/README.md)** - Current FastAPI application server and REST API modules
- **[Runtime](./openhands/runtime/README.md)** - Execution environments
- **[Evaluation](https://github.com/OpenHands/benchmarks)** - Testing and benchmarks

## What Can You Build?

### Frontend & UI/UX
- React & TypeScript development
- UI/UX improvements
- Mobile responsiveness
- Component libraries

For bigger changes, join the #proj-gui channel in [Slack](https://openhands.dev/joinslack) first.

### Agent Development
- Prompt engineering
- New agent types
- Agent evaluation
- Multi-agent systems

We use [SWE-bench](https://www.swebench.com/) to evaluate agents.

### Backend & Infrastructure
- Python development
- Runtime systems (Docker containers, sandboxes)
- Cloud integrations
- Performance optimization

### Testing & Quality Assurance
- Unit testing
- Integration testing
- Bug hunting
- Performance testing

### Documentation & Education
- Technical documentation
- Translation
- Community support

## Pull Request Process

### Small Improvements
- Quick review and approval
- Ensure CI tests pass
- Include clear description of changes

### Core Agent Changes
These are evaluated based on:
- **Accuracy** - Does it make the agent better at solving problems?
- **Efficiency** - Does it improve speed or reduce resource usage?
- **Code Quality** - Is the code maintainable and well-tested?

Discuss major changes in [GitHub issues](https://github.com/OpenHands/OpenHands/issues) or [Slack](https://openhands.dev/joinslack) first.

## Sending Pull Requests to OpenHands

You'll need to fork our repository to send us a Pull Request. You can learn more
about how to fork a GitHub repo and open a PR with your changes in [this article](https://medium.com/swlh/forks-and-pull-requests-how-to-contribute-to-github-repos-8843fac34ce8).

You may also check out previous PRs in the [PR list](https://github.com/OpenHands/OpenHands/pulls).

### Pull Request Title Format

As described [here](https://github.com/commitizen/conventional-commit-types/blob/master/index.json), a valid PR title should begin with one of the following prefixes:

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code (white space, formatting, missing semicolons, etc.)
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `build`: Changes that affect the build system or external dependencies (example scopes: gulp, broccoli, npm)
- `ci`: Changes to our CI configuration files and scripts (example scopes: Travis, Circle, BrowserStack, SauceLabs)
- `chore`: Other changes that don't modify src or test files
- `revert`: Reverts a previous commit

For example, a PR title could be:
- `refactor: modify package path`
- `feat(frontend): xxxx`, where `(frontend)` means that this PR mainly focuses on the frontend component.

### Pull Request Description

- Explain what the PR does and why
- Link to related issues
- Include screenshots for UI changes
- If your changes are user-facing (e.g. a new feature in the UI, a change in behavior, or a bugfix),
  please include a short message that we can add to our changelog

## Becoming a Maintainer

For contributors who have made significant and sustained contributions to the project, there is a possibility of joining the maintainer team.
The process for this is as follows:

1. Any contributor who has made sustained and high-quality contributions to the codebase can be nominated by any maintainer. If you feel that you may qualify you can reach out to any of the maintainers that have reviewed your PRs and ask if you can be nominated.
2. Once a maintainer nominates a new maintainer, there will be a discussion period among the maintainers for at least 3 days.
3. If no concerns are raised the nomination will be accepted by acclamation, and if concerns are raised there will be a discussion and possible vote.

Note that just making many PRs does not immediately imply that you will become a maintainer. We will be looking at sustained high-quality contributions over a period of time, as well as good teamwork and adherence to our [Code of Conduct](./CODE_OF_CONDUCT.md).

## Need Help?

- **Slack**: [Join our community](https://openhands.dev/joinslack)
- **GitHub Issues**: [Open an issue](https://github.com/OpenHands/OpenHands/issues)
- **Email**: contact@openhands.dev
