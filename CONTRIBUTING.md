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
- **Configuration**: `make setup-config`
- **Run locally**: `make run`

Full details in our [Development Guide](https://github.com/OpenHands/OpenHands/blob/main/Development.md).

### Find Your First Issue

- Browse [good first issues](https://github.com/OpenHands/OpenHands/labels/good%20first%20issue)
- Check our [project boards](https://github.com/OpenHands/OpenHands/projects) for organized tasks
- Join our [Slack community](https://openhands.dev/joinslack) to ask what needs help

## Understanding the Codebase

- **[Frontend](https://github.com/OpenHands/OpenHands/tree/main/frontend/README.md)** - React application
- **[Backend](https://github.com/OpenHands/OpenHands/tree/main/openhands/README.md)** - Python core
- **[Agents](https://github.com/OpenHands/OpenHands/tree/main/openhands/agenthub/README.md)** - AI agent implementations
- **[Runtime](https://github.com/OpenHands/OpenHands/tree/main/openhands/runtime/README.md)** - Execution environments
- **[Evaluation](https://github.com/OpenHands/benchmarks)** - Testing and benchmarks

## What Can You Build?

### Frontend & UI/UX
- React & TypeScript development
- UI/UX improvements
- Mobile responsiveness
- Component libraries

For bigger changes, join the #eng-ui-ux channel in [Slack](https://openhands.dev/joinslack) first.

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

## Pull Request Guidelines

### Title Format
- `feat: Add new agent capability`
- `fix: Resolve memory leak in runtime`
- `docs: Update installation guide`
- `style: Fix code formatting`
- `refactor: Simplify authentication logic`
- `test: Add unit tests for parser`

### Description
- Explain what the PR does and why
- Link to related issues
- Include screenshots for UI changes
- Add changelog entry for user-facing changes

## License

OpenHands is released under the **MIT License**:

### You Can
- Use OpenHands for any purpose, including commercial projects
- Modify the code to fit your needs
- Share your modifications
- Distribute or sell copies

### You Must
- Include the original copyright notice and license text
- Preserve the license in any substantial portions you use

### No Warranty
- OpenHands is provided "as is" without warranty
- Contributors are not liable for any damages

Full license text: [LICENSE](https://github.com/OpenHands/OpenHands/blob/main/LICENSE)

Note: Content in the `enterprise/` directory has a separate license. See `enterprise/LICENSE`.

## Need Help?

- **Slack**: [Join our community](https://openhands.dev/joinslack)
- **GitHub Issues**: [Open an issue](https://github.com/OpenHands/OpenHands/issues)
- **Email**: contact@openhands.dev
