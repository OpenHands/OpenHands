# Contributing

Thanks for your interest in contributing to OpenHands! We welcome and appreciate contributions.

## Understanding OpenHands's CodeBase

To understand the codebase, please refer to the README in each module:
- [frontend](./frontend/README.md)
- [evaluation](./evaluation/README.md)
- [openhands](./openhands/README.md)
   - [agenthub](./openhands/agenthub/README.md)
   - [server](./openhands/server/README.md)

## Setting up Your Development Environment

We have a separate doc [Development.md](https://github.com/OpenHands/OpenHands/blob/main/Development.md) that tells
you how to set up a development workflow.

## How Can I Contribute?

There are many ways that you can contribute:

1. **Download and use** OpenHands, and send [issues](https://github.com/OpenHands/OpenHands/issues) when you encounter something that isn't working or a feature that you'd like to see.
2. **Send feedback** after each session by [clicking the thumbs-up thumbs-down buttons](https://docs.openhands.dev/usage/feedback), so we can see where things are working and failing, and also build an open dataset for training code agents.
3. **Improve the Codebase** by sending [PRs](#sending-pull-requests-to-openhands) (see details below). In particular, we have some [good first issues](https://github.com/OpenHands/OpenHands/labels/good%20first%20issue) that may be ones to start on.

## What Can I Build?

Here are a few ways you can help improve the codebase.

#### UI/UX

We're always looking to improve the look and feel of the application. If you've got a small fix
for something that's bugging you, feel free to open up a PR that changes the [`./frontend`](./frontend) directory.

If you're looking to make a bigger change, add a new UI element, or significantly alter the style
of the application, please open an issue first, or better, join the #dev-ui-ux channel in our Slack
to gather consensus from our design team first.

#### Improving the agent

Our main agent is the CodeAct agent. You can [see its prompts here](https://github.com/OpenHands/OpenHands/tree/main/openhands/agenthub/codeact_agent).

Changes to these prompts, and to the underlying behavior in Python, can have a huge impact on user experience.
You can try modifying the prompts to see how they change the behavior of the agent as you use the app
locally, but we will need to do an end-to-end evaluation of any changes here to ensure that the agent
is getting better over time.

We use the [SWE-bench](https://www.swebench.com/) benchmark to test our agent. You can join the #evaluation
channel in Slack to learn more.

#### Adding a new agent

You may want to experiment with building new types of agents. You can add an agent to [`openhands/agenthub`](./openhands/agenthub)
to help expand the capabilities of OpenHands.

#### Adding a new runtime

The agent needs a place to run code and commands. When you run OpenHands on your laptop, it uses a Docker container
to do this by default. But there are other ways of creating a sandbox for the agent.

If you work for a company that provides a cloud-based runtime, you could help us add support for that runtime
by implementing the [interface specified here](https://github.com/OpenHands/OpenHands/blob/main/openhands/runtime/base.py).

#### Testing

When you write code, it is also good to write tests. Please navigate to the [`./tests`](./tests) folder to see existing
test suites. At the moment, we have these kinds of tests: [`unit`](./tests/unit), [`runtime`](./tests/runtime), and [`end-to-end (e2e)`](./tests/e2e).
Please refer to the README for each test suite. These tests also run on GitHub's continuous integration to ensure
quality of the project.

## Sending Pull Requests to OpenHands

You'll need to fork our repository to send us a Pull Request. You can learn more
about how to fork a GitHub repo and open a PR with your changes in [this article](https://medium.com/swlh/forks-and-pull-requests-how-to-contribute-to-github-repos-8843fac34ce8).

### Pull Request title

As described [here](https://github.com/commitizen/conventional-commit-types/blob/master/index.json), ideally a valid PR title should begin with one of the following prefixes:

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

You may also check out previous PRs in the [PR list](https://github.com/OpenHands/OpenHands/pulls).

### Pull Request description

- If your PR is small (such as a typo fix), you can go brief.
- If it contains a lot of changes, it's better to write more details.

If your changes are user-facing (e.g. a new feature in the UI, a change in behavior, or a bugfix)
please include a short message that we can add to our changelog.

## How to Make Effective Contributions

### Opening Issues

If you notice any bugs or have any feature requests please open them via the [issues page](https://github.com/OpenHands/OpenHands/issues). We will triage
based on how critical the bug is or how potentially useful the improvement is, discuss, and implement the ones that
the community has interest/effort for.

Further, if you see an issue you like, please leave a "thumbs-up" or a comment, which will help us prioritize.

### Making Pull Requests

We're generally happy to consider all pull requests with the evaluation process varying based on the type of change:

#### For Small Improvements

Small improvements with few downsides are typically reviewed and approved quickly.
One thing to check when making changes is to ensure that all continuous integration tests pass, which you can check
before getting a review.

#### For Core Agent Changes

We need to be more careful with changes to the core agent, as it is imperative to maintain high quality. These PRs are
evaluated based on three key metrics:

1. **Accuracy**
2. **Efficiency**
3. **Code Complexity**

If it improves accuracy, efficiency, or both with only a minimal change to code quality, that's great we're happy to merge it in!
If there are bigger tradeoffs (e.g. helping efficiency a lot and hurting accuracy a little) we might want to put it behind a feature flag.
Either way, please feel free to discuss on github issues or slack, and we will give guidance and preliminary feedback.
