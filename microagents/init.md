---
name: init
type: task
version: 1.0.0
agent: CodeActAgent
triggers:
- /init
---

Please browse the repository, look at the documentation and relevant code, and understand the purpose of this repository.

Specifically, I want you to create a `.openhands/microagents/repo.md` file. This file should contain succinct information that summarizes:
1. The purpose of this repository
2. The general setup of this repo
3. A brief description of the structure of this repo

Read all the GitHub workflows under .github/ of the repository (if this folder exists) to understand the CI checks (e.g., linter, pre-commit), and include those in the repo.md file.

The repo.md file should be a microagent with type "knowledge" that provides context about the repository for future conversations.
