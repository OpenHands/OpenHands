---
name: agent_memory
type: knowledge
version: 1.0.0
agent: CodeActAgent
triggers:
- /remember
---

* Repository memory: Use .openhands/microagents/repo.md under each repository root to store and access important information.
  - If this file exists, it will be added to your context automatically.
  - If missing, you should create it unless the user has explicitly asked you to not do so.

* Store and maintain **general knowledge** that will be helpful for most future tasks:
  1. Repository structure
  2. Common commands (build, lint, test, pre-commit, etc.)
  3. Code style preferences
  4. Workflows and best practices
  5. Any other repository-specific knowledge you learn

* IMPORTANT: ONLY LOG the information that would be helpful for different future tasks, for example, how to configure the settings, how to setup the repository. Do NOT add issue-specific information (e.g., what specific error you have ran into and how you fix it).

* When adding new information:
  - ALWAYS ask for user confirmation first by listing the exact items (numbered 1, 2, 3, etc.) you plan to save to repo.md
  - Only save the items the user approves (they may ask you to save a subset)
  - Ensure it integrates nicely with existing knowledge in repo.md
  - Reorganize the content if needed to maintain clarity and organization
  - Group related information together under appropriate sections or headings
  - If you've only explored a portion of the codebase, clearly note this limitation in the repository structure documentation
  - If you don't know the essential commands for working with the repository, such as lint or typecheck, ask the user and suggest adding them to repo.md for future reference (with permission)

When you receive this message, please review and summarize your recent actions and observations, then present a list of valuable information that should be saved in repo.md to the user.
