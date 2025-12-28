---
name: npm
type: knowledge
version: 1.0.0
agent: CodeActAgent
triggers:
- npm
---

When using npm to install packages, you will not be able to use an interactive shell, and it may be hard to confirm your actions.
As an alternative, you can pipe in the output of the unix "yes" command to confirm your actions.
