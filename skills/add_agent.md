---
name: add_agent
type: knowledge
version: 1.0.0
agent: CodeActAgent
triggers:
  - new agent
  - new microagent
  - create agent
  - create an agent
  - create microagent
  - create a microagent
  - add agent
  - add an agent
  - add microagent
  - add a microagent
  - microagent template
---

This agent helps create new microagents in the `.openhands/microagents` directory by providing guidance and templates.

Microagents are specialized prompts that provide context and capabilities for specific domains or tasks. They are activated by trigger words in the conversation and help the AI assistant understand what capabilities are available, how to use specific APIs or tools, what limitations exist, and how to handle common scenarios.

When creating a new microagent:

- Create a markdown file in `.openhands/microagents/` with an appropriate name (e.g., `github.md`, `google_workspace.md`)
- Include YAML frontmatter with metadata (name, type, version, agent, triggers)
- type is by DEFAULT knowledge
- version is DEFAULT 1.0.0
- agent is by DEFAULT CodeActAgent
- Document any credentials, environment variables, or API access needed
- Keep trigger words specific to avoid false activations
- Include error handling guidance and limitations
- Provide clear usage examples
- Keep the prompt focused and concise

For detailed information, see:

- [Microagents Overview](https://docs.all-hands.dev/usage/prompting/microagents-overview)
- [Example GitHub Skill](https://github.com/OpenHands/OpenHands/blob/main/skills/github.md)
