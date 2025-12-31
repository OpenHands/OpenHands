---
title: Context
description: Skills and knowledge that agents can rely on during conversations. Provides repository context and structured knowledge.
---

# Context

Context provides skills and knowledge the agent can rely on during a conversation.

## Key Components

- **AgentContext**: Composes skills; pass to Agent to condition behavior
- **Skill**: Embeds structured knowledge with different trigger types:
  - **trigger=None**: Activates for all conversations (repository-wide context)
  - **KeywordTrigger**: Activates when specific keywords appear in user messages
  - **TaskTrigger**: Activates based on task-specific conditions

## Quick Example

```python
from openhands.sdk.context import AgentContext, KeywordTrigger, Skill

agent_context = AgentContext(
    skills=[
        Skill(
            name="repo-guidelines",
            content="Repository-wide coding standards and best practices.",
            source="repo.md",
            trigger=None,  # Always-active skill
        ),
        Skill(
            name="flarglebargle",
            content="If the user says flarglebargle, compliment them.",
            source="flarglebargle.md",
            trigger=KeywordTrigger(keywords=["flarglebargle"]),
        ),
    ]
)
```
