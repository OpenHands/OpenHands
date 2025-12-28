# OpenHands Skills

Skills are specialized prompts that enhance OpenHands with domain-specific knowledge and task-specific workflows. They help developers by providing expert guidance, automating common tasks, and ensuring consistent practices across projects. Each skill is designed to excel in a specific area, from Git operations to code review processes.

## Terminology Note

**Version 0 (V0)**: The term "microagents" continues to be used for V0 conversations. V0 is the current stable version of OpenHands.

**Version 1 (V1)**: The term "skills" is used for V1 conversations. V1 UI and app server have not yet been released, but the codebase has been updated to use "skills" terminology in preparation for the V1 release.

This directory (`OpenHands/skills/`) contains shareable skills that will be used in V1 conversations. For V0 conversations, the system continues to use microagents from the same underlying files.

## Sources of Skills/Microagents

OpenHands loads skills (V1) or microagents (V0) from two sources:

### 1. Shareable Skills/Microagents (Public)

This directory (`OpenHands/skills/`) contains shareable skills (V1) or microagents (V0) that are:

- Available to all OpenHands users
- Maintained in the OpenHands repository
- Perfect for reusable knowledge and common workflows
- Used as "skills" in V1 conversations and "microagents" in V0 conversations

Directory structure:

```
OpenHands/skills/
├── # Keyword-triggered expertise
│   ├── git.md         # Git operations
│   ├── testing.md     # Testing practices
│   └── docker.md      # Docker guidelines
└── # These skills/microagents are always loaded
    ├── pr_review.md   # PR review process
    ├── bug_fix.md     # Bug fixing workflow
    └── feature.md     # Feature implementation
```

### 2. Repository Instructions (Private)

Each repository can have its own instructions in `.openhands/microagents/` (V0) or `.openhands/skills/` (V1). These instructions are:

- Private to that repository
- Automatically loaded when working with that repository
- Perfect for repository-specific guidelines and team practices
- V1 supports both `.openhands/skills/` (preferred) and `.openhands/microagents/` (backward compatibility)

Example repository structure:

```
your-repository/
└── .openhands/
    ├── skills/        # V1: Preferred location for repository-specific skills
    │   └── repo.md    # Repository-specific instructions
    │   └── ...        # Private skills that are only available inside this
    └── microagents/   # V0: Current location (also supported in V1 for backward compatibility)
        └── repo.md    # Repository-specific instructions
        └── ...        # Private micro-agents that are only available inside this repo
```

## Loading Order

When OpenHands works with a repository, it:

1. Loads repository-specific instructions from `.openhands/microagents/repo.md` (V0) or `.openhands/skills/` (V1) if present
2. Loads relevant knowledge agents based on keywords in conversations

**Note**: V1 also supports loading from `.openhands/microagents/` for backward compatibility.

## Types of Skills/Microagents

Most skills/microagents use markdown files with YAML frontmatter. For repository agents (repo.md), the frontmatter is optional - if not provided, the file will be loaded with default settings as a repository agent.

### 1. Knowledge Agents

Knowledge agents provide specialized expertise that's triggered by keywords in conversations. They help with:

- Language best practices
- Framework guidelines
- Common patterns
- Tool usage

Key characteristics:

- **Trigger-based**: Activated by specific keywords in conversations
- **Context-aware**: Provide relevant advice based on file types and content
- **Reusable**: Knowledge can be applied across multiple projects
- **Versioned**: Support multiple versions of tools/frameworks

You can see an example of a knowledge-based agent in [OpenHands's github skill](https://github.com/OpenHands/OpenHands/tree/main/skills/github.md).

### 2. Repository Agents

Repository agents provide repository-specific knowledge and guidelines. They are:

- Loaded from `.openhands/microagents/repo.md` (V0) or `.openhands/skills/` directory (V1)
- V1 also supports `.openhands/microagents/` for backward compatibility
- Specific to individual repositories
- Automatically activated for their repository
- Perfect for team practices and project conventions

Key features:

- **Project-specific**: Contains guidelines unique to the repository
- **Team-focused**: Enforces team conventions and practices
- **Always active**: Automatically loaded for the repository
- **Locally maintained**: Updated with the project

You can see an example of a repo agent in [the agent for the OpenHands repo itself](https://github.com/OpenHands/OpenHands/blob/main/.openhands/microagents/repo.md).

## Contributing

### When to Contribute

1. **Knowledge Agents** - When you have:

   - Language/framework best practices
   - Tool usage patterns
   - Common problem solutions
   - General development guidelines

2. **Repository Agents** - When you need:
   - Project-specific guidelines
   - Team conventions and practices
   - Custom workflow documentation
   - Repository-specific setup instructions

### Best Practices

1. **For Knowledge Agents**:

   - Choose distinctive triggers
   - Focus on one area of expertise
   - Include practical examples
   - Use file patterns when relevant
   - Keep knowledge general and reusable

2. **For Repository Agents**:
   - Document clear setup instructions
   - Include repository structure details
   - Specify testing and build procedures
   - List environment requirements
   - Document CI workflows and checks
   - Include information about code quality standards
   - Maintain up-to-date team practices
   - Consider using OpenHands to generate a comprehensive repo.md (see [Creating a Repository Agent](#creating-a-repository-agent))
   - YAML frontmatter is optional - files without frontmatter will be loaded with default settings

### Submission Process

1. Create your agent file in the appropriate directory:
   - `skills/` for expertise (public, shareable)
   - Note: Repository-specific agents should remain in their respective repositories' `.openhands/skills/` (V1) or `.openhands/microagents/` (V0) directory
2. Test thoroughly
3. Submit a pull request to OpenHands

## License

All skills/microagents are subject to the same license as OpenHands. See the root LICENSE file for details.
