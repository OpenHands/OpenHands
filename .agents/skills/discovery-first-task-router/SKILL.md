# Discovery-First Task Router

## Trigger

Use before any non-trivial coding, design, debugging, integration, migration, deployment, or repository-maintenance task.

## Goal

Convert the request into a precise, low-risk execution path before changing files.

## Workflow

1. Read the root and nearest `AGENTS.md` files.
2. Inspect repository status, architecture, relevant configuration, and existing implementations.
3. Translate the request into explicit acceptance criteria.
4. Identify required tools, existing repository capabilities, security constraints, and likely failure modes.
5. Search locally before adding dependencies or new abstractions.
6. Choose the smallest compatible implementation.
7. Plan validation before editing.
8. Execute incrementally, preserving unrelated work.
9. Critique the result against the acceptance criteria and fix clear gaps.
10. Report changed files, checks run, failures, assumptions, and remaining manual steps.

## Safety

- Never commit credentials or reveal complete secrets.
- Never use destructive Git commands to clean the working tree.
- Never delete tracked files without evidence they are obsolete.
- Do not claim tests passed unless they were executed successfully.
- Prefer upstream-compatible extension points over hard forks.

## Completion Format

- Outcome
- Files changed
- Validation
- Manual steps
- Risks or unresolved items
