export const ADD_AGENT_DOCS_URL = "https://docs.openhands.dev/";

/** Default on-disk location for project-scoped file-based agents. */
export const ADD_AGENT_PROJECT_DIR = ".agents/agents/";

/** Example agent definition shown in the "Add agent" docs modal. */
export const ADD_AGENT_EXAMPLE_FILE = `---
name: changelog-writer
description: Use to draft release notes from a range of git commits.
tools: [bash, str_replace_editor]
model: inherit
---

You are a changelog specialist. Summarize the given commits into concise,
user-facing release notes grouped by Added / Changed / Fixed.`;
