---
name: refresh-knowledge
trigger_type: always
---

# Build.One Framework Knowledge

Read the knowledge files to understand the Build.One framework before working on tasks.

## Instructions

1. Resolve the knowledge files path from the repository `.env`:
   ```
   KNOWLEDGE_PATH=$(grep -s '^BUILDONE_KNOWLEDGE_FILES_PATH=' /workspace/.env | head -1 | cut -d'=' -f2- | tr -d '"' | tr -d "'"); \
   if [ -z "$KNOWLEDGE_PATH" ]; then KNOWLEDGE_PATH="/knowledge"; \
   elif [[ "$KNOWLEDGE_PATH" != /* ]]; then KNOWLEDGE_PATH="/workspace/$KNOWLEDGE_PATH"; fi; \
   echo "$KNOWLEDGE_PATH"
   ```

2. Read the main knowledge index:
   ```
   cat KNOWLEDGE_PATH/CLAUDE.md
   ```

3. Based on the task at hand, read relevant documentation:

   **For architecture, DevOps, CLI, deployment:**
   ```
   cat KNOWLEDGE_PATH/architecture_info/CLAUDE.md
   ```

   **For Blueprint DSL, screens, forms, grids:**
   ```
   cat KNOWLEDGE_PATH/blueprint_dsl/CLAUDE.md
   ```

4. Follow the reading order specified in each CLAUDE.md file to understand the concepts.

## Key Concepts

- **Blueprint**: Model-based DSL stored in database, interpreted by rendering engine at runtime
- **Blueprint MCP**: ALWAYS use MCP tools to query/update blueprints, NEVER edit JSON files directly
- **Swat Samples Module**: ONLY use objects from this module for patterns and best practices

## Knowledge Location

The knowledge path is configured via `BUILDONE_KNOWLEDGE_FILES_PATH` in the repository's `/workspace/.env` file. If the variable is missing or the `.env` file does not exist, it falls back to `/knowledge/`. Relative paths are resolved from `/workspace/`.

Read the appropriate files based on what you need to accomplish.
