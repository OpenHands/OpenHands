# Workspace File Discovery

### WFD-001: Configurable local workspace discovery

- With no saved configuration, list at most 2,000 regular files using the existing directory exclusions.
- Users can replace directory exclusions with name or workspace-relative path patterns. `*/bin` excludes nested `bin` directories while preserving root `bin`.
- Quote every configured pattern as shell data. Patterns cannot execute commands.
- A zero file limit means unlimited. A positive integer bounds the returned paths.
- Optional symlink inclusion lists links to regular files without traversing directory links.
- Read one extra path for a finite limit. Show an incomplete-tree message only when more paths exist than the configured limit.
- Cloud file listing keeps its existing request and behavior.

### WFD-002: Workspace-scoped server persistence

- The local backend owns `misc_settings.app_preferences.workspace_file_discovery`, keyed by working directory.
- Save a sparse map containing only the edited workspace so other workspace preferences survive the server's deep merge.
- Conversations sharing a workspace use its configuration. A different workspace or backend uses its own configuration or defaults.
- Saved preferences survive reload. Missing or invalid values fall back to conservative defaults.
- Reset prepares the defaults for Save; Cancel discards the form draft. Failed saves keep the dialog open with an error.
- Switching workspace or backend discards the open editor. Cloud never displays the editor or saves these settings.
