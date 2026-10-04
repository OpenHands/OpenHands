# F08 — Workspace drawer: files and changes

DRAFT in progress (F08 mapping agent): only the bullets below with commands were driven live so far.

Source: `src/components/features/conversation/conversation-tabs/`, `src/components/features/conversation/right-panel-toggle.tsx`, `src/components/features/conversation/conversation-main/`, `src/routes/files-tab.tsx`, `src/components/features/files-tab/`, `src/routes/commits-tab.tsx`, `src/components/features/diff-viewer/`.

## Sub-features

- `F08.panel-toggle`: the chat header toggle opens and closes the drawer; it defaults to Files.
- `F08.tab-bar`: drawer tabs switch the body; clicking the active tab closes the drawer.
- `F08.drawer-state-reload`: after a reload the drawer is closed; the selected tab is restored when reopened.
- `F08.tabs-menu`: the ellipsis menu opens any tab.
- `F08.tabs-pin`: pin/unpin tabs from the bar; persists.
- `F08.drawer-resize`: drag the divider (blocked: no drag verb).
- `F08.files-workspace-path`: workspace path row with copy.
- `F08.files-tree`: collapsible tree.
- `F08.files-tree-toggle`: hide/show tree, persisted.
- `F08.files-open-tabs`: open-file strip, close, persisted.
- `F08.files-no-selection`: empty content message.
- `F08.files-rich-plain`: Rich/Plain rendering.
- `F08.uncommitted-changes`: Uncommitted row.
- `F08.commit-rows`: commit rows.
- `F08.diff-view-modes`: old/diff/new.
- `F08.diff-markdown-preview`: markdown preview.
- `F08.diff-deleted-file`: deleted-file message.

## How to get to it (user POV)

- Chat header toggle `right-panel-toggle`.

## Driving it with control-openhands

Preconditions:

- Draft.

## Gotchas

- Draft.
