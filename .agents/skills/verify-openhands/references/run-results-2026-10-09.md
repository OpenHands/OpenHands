# 2026-10-09 agentic sweep: per-ID outcomes

Frozen target: `OpenHands/OpenHands main@3500d9e5c7cbdfb5a2cbfe9499ce4448eb10c0a2`. These results belong to the [dated run report](run-history.md); the accepted maintenance baseline remains unchanged.

Total: 736 IDs; pass 534, fail 40, blocked 162, not-run 0, missing 0.

`pass` means the mapped behavior was observed on the target; `fail` means a reproduced mismatch; `blocked` means a named prerequisite was unavailable; `not-run` means no live action established the behavior. `missing` means no reconciled ledger entry and must be zero before this table is final. The run report links confirmed bugs and explains environment limits.

## F01

| Map ID | Result | Triage |
| --- | --- | --- |
| [F01.first-run-gate](feature-map/F01-first-run-and-sign-in.md) | pass | — |
| [F01.telemetry-consent](feature-map/F01-first-run-and-sign-in.md) | pass | — |
| [F01.onboarding-modal](feature-map/F01-first-run-and-sign-in.md) | pass | — |
| [F01.onboarding-backend-step](feature-map/F01-first-run-and-sign-in.md) | pass | — |
| [F01.onboarding-choose-agent](feature-map/F01-first-run-and-sign-in.md) | pass | — |
| [F01.onboarding-setup-llm](feature-map/F01-first-run-and-sign-in.md) | blocked | — |
| [F01.onboarding-acp-secrets](feature-map/F01-first-run-and-sign-in.md) | pass | — |
| [F01.onboarding-say-hello](feature-map/F01-first-run-and-sign-in.md) | blocked | — |
| [F01.onboarding-hello-close](feature-map/F01-first-run-and-sign-in.md) | pass | — |
| [F01.onboarding-recommended-automation](feature-map/F01-first-run-and-sign-in.md) | pass | — |
| [F01.onboarding-skip-checklist](feature-map/F01-first-run-and-sign-in.md) | pass | — |
| [F01.onboarding-skip](feature-map/F01-first-run-and-sign-in.md) | pass | — |
| [F01.onboarding-preview](feature-map/F01-first-run-and-sign-in.md) | pass | — |
| [F01.onboarding-phone](feature-map/F01-first-run-and-sign-in.md) | pass | — |
| [F01.api-key-entry](feature-map/F01-first-run-and-sign-in.md) | pass | — |
| [F01.onboarding-cloud-login](feature-map/F01-first-run-and-sign-in.md) | blocked | — |
| [F01.route-error-boundary](feature-map/F01-first-run-and-sign-in.md) | pass | — |
| [F01.document-title](feature-map/F01-first-run-and-sign-in.md) | blocked | — |
| [F01.bootstrap-loading](feature-map/F01-first-run-and-sign-in.md) | blocked | — |
| [F01.locked-cloud-first-run](feature-map/F01-first-run-and-sign-in.md) | blocked | — |
| [F01.cookie-auth-redirect](feature-map/F01-first-run-and-sign-in.md) | blocked | — |

## F02

| Map ID | Result | Triage |
| --- | --- | --- |
| [F02.nav-links](feature-map/F02-app-shell.md) | pass | — |
| [F02.pin-as-home](feature-map/F02-app-shell.md) | pass | — |
| [F02.sidebar-collapse](feature-map/F02-app-shell.md) | pass | — |
| [F02.settings-link](feature-map/F02-app-shell.md) | pass | — |
| [F02.conversation-list](feature-map/F02-app-shell.md) | pass | — |
| [F02.command-menu-open-close](feature-map/F02-app-shell.md) | pass | — |
| [F02.command-menu-focus](feature-map/F02-app-shell.md) | pass | — |
| [F02.command-menu-items](feature-map/F02-app-shell.md) | pass | — |
| [F02.command-menu-toggle-sidebar](feature-map/F02-app-shell.md) | pass | — |
| [F02.command-menu-search](feature-map/F02-app-shell.md) | pass | — |
| [F02.command-menu-new-tab](feature-map/F02-app-shell.md) | pass | — |
| [F02.command-menu-phone](feature-map/F02-app-shell.md) | pass | — |
| [F02.mobile-drawer](feature-map/F02-app-shell.md) | pass | — |
| [F02.mobile-top-bar](feature-map/F02-app-shell.md) | pass | — |
| [F02.checklist](feature-map/F02-app-shell.md) | pass | — |
| [F02.checklist-item-links](feature-map/F02-app-shell.md) | pass | — |
| [F02.checklist-progress](feature-map/F02-app-shell.md) | pass | — |
| [F02.checklist-preview](feature-map/F02-app-shell.md) | pass | — |
| [F02.checklist-minimize](feature-map/F02-app-shell.md) | pass | — |
| [F02.checklist-hide](feature-map/F02-app-shell.md) | pass | — |
| [F02.checklist-all-complete](feature-map/F02-app-shell.md) | pass | — |
| [F02.update-tile](feature-map/F02-app-shell.md) | blocked | — |
| [F02.alert-banner](feature-map/F02-app-shell.md) | blocked | — |
| [F02.settings-404-modal](feature-map/F02-app-shell.md) | blocked | — |
| [F02.error-toasts](feature-map/F02-app-shell.md) | pass | — |

## F03

| Map ID | Result | Triage |
| --- | --- | --- |
| [F03.home-entry-points](feature-map/F03-home.md) | pass | — |
| [F03.llm-not-configured-banner](feature-map/F03-home.md) | pass | — |
| [F03.composer-blocked-without-llm](feature-map/F03-home.md) | pass | — |
| [F03.draft-persistence](feature-map/F03-home.md) | pass | — |
| [F03.home-model-command](feature-map/F03-home.md) | pass | — |
| [F03.launch-in-workspace](feature-map/F03-home.md) | pass | — |
| [F03.launch-without-workspace](feature-map/F03-home.md) | pass | — |
| [F03.create-error-toast](feature-map/F03-home.md) | pass | — |
| [F03.plugin-picker](feature-map/F03-home.md) | pass | — |
| [F03.launch-with-plugin](feature-map/F03-home.md) | pass | — |
| [F03.open-workspace-dialog](feature-map/F03-home.md) | pass | — |
| [F03.workspace-dropdown-search](feature-map/F03-home.md) | pass | — |
| [F03.selection-after-reload](feature-map/F03-home.md) | pass | — |
| [F03.workspace-mode](feature-map/F03-home.md) | pass | — |
| [F03.folder-browser](feature-map/F03-home.md) | pass | — |
| [F03.folder-browser-locations](feature-map/F03-home.md) | pass | — |
| [F03.manage-workspaces](feature-map/F03-home.md) | pass | — |
| [F03.open-repository-dialog](feature-map/F03-home.md) | blocked | — |
| [F03.isolated-workspace-notice](feature-map/F03-home.md) | blocked | — |
| [F03.workspaces-unsupported](feature-map/F03-home.md) | blocked | — |
| [F03.recommended-automations-rail](feature-map/F03-home.md) | pass | — |
| [F03.recommended-responder-choice](feature-map/F03-home.md) | pass | — |
| [F03.responder-cloud-option](feature-map/F03-home.md) | pass | — |
| [F03.recommended-missing-integration](feature-map/F03-home.md) | pass | — |
| [F03.recommended-hides-added](feature-map/F03-home.md) | pass | — |
| [F03.recommended-prompt-launch](feature-map/F03-home.md) | blocked | — |
| [F03.home-automations-list](feature-map/F03-home.md) | pass | — |
| [F03.home-automation-links](feature-map/F03-home.md) | pass | — |
| [F03.home-run-tooltip](feature-map/F03-home.md) | pass | — |
| [F03.home-row-menu](feature-map/F03-home.md) | pass | — |
| [F03.cancel-in-flight-run](feature-map/F03-home.md) | pass | — |
| [F03.turn-off-confirmation](feature-map/F03-home.md) | pass | — |
| [F03.pinned-automations-grid](feature-map/F03-home.md) | pass | — |
| [F03.pinned-card-menu](feature-map/F03-home.md) | pass | — |
| [F03.pinned-card-links](feature-map/F03-home.md) | pass | — |
| [F03.pinned-reorder-drag](feature-map/F03-home.md) | pass | — |
| [F03.home-automations-view-more](feature-map/F03-home.md) | pass | — |
| [F03.automations-backend-down](feature-map/F03-home.md) | pass | — |
| [F03.phone](feature-map/F03-home.md) | pass | — |

## F04

| Map ID | Result | Triage |
| --- | --- | --- |
| [F04.empty-state](feature-map/F04-conversation-list.md) | pass | — |
| [F04.view-menu](feature-map/F04-conversation-list.md) | pass | — |
| [F04.new-thread-picker](feature-map/F04-conversation-list.md) | pass | — |
| [F04.card](feature-map/F04-conversation-list.md) | pass | — |
| [F04.card-live-status](feature-map/F04-conversation-list.md) | blocked | — |
| [F04.list-header](feature-map/F04-conversation-list.md) | pass | — |
| [F04.organize](feature-map/F04-conversation-list.md) | pass | — |
| [F04.grouped-folders](feature-map/F04-conversation-list.md) | pass | — |
| [F04.folder-collapse](feature-map/F04-conversation-list.md) | pass | — |
| [F04.collapse-all](feature-map/F04-conversation-list.md) | pass | — |
| [F04.folder-new-conversation](feature-map/F04-conversation-list.md) | pass | — |
| [F04.folder-preview-more](feature-map/F04-conversation-list.md) | pass | — |
| [F04.folder-reorder](feature-map/F04-conversation-list.md) | pass | — |
| [F04.thread-scope](feature-map/F04-conversation-list.md) | pass | — |
| [F04.load-more](feature-map/F04-conversation-list.md) | pass | — |
| [F04.sort](feature-map/F04-conversation-list.md) | pass | — |
| [F04.presets](feature-map/F04-conversation-list.md) | pass | — |
| [F04.hide-older](feature-map/F04-conversation-list.md) | blocked | — |
| [F04.automation-filter](feature-map/F04-conversation-list.md) | blocked | — |
| [F04.tag-filter](feature-map/F04-conversation-list.md) | pass | — |
| [F04.filter-chips](feature-map/F04-conversation-list.md) | pass | — |
| [F04.metadata-toggles](feature-map/F04-conversation-list.md) | pass | — |
| [F04.hover-preview](feature-map/F04-conversation-list.md) | pass | — |
| [F04.pin](feature-map/F04-conversation-list.md) | pass | — |
| [F04.pinned-preview-more](feature-map/F04-conversation-list.md) | pass | — |
| [F04.card-rename](feature-map/F04-conversation-list.md) | pass | — |
| [F04.edit-tags](feature-map/F04-conversation-list.md) | pass | — |
| [F04.edit-tags-remove](feature-map/F04-conversation-list.md) | pass | — |
| [F04.tag-overflow](feature-map/F04-conversation-list.md) | pass | — |
| [F04.stop](feature-map/F04-conversation-list.md) | pass | — |
| [F04.download](feature-map/F04-conversation-list.md) | pass | — |
| [F04.download-error](feature-map/F04-conversation-list.md) | pass | — |
| [F04.archive](feature-map/F04-conversation-list.md) | pass | — |
| [F04.show-archived-unarchive](feature-map/F04-conversation-list.md) | pass | — |
| [F04.delete](feature-map/F04-conversation-list.md) | pass | — |
| [F04.delete-all](feature-map/F04-conversation-list.md) | fail | [OpenHands/OpenHands#17919](https://github.com/OpenHands/OpenHands/issues/17919) |
| [F04.collapsed-rail](feature-map/F04-conversation-list.md) | pass | — |
| [F04.phone](feature-map/F04-conversation-list.md) | pass | — |
| [F04.cloud-picker](feature-map/F04-conversation-list.md) | blocked | — |
| [F04.start-task-cards](feature-map/F04-conversation-list.md) | blocked | — |

## F05

| Map ID | Result | Triage |
| --- | --- | --- |
| [F05.send-message](feature-map/F05-composer.md) | fail | [OpenHands/OpenHands#18202](https://github.com/OpenHands/OpenHands/issues/18202) |
| [F05.draft-persistence](feature-map/F05-composer.md) | pass | — |
| [F05.composer-resize](feature-map/F05-composer.md) | pass | — |
| [F05.plus-menu](feature-map/F05-composer.md) | pass | — |
| [F05.agent-profile-switch](feature-map/F05-composer.md) | pass | — |
| [F05.attach-files](feature-map/F05-composer.md) | pass | — |
| [F05.attach-size-limit](feature-map/F05-composer.md) | pass | — |
| [F05.image-only-send](feature-map/F05-composer.md) | pass | — |
| [F05.paste](feature-map/F05-composer.md) | pass | — |
| [F05.dictation](feature-map/F05-composer.md) | blocked | — |
| [F05.llm-profile-picker](feature-map/F05-composer.md) | pass | — |
| [F05.profile-identity](feature-map/F05-composer.md) | pass | — |
| [F05.overflow-menu](feature-map/F05-composer.md) | fail | [OpenHands/OpenHands#18172](https://github.com/OpenHands/OpenHands/issues/18172) |
| [F05.context-window-meter](feature-map/F05-composer.md) | blocked | — |
| [F05.stop-resume](feature-map/F05-composer.md) | blocked | — |
| [F05.queued-message](feature-map/F05-composer.md) | blocked | — |
| [F05.slash-menu](feature-map/F05-composer.md) | pass | — |
| [F05.slash-model](feature-map/F05-composer.md) | pass | — |
| [F05.slash-btw](feature-map/F05-composer.md) | fail | [OpenHands/OpenHands#17429](https://github.com/OpenHands/OpenHands/issues/17429) |
| [F05.slash-goal](feature-map/F05-composer.md) | blocked | — |
| [F05.slash-plan-code](feature-map/F05-composer.md) | pass | — |
| [F05.slash-plan-code-task](feature-map/F05-composer.md) | blocked | — |
| [F05.plan-preview](feature-map/F05-composer.md) | blocked | — |
| [F05.build-plan-shortcut](feature-map/F05-composer.md) | blocked | — |
| [F05.phone](feature-map/F05-composer.md) | pass | — |

## F06

| Map ID | Result | Triage |
| --- | --- | --- |
| [F06.empty-state-suggestions](feature-map/F06-agent-activity.md) | fail | [OpenHands/OpenHands#18202](https://github.com/OpenHands/OpenHands/issues/18202) |
| [F06.user-and-agent-messages](feature-map/F06-agent-activity.md) | pass | — |
| [F06.message-copy](feature-map/F06-agent-activity.md) | pass | — |
| [F06.code-block-copy](feature-map/F06-agent-activity.md) | blocked | — |
| [F06.timestamps](feature-map/F06-agent-activity.md) | pass | — |
| [F06.pending-messages](feature-map/F06-agent-activity.md) | pass | — |
| [F06.clock-skew-send](feature-map/F06-agent-activity.md) | pass | — |
| [F06.failed-send-persists](feature-map/F06-agent-activity.md) | pass | — |
| [F06.markdown-rendering](feature-map/F06-agent-activity.md) | blocked | — |
| [F06.workspace-path-links](feature-map/F06-agent-activity.md) | blocked | — |
| [F06.thinking](feature-map/F06-agent-activity.md) | blocked | — |
| [F06.event-groups](feature-map/F06-agent-activity.md) | blocked | — |
| [F06.tool-visualizers](feature-map/F06-agent-activity.md) | blocked | — |
| [F06.markdown-file-preview](feature-map/F06-agent-activity.md) | blocked | — |
| [F06.task-list](feature-map/F06-agent-activity.md) | blocked | — |
| [F06.events-match](feature-map/F06-agent-activity.md) | blocked | — |
| [F06.load-older-history](feature-map/F06-agent-activity.md) | pass | — |
| [F06.scroll-to-bottom](feature-map/F06-agent-activity.md) | pass | — |
| [F06.live-activity](feature-map/F06-agent-activity.md) | blocked | — |
| [F06.status-indicator](feature-map/F06-agent-activity.md) | pass | — |
| [F06.stop-resume](feature-map/F06-agent-activity.md) | blocked | — |
| [F06.reload-mid-run](feature-map/F06-agent-activity.md) | blocked | — |
| [F06.reconnect](feature-map/F06-agent-activity.md) | pass | — |
| [F06.error-banner](feature-map/F06-agent-activity.md) | pass | — |
| [F06.error-events](feature-map/F06-agent-activity.md) | fail | [OpenHands/OpenHands#17887](https://github.com/OpenHands/OpenHands/issues/17887) |
| [F06.confirmation-mode](feature-map/F06-agent-activity.md) | blocked | — |
| [F06.confirmation-shortcuts](feature-map/F06-agent-activity.md) | blocked | — |
| [F06.branch-from-here](feature-map/F06-agent-activity.md) | fail | [OpenHands/OpenHands#18202](https://github.com/OpenHands/OpenHands/issues/18202) |
| [F06.image-attachments](feature-map/F06-agent-activity.md) | pass | — |
| [F06.skill-install-banner](feature-map/F06-agent-activity.md) | blocked | — |
| [F06.llm-not-configured-banner](feature-map/F06-agent-activity.md) | pass | — |
| [F06.critic-result](feature-map/F06-agent-activity.md) | blocked | — |
| [F06.hook-events](feature-map/F06-agent-activity.md) | blocked | — |
| [F06.corrective-nudge](feature-map/F06-agent-activity.md) | blocked | — |
| [F06.phone](feature-map/F06-agent-activity.md) | pass | — |

## F07

| Map ID | Result | Triage |
| --- | --- | --- |
| [F07.open-by-url](feature-map/F07-conversation-page.md) | pass | — |
| [F07.missing-redirect](feature-map/F07-conversation-page.md) | pass | — |
| [F07.history-skeleton](feature-map/F07-conversation-page.md) | pass | — |
| [F07.panel-route](feature-map/F07-conversation-page.md) | pass | — |
| [F07.rename-inline](feature-map/F07-conversation-page.md) | pass | — |
| [F07.status-menu](feature-map/F07-conversation-page.md) | pass | — |
| [F07.menu-open](feature-map/F07-conversation-page.md) | pass | — |
| [F07.menu-items-by-state](feature-map/F07-conversation-page.md) | pass | — |
| [F07.skills-modal](feature-map/F07-conversation-page.md) | pass | — |
| [F07.hooks-modal](feature-map/F07-conversation-page.md) | blocked | — |
| [F07.skills-modal-project](feature-map/F07-conversation-page.md) | pass | — |
| [F07.agent-tools-modal](feature-map/F07-conversation-page.md) | pass | — |
| [F07.export-transcript](feature-map/F07-conversation-page.md) | pass | — |
| [F07.download-zip](feature-map/F07-conversation-page.md) | pass | — |
| [F07.display-cost](feature-map/F07-conversation-page.md) | pass | — |
| [F07.stop-confirm](feature-map/F07-conversation-page.md) | blocked | — |
| [F07.delete-confirm](feature-map/F07-conversation-page.md) | pass | — |
| [F07.branch-from-message](feature-map/F07-conversation-page.md) | pass | — |
| [F07.right-panel-toggle](feature-map/F07-conversation-page.md) | pass | — |
| [F07.overview-toggle](feature-map/F07-conversation-page.md) | pass | — |
| [F07.git-actions-menu](feature-map/F07-conversation-page.md) | fail | [OpenHands/OpenHands#18202](https://github.com/OpenHands/OpenHands/issues/18202) |
| [F07.git-control-bar](feature-map/F07-conversation-page.md) | pass | — |
| [F07.git-control-bar-send](feature-map/F07-conversation-page.md) | pass | — |
| [F07.cloud-only](feature-map/F07-conversation-page.md) | blocked | — |
| [F07.phone](feature-map/F07-conversation-page.md) | pass | — |

## F08

| Map ID | Result | Triage |
| --- | --- | --- |
| [F08.panel-toggle](feature-map/F08-workspace-files-and-changes.md) | pass | — |
| [F08.tab-bar](feature-map/F08-workspace-files-and-changes.md) | pass | — |
| [F08.drawer-state-reload](feature-map/F08-workspace-files-and-changes.md) | pass | — |
| [F08.tabs-menu](feature-map/F08-workspace-files-and-changes.md) | pass | — |
| [F08.tabs-pin](feature-map/F08-workspace-files-and-changes.md) | blocked | — |
| [F08.drawer-resize](feature-map/F08-workspace-files-and-changes.md) | blocked | — |
| [F08.vscode-link](feature-map/F08-workspace-files-and-changes.md) | blocked | — |
| [F08.archived-disabled](feature-map/F08-workspace-files-and-changes.md) | blocked | — |
| [F08.phone-panel-page](feature-map/F08-workspace-files-and-changes.md) | pass | — |
| [F08.panel-direct-url](feature-map/F08-workspace-files-and-changes.md) | pass | — |
| [F08.files-workspace-path](feature-map/F08-workspace-files-and-changes.md) | pass | — |
| [F08.files-tree](feature-map/F08-workspace-files-and-changes.md) | pass | — |
| [F08.files-empty-workspace](feature-map/F08-workspace-files-and-changes.md) | pass | — |
| [F08.files-tree-toggle](feature-map/F08-workspace-files-and-changes.md) | pass | — |
| [F08.files-tree-resize](feature-map/F08-workspace-files-and-changes.md) | blocked | — |
| [F08.files-open-tabs](feature-map/F08-workspace-files-and-changes.md) | pass | — |
| [F08.files-no-selection](feature-map/F08-workspace-files-and-changes.md) | pass | — |
| [F08.files-rich-plain](feature-map/F08-workspace-files-and-changes.md) | pass | — |
| [F08.files-fallbacks](feature-map/F08-workspace-files-and-changes.md) | blocked | — |
| [F08.files-load-error](feature-map/F08-workspace-files-and-changes.md) | blocked | — |
| [F08.files-open-new-window](feature-map/F08-workspace-files-and-changes.md) | pass | — |
| [F08.files-refresh](feature-map/F08-workspace-files-and-changes.md) | pass | — |
| [F08.files-auto-refresh](feature-map/F08-workspace-files-and-changes.md) | blocked | — |
| [F08.files-open-from-chat](feature-map/F08-workspace-files-and-changes.md) | blocked | — |
| [F08.files-open-from-tool-chip](feature-map/F08-workspace-files-and-changes.md) | blocked | — |
| [F08.commits-states](feature-map/F08-workspace-files-and-changes.md) | pass | — |
| [F08.uncommitted-changes](feature-map/F08-workspace-files-and-changes.md) | blocked | — |
| [F08.commit-rows](feature-map/F08-workspace-files-and-changes.md) | blocked | — |
| [F08.commits-cap](feature-map/F08-workspace-files-and-changes.md) | blocked | — |
| [F08.diff-view-modes](feature-map/F08-workspace-files-and-changes.md) | blocked | — |
| [F08.diff-markdown-preview](feature-map/F08-workspace-files-and-changes.md) | blocked | — |
| [F08.diff-deleted-file](feature-map/F08-workspace-files-and-changes.md) | blocked | — |
| [F08.changes-from-overview](feature-map/F08-workspace-files-and-changes.md) | blocked | — |

## F09

| Map ID | Result | Triage |
| --- | --- | --- |
| [F09.entry-gear](feature-map/F09-settings-shell.md) | pass | — |
| [F09.entry-collapsed-rail](feature-map/F09-settings-shell.md) | pass | — |
| [F09.entry-command-menu](feature-map/F09-settings-shell.md) | pass | — |
| [F09.command-menu-coverage](feature-map/F09-settings-shell.md) | pass | — |
| [F09.entry-deep-links](feature-map/F09-settings-shell.md) | pass | — |
| [F09.index-redirect](feature-map/F09-settings-shell.md) | pass | — |
| [F09.locked-cloud-landing](feature-map/F09-settings-shell.md) | blocked | — |
| [F09.legacy-agent-redirect](feature-map/F09-settings-shell.md) | pass | — |
| [F09.hidden-llm-redirect](feature-map/F09-settings-shell.md) | blocked | — |
| [F09.desktop-nav](feature-map/F09-settings-shell.md) | pass | — |
| [F09.page-header](feature-map/F09-settings-shell.md) | pass | — |
| [F09.landmarks](feature-map/F09-settings-shell.md) | pass | — |
| [F09.phone-hub](feature-map/F09-settings-shell.md) | pass | — |
| [F09.hub-resize-redirect](feature-map/F09-settings-shell.md) | pass | — |
| [F09.breakpoint-sweep](feature-map/F09-settings-shell.md) | pass | — |
| [F09.phone-back](feature-map/F09-settings-shell.md) | pass | — |
| [F09.synced-badge](feature-map/F09-settings-shell.md) | pass | — |
| [F09.cloud-links](feature-map/F09-settings-shell.md) | pass | — |
| [F09.update-card](feature-map/F09-settings-shell.md) | pass | — |
| [F09.update-modal](feature-map/F09-settings-shell.md) | pass | — |
| [F09.update-up-to-date](feature-map/F09-settings-shell.md) | pass | — |
| [F09.update-available](feature-map/F09-settings-shell.md) | blocked | — |
| [F09.update-modal-phone](feature-map/F09-settings-shell.md) | pass | — |
| [F09.sidebar-version-tile](feature-map/F09-settings-shell.md) | blocked | — |
| [F09.unknown-subpath](feature-map/F09-settings-shell.md) | pass | — |

## F10

| Map ID | Result | Triage |
| --- | --- | --- |
| [F10.list](feature-map/F10-llm-profiles.md) | pass | — |
| [F10.list-empty](feature-map/F10-llm-profiles.md) | pass | — |
| [F10.list-states](feature-map/F10-llm-profiles.md) | blocked | — |
| [F10.list-grouped](feature-map/F10-llm-profiles.md) | pass | — |
| [F10.broken-link](feature-map/F10-llm-profiles.md) | pass | — |
| [F10.actions-menu](feature-map/F10-llm-profiles.md) | pass | — |
| [F10.set-default](feature-map/F10-llm-profiles.md) | pass | — |
| [F10.duplicate](feature-map/F10-llm-profiles.md) | pass | — |
| [F10.rename](feature-map/F10-llm-profiles.md) | pass | — |
| [F10.rename-conflict](feature-map/F10-llm-profiles.md) | pass | — |
| [F10.delete](feature-map/F10-llm-profiles.md) | pass | — |
| [F10.delete-default](feature-map/F10-llm-profiles.md) | pass | — |
| [F10.create](feature-map/F10-llm-profiles.md) | blocked | — |
| [F10.create-prefill](feature-map/F10-llm-profiles.md) | fail | [OpenHands/OpenHands#17955](https://github.com/OpenHands/OpenHands/issues/17955) |
| [F10.api-key-help](feature-map/F10-llm-profiles.md) | pass | — |
| [F10.name-autofill](feature-map/F10-llm-profiles.md) | pass | — |
| [F10.name-validation](feature-map/F10-llm-profiles.md) | pass | — |
| [F10.create-validation](feature-map/F10-llm-profiles.md) | pass | — |
| [F10.custom-model](feature-map/F10-llm-profiles.md) | blocked | — |
| [F10.model-picker](feature-map/F10-llm-profiles.md) | pass | — |
| [F10.provider-list-complete](feature-map/F10-llm-profiles.md) | pass | — |
| [F10.cloud-provider-pagination](feature-map/F10-llm-profiles.md) | blocked | — |
| [F10.edit](feature-map/F10-llm-profiles.md) | blocked | — |
| [F10.basic-save-keeps-base-url](feature-map/F10-llm-profiles.md) | blocked | — |
| [F10.edit-default-reapply](feature-map/F10-llm-profiles.md) | pass | — |
| [F10.editor-discard](feature-map/F10-llm-profiles.md) | pass | — |
| [F10.schema-validation](feature-map/F10-llm-profiles.md) | pass | — |
| [F10.connection-link](feature-map/F10-llm-profiles.md) | pass | — |
| [F10.add-models](feature-map/F10-llm-profiles.md) | pass | — |
| [F10.subscription](feature-map/F10-llm-profiles.md) | blocked | — |
| [F10.schema-unavailable](feature-map/F10-llm-profiles.md) | blocked | — |
| [F10.cloud-readonly](feature-map/F10-llm-profiles.md) | blocked | — |
| [F10.phone](feature-map/F10-llm-profiles.md) | pass | — |
| [F10.default-used](feature-map/F10-llm-profiles.md) | blocked | — |

## F11

| Map ID | Result | Triage |
| --- | --- | --- |
| [F11.list-empty](feature-map/F11-provider-connections.md) | pass | — |
| [F11.list](feature-map/F11-provider-connections.md) | pass | — |
| [F11.create-validation](feature-map/F11-provider-connections.md) | pass | — |
| [F11.provider-picker](feature-map/F11-provider-connections.md) | pass | — |
| [F11.create](feature-map/F11-provider-connections.md) | pass | — |
| [F11.create-cancel](feature-map/F11-provider-connections.md) | pass | — |
| [F11.create-error](feature-map/F11-provider-connections.md) | pass | — |
| [F11.row-menu](feature-map/F11-provider-connections.md) | fail | [OpenHands/OpenHands#18204](https://github.com/OpenHands/OpenHands/issues/18204) |
| [F11.row-menu-focus](feature-map/F11-provider-connections.md) | fail | [OpenHands/OpenHands#18204](https://github.com/OpenHands/OpenHands/issues/18204) |
| [F11.edit](feature-map/F11-provider-connections.md) | pass | — |
| [F11.add-models](feature-map/F11-provider-connections.md) | pass | — |
| [F11.agent-uses-connection](feature-map/F11-provider-connections.md) | blocked | — |
| [F11.rotate](feature-map/F11-provider-connections.md) | blocked | — |
| [F11.delete](feature-map/F11-provider-connections.md) | pass | — |
| [F11.delete-referenced](feature-map/F11-provider-connections.md) | pass | — |
| [F11.delete-stale-reference](feature-map/F11-provider-connections.md) | fail | [OpenHands/software-agent-sdk#5498](https://github.com/OpenHands/software-agent-sdk/issues/5498) |
| [F11.phone](feature-map/F11-provider-connections.md) | pass | — |
| [F11.load-error](feature-map/F11-provider-connections.md) | pass | — |
| [F11.long-name](feature-map/F11-provider-connections.md) | pass | — |
| [F11.edit-unlisted-provider](feature-map/F11-provider-connections.md) | pass | — |
| [F11.cloud](feature-map/F11-provider-connections.md) | blocked | — |

## F12

| Map ID | Result | Triage |
| --- | --- | --- |
| [F12.page](feature-map/F12-model-router.md) | pass | — |
| [F12.empty-states](feature-map/F12-model-router.md) | pass | — |
| [F12.load-error](feature-map/F12-model-router.md) | blocked | — |
| [F12.unsupported](feature-map/F12-model-router.md) | blocked | — |
| [F12.cloud](feature-map/F12-model-router.md) | blocked | — |
| [F12.template-chooser](feature-map/F12-model-router.md) | pass | — |
| [F12.template-custom](feature-map/F12-model-router.md) | pass | — |
| [F12.editor-validation](feature-map/F12-model-router.md) | pass | — |
| [F12.duplicate-name](feature-map/F12-model-router.md) | pass | — |
| [F12.add-connection](feature-map/F12-model-router.md) | pass | — |
| [F12.add-connection-cancel](feature-map/F12-model-router.md) | pass | — |
| [F12.editor-cancel](feature-map/F12-model-router.md) | pass | — |
| [F12.router-profiles](feature-map/F12-model-router.md) | pass | — |
| [F12.create-first](feature-map/F12-model-router.md) | pass | — |
| [F12.create-more](feature-map/F12-model-router.md) | pass | — |
| [F12.list](feature-map/F12-model-router.md) | pass | — |
| [F12.row-menu](feature-map/F12-model-router.md) | pass | — |
| [F12.activate](feature-map/F12-model-router.md) | pass | — |
| [F12.edit](feature-map/F12-model-router.md) | pass | — |
| [F12.delete](feature-map/F12-model-router.md) | pass | — |
| [F12.delete-active](feature-map/F12-model-router.md) | blocked | — |
| [F12.run-first-message-toggle](feature-map/F12-model-router.md) | pass | — |
| [F12.route-first-message](feature-map/F12-model-router.md) | blocked | — |
| [F12.light-theme](feature-map/F12-model-router.md) | pass | — |
| [F12.phone](feature-map/F12-model-router.md) | pass | — |

## F13

| Map ID | Result | Triage |
| --- | --- | --- |
| [F13.list](feature-map/F13-agent-profiles.md) | pass | — |
| [F13.create](feature-map/F13-agent-profiles.md) | pass | — |
| [F13.name-validation](feature-map/F13-agent-profiles.md) | pass | — |
| [F13.editor-no-llm](feature-map/F13-agent-profiles.md) | pass | — |
| [F13.openhands-options](feature-map/F13-agent-profiles.md) | pass | — |
| [F13.edit](feature-map/F13-agent-profiles.md) | pass | — |
| [F13.stale-llm-ref](feature-map/F13-agent-profiles.md) | pass | — |
| [F13.set-active](feature-map/F13-agent-profiles.md) | pass | — |
| [F13.active-drives-conversation](feature-map/F13-agent-profiles.md) | blocked | — |
| [F13.llm-pill-precedence](feature-map/F13-agent-profiles.md) | pass | — |
| [F13.mcp-scope](feature-map/F13-agent-profiles.md) | pass | — |
| [F13.secrets-scope](feature-map/F13-agent-profiles.md) | pass | — |
| [F13.acp-form](feature-map/F13-agent-profiles.md) | pass | — |
| [F13.acp-credentials](feature-map/F13-agent-profiles.md) | pass | — |
| [F13.acp-preset-credentials](feature-map/F13-agent-profiles.md) | pass | — |
| [F13.menu-keyboard](feature-map/F13-agent-profiles.md) | fail | [OpenHands/OpenHands#18060](https://github.com/OpenHands/OpenHands/issues/18060) |
| [F13.delete](feature-map/F13-agent-profiles.md) | pass | — |
| [F13.phone](feature-map/F13-agent-profiles.md) | pass | — |
| [F13.acp-conversation](feature-map/F13-agent-profiles.md) | blocked | — |
| [F13.cloud-read-only](feature-map/F13-agent-profiles.md) | blocked | — |

## F14

| Map ID | Result | Triage |
| --- | --- | --- |
| [F14.list](feature-map/F14-secrets.md) | pass | — |
| [F14.create](feature-map/F14-secrets.md) | pass | — |
| [F14.create-validation](feature-map/F14-secrets.md) | pass | — |
| [F14.edit](feature-map/F14-secrets.md) | pass | — |
| [F14.delete](feature-map/F14-secrets.md) | pass | — |
| [F14.agent-access](feature-map/F14-secrets.md) | blocked | — |
| [F14.phone](feature-map/F14-secrets.md) | pass | — |
| [F14.form-back](feature-map/F14-secrets.md) | pass | — |
| [F14.rename](feature-map/F14-secrets.md) | pass | — |
| [F14.edit-value](feature-map/F14-secrets.md) | pass | — |
| [F14.delete-escape](feature-map/F14-secrets.md) | pass | — |

## F15

| Map ID | Result | Triage |
| --- | --- | --- |
| [F15.condenser-page](feature-map/F15-agent-behavior-settings.md) | pass | — |
| [F15.condenser-all-view](feature-map/F15-agent-behavior-settings.md) | pass | — |
| [F15.condenser-save](feature-map/F15-agent-behavior-settings.md) | pass | — |
| [F15.condenser-dependents](feature-map/F15-agent-behavior-settings.md) | pass | — |
| [F15.condenser-kind](feature-map/F15-agent-behavior-settings.md) | fail | [OpenHands/OpenHands#18049](https://github.com/OpenHands/OpenHands/issues/18049) |
| [F15.save-dirty-state](feature-map/F15-agent-behavior-settings.md) | pass | — |
| [F15.save-errors](feature-map/F15-agent-behavior-settings.md) | pass | — |
| [F15.view-scoped-save](feature-map/F15-agent-behavior-settings.md) | fail | [OpenHands/OpenHands#18049](https://github.com/OpenHands/OpenHands/issues/18049) |
| [F15.agent-context-page](feature-map/F15-agent-behavior-settings.md) | pass | — |
| [F15.agent-context-save](feature-map/F15-agent-behavior-settings.md) | pass | — |
| [F15.verification-page](feature-map/F15-agent-behavior-settings.md) | pass | — |
| [F15.verification-critic-fields](feature-map/F15-agent-behavior-settings.md) | pass | — |
| [F15.confirmation-mode-save](feature-map/F15-agent-behavior-settings.md) | pass | — |
| [F15.confirmation-mode-effect](feature-map/F15-agent-behavior-settings.md) | blocked | — |
| [F15.critic-tuning](feature-map/F15-agent-behavior-settings.md) | pass | — |
| [F15.critic-save](feature-map/F15-agent-behavior-settings.md) | pass | — |
| [F15.agent-context-effect](feature-map/F15-agent-behavior-settings.md) | pass | — |
| [F15.tier-after-save](feature-map/F15-agent-behavior-settings.md) | pass | — |
| [F15.unsaved-navigation](feature-map/F15-agent-behavior-settings.md) | pass | — |
| [F15.command-menu](feature-map/F15-agent-behavior-settings.md) | pass | — |
| [F15.direct-url](feature-map/F15-agent-behavior-settings.md) | pass | — |
| [F15.phone](feature-map/F15-agent-behavior-settings.md) | pass | — |
| [F15.schema-unavailable](feature-map/F15-agent-behavior-settings.md) | blocked | — |

## F16

| Map ID | Result | Triage |
| --- | --- | --- |
| [F16.open](feature-map/F16-application-settings.md) | pass | — |
| [F16.language](feature-map/F16-application-settings.md) | pass | — |
| [F16.color-theme](feature-map/F16-application-settings.md) | pass | — |
| [F16.analytics](feature-map/F16-application-settings.md) | pass | — |
| [F16.analytics-cloud](feature-map/F16-application-settings.md) | blocked | — |
| [F16.sound](feature-map/F16-application-settings.md) | blocked | — |
| [F16.checklist](feature-map/F16-application-settings.md) | pass | — |
| [F16.title-model](feature-map/F16-application-settings.md) | blocked | — |
| [F16.title-model-fallback](feature-map/F16-application-settings.md) | pass | — |
| [F16.manage-profiles-link](feature-map/F16-application-settings.md) | pass | — |
| [F16.voice-endpoint](feature-map/F16-application-settings.md) | pass | — |
| [F16.git-identity](feature-map/F16-application-settings.md) | blocked | — |
| [F16.git-identity-validation](feature-map/F16-application-settings.md) | pass | — |
| [F16.save-states](feature-map/F16-application-settings.md) | fail | [OpenHands/OpenHands#17927](https://github.com/OpenHands/OpenHands/issues/17927) |
| [F16.keyboard](feature-map/F16-application-settings.md) | fail | [OpenHands/OpenHands#17900](https://github.com/OpenHands/OpenHands/issues/17900) |
| [F16.phone](feature-map/F16-application-settings.md) | pass | — |
| [F16.title-model-clear](feature-map/F16-application-settings.md) | pass | — |
| [F16.dropdown-filter](feature-map/F16-application-settings.md) | pass | — |
| [F16.browser-scope](feature-map/F16-application-settings.md) | pass | — |

## F17

| Map ID | Result | Triage |
| --- | --- | --- |
| [F17.customize-redirect](feature-map/F17-mcp-servers.md) | pass | — |
| [F17.desktop-subnav](feature-map/F17-mcp-servers.md) | pass | — |
| [F17.mobile-hub](feature-map/F17-mcp-servers.md) | pass | — |
| [F17.breakpoint-sweep](feature-map/F17-mcp-servers.md) | pass | — |
| [F17.cloud-nav](feature-map/F17-mcp-servers.md) | blocked | — |
| [F17.page-states](feature-map/F17-mcp-servers.md) | pass | — |
| [F17.search](feature-map/F17-mcp-servers.md) | pass | — |
| [F17.section-filter](feature-map/F17-mcp-servers.md) | pass | — |
| [F17.library-catalog](feature-map/F17-mcp-servers.md) | pass | — |
| [F17.install-remote](feature-map/F17-mcp-servers.md) | pass | — |
| [F17.install-oauth](feature-map/F17-mcp-servers.md) | pass | — |
| [F17.install-unsupported](feature-map/F17-mcp-servers.md) | fail | [OpenHands/OpenHands#17922](https://github.com/OpenHands/OpenHands/issues/17922) |
| [F17.install-args](feature-map/F17-mcp-servers.md) | pass | — |
| [F17.install-stdio](feature-map/F17-mcp-servers.md) | pass | — |
| [F17.install-error](feature-map/F17-mcp-servers.md) | pass | — |
| [F17.save-as-secret](feature-map/F17-mcp-servers.md) | pass | — |
| [F17.custom-validation](feature-map/F17-mcp-servers.md) | fail | [OpenHands/OpenHands#17958](https://github.com/OpenHands/OpenHands/issues/17958) |
| [F17.test-connection](feature-map/F17-mcp-servers.md) | pass | — |
| [F17.custom-add](feature-map/F17-mcp-servers.md) | pass | — |
| [F17.custom-add-remote](feature-map/F17-mcp-servers.md) | blocked | — |
| [F17.server-health](feature-map/F17-mcp-servers.md) | pass | — |
| [F17.enable-disable](feature-map/F17-mcp-servers.md) | blocked | — |
| [F17.custom-edit](feature-map/F17-mcp-servers.md) | pass | — |
| [F17.stored-secret-reuse](feature-map/F17-mcp-servers.md) | pass | — |
| [F17.single-request-mutations](feature-map/F17-mcp-servers.md) | pass | — |
| [F17.credential-probe](feature-map/F17-mcp-servers.md) | fail | [OpenHands/software-agent-sdk#5607](https://github.com/OpenHands/software-agent-sdk/issues/5607) |
| [F17.delete-server](feature-map/F17-mcp-servers.md) | pass | — |
| [F17.native-git-tabs](feature-map/F17-mcp-servers.md) | blocked | — |
| [F17.agent-uses-server](feature-map/F17-mcp-servers.md) | blocked | — |
| [F17.phone](feature-map/F17-mcp-servers.md) | pass | — |

## F18

| Map ID | Result | Triage |
| --- | --- | --- |
| [F18.page](feature-map/F18-skills.md) | pass | — |
| [F18.search](feature-map/F18-skills.md) | pass | — |
| [F18.search-history](feature-map/F18-skills.md) | pass | — |
| [F18.facets](feature-map/F18-skills.md) | pass | — |
| [F18.filters-modal](feature-map/F18-skills.md) | pass | — |
| [F18.card](feature-map/F18-skills.md) | pass | — |
| [F18.copy-source](feature-map/F18-skills.md) | pass | — |
| [F18.toggle](feature-map/F18-skills.md) | pass | — |
| [F18.toggle-local](feature-map/F18-skills.md) | blocked | — |
| [F18.toggle-error](feature-map/F18-skills.md) | fail | [OpenHands/OpenHands#17941](https://github.com/OpenHands/OpenHands/issues/17941) |
| [F18.skill-in-chat](feature-map/F18-skills.md) | blocked | — |
| [F18.project-skill](feature-map/F18-skills.md) | blocked | — |
| [F18.skill-deleted](feature-map/F18-skills.md) | pass | — |
| [F18.detail-modal](feature-map/F18-skills.md) | pass | — |
| [F18.detail-close](feature-map/F18-skills.md) | pass | — |
| [F18.detail-pills](feature-map/F18-skills.md) | fail | [OpenHands/OpenHands#17957](https://github.com/OpenHands/OpenHands/issues/17957) |
| [F18.use-skill](feature-map/F18-skills.md) | pass | — |
| [F18.add-skill-modal](feature-map/F18-skills.md) | pass | — |
| [F18.install-banner](feature-map/F18-skills.md) | blocked | — |
| [F18.phone](feature-map/F18-skills.md) | pass | — |
| [F18.cloud-link](feature-map/F18-skills.md) | blocked | — |

## F19

| Map ID | Result | Triage |
| --- | --- | --- |
| [F19.page](feature-map/F19-plugins.md) | pass | — |
| [F19.search](feature-map/F19-plugins.md) | pass | — |
| [F19.status-filter](feature-map/F19-plugins.md) | pass | — |
| [F19.detail-modal](feature-map/F19-plugins.md) | pass | — |
| [F19.files-browser](feature-map/F19-plugins.md) | pass | — |
| [F19.install-card](feature-map/F19-plugins.md) | pass | — |
| [F19.install-modal](feature-map/F19-plugins.md) | pass | — |
| [F19.enable-toggle](feature-map/F19-plugins.md) | fail | [OpenHands/software-agent-sdk#5496](https://github.com/OpenHands/software-agent-sdk/issues/5496) |
| [F19.enabled-autoload](feature-map/F19-plugins.md) | blocked | — |
| [F19.refresh](feature-map/F19-plugins.md) | fail | [OpenHands/software-agent-sdk#5496](https://github.com/OpenHands/software-agent-sdk/issues/5496) |
| [F19.uninstall](feature-map/F19-plugins.md) | pass | — |
| [F19.add-plugin](feature-map/F19-plugins.md) | pass | — |
| [F19.add-plugin-cancel](feature-map/F19-plugins.md) | pass | — |
| [F19.add-plugin-error](feature-map/F19-plugins.md) | pass | — |
| [F19.installed-coordinates](feature-map/F19-plugins.md) | pass | — |
| [F19.local-plugin](feature-map/F19-plugins.md) | pass | — |
| [F19.start-conversation](feature-map/F19-plugins.md) | blocked | — |
| [F19.launch-review](feature-map/F19-plugins.md) | pass | — |
| [F19.launch-parameters](feature-map/F19-plugins.md) | pass | — |
| [F19.launch-trust](feature-map/F19-plugins.md) | blocked | — |
| [F19.launch-dev-params](feature-map/F19-plugins.md) | pass | — |
| [F19.launch-errors](feature-map/F19-plugins.md) | pass | — |
| [F19.launch-creation-failed](feature-map/F19-plugins.md) | pass | — |
| [F19.launch-unicode](feature-map/F19-plugins.md) | fail | [OpenHands/OpenHands#16735](https://github.com/OpenHands/OpenHands/issues/16735) |
| [F19.phone](feature-map/F19-plugins.md) | pass | — |

## F20

| Map ID | Result | Triage |
| --- | --- | --- |
| [F20.entry-points](feature-map/F20-canvas-apps.md) | pass | — |
| [F20.page](feature-map/F20-canvas-apps.md) | pass | — |
| [F20.add-modal](feature-map/F20-canvas-apps.md) | pass | — |
| [F20.install](feature-map/F20-canvas-apps.md) | pass | — |
| [F20.install-error](feature-map/F20-canvas-apps.md) | pass | — |
| [F20.git-tree-url](feature-map/F20-canvas-apps.md) | pass | — |
| [F20.git-ref-path](feature-map/F20-canvas-apps.md) | pass | — |
| [F20.install-duplicate](feature-map/F20-canvas-apps.md) | pass | — |
| [F20.card](feature-map/F20-canvas-apps.md) | pass | — |
| [F20.enable-confirm](feature-map/F20-canvas-apps.md) | fail | [OpenHands/OpenHands#18196](https://github.com/OpenHands/OpenHands/issues/18196) |
| [F20.rail-entry](feature-map/F20-canvas-apps.md) | pass | — |
| [F20.rail-active](feature-map/F20-canvas-apps.md) | pass | — |
| [F20.rail-label](feature-map/F20-canvas-apps.md) | fail | [OpenHands/software-agent-sdk#5501](https://github.com/OpenHands/software-agent-sdk/issues/5501) |
| [F20.page-render](feature-map/F20-canvas-apps.md) | pass | — |
| [F20.app-backend-view](feature-map/F20-canvas-apps.md) | pass | — |
| [F20.page-unavailable](feature-map/F20-canvas-apps.md) | pass | — |
| [F20.phone](feature-map/F20-canvas-apps.md) | pass | — |
| [F20.refresh](feature-map/F20-canvas-apps.md) | pass | — |
| [F20.persist-restart](feature-map/F20-canvas-apps.md) | pass | — |
| [F20.disable](feature-map/F20-canvas-apps.md) | pass | — |
| [F20.uninstall](feature-map/F20-canvas-apps.md) | pass | — |
| [F20.busy-lock](feature-map/F20-canvas-apps.md) | pass | — |
| [F20.load-error](feature-map/F20-canvas-apps.md) | pass | — |
| [F20.unsupported](feature-map/F20-canvas-apps.md) | pass | — |

## F21

| Map ID | Result | Triage |
| --- | --- | --- |
| [F21.sidebar-entry](feature-map/F21-automations-dashboard.md) | pass | — |
| [F21.command-menu](feature-map/F21-automations-dashboard.md) | pass | — |
| [F21.pin-home](feature-map/F21-automations-dashboard.md) | pass | — |
| [F21.subpage-nav](feature-map/F21-automations-dashboard.md) | pass | — |
| [F21.manifest-404](feature-map/F21-automations-dashboard.md) | fail | [OpenHands/OpenHands#18210](https://github.com/OpenHands/OpenHands/issues/18210) |
| [F21.health-loading](feature-map/F21-automations-dashboard.md) | blocked | — |
| [F21.list-loading](feature-map/F21-automations-dashboard.md) | pass | — |
| [F21.backend-unavailable](feature-map/F21-automations-dashboard.md) | pass | — |
| [F21.list-error](feature-map/F21-automations-dashboard.md) | pass | — |
| [F21.add-menu](feature-map/F21-automations-dashboard.md) | pass | — |
| [F21.empty-state](feature-map/F21-automations-dashboard.md) | pass | — |
| [F21.empty-rail](feature-map/F21-automations-dashboard.md) | pass | — |
| [F21.overview-tiles](feature-map/F21-automations-dashboard.md) | pass | — |
| [F21.search](feature-map/F21-automations-dashboard.md) | pass | — |
| [F21.filters](feature-map/F21-automations-dashboard.md) | pass | — |
| [F21.sort](feature-map/F21-automations-dashboard.md) | pass | — |
| [F21.filtered-empty](feature-map/F21-automations-dashboard.md) | pass | — |
| [F21.view-toggle](feature-map/F21-automations-dashboard.md) | pass | — |
| [F21.card-grid](feature-map/F21-automations-dashboard.md) | pass | — |
| [F21.list-row](feature-map/F21-automations-dashboard.md) | pass | — |
| [F21.row-tooltip](feature-map/F21-automations-dashboard.md) | pass | — |
| [F21.summary-hovercard](feature-map/F21-automations-dashboard.md) | blocked | — |
| [F21.sparkline-deeplink](feature-map/F21-automations-dashboard.md) | blocked | — |
| [F21.open-detail](feature-map/F21-automations-dashboard.md) | pass | — |
| [F21.load-more](feature-map/F21-automations-dashboard.md) | pass | — |
| [F21.git-sync-button](feature-map/F21-automations-dashboard.md) | pass | — |
| [F21.kebab-menu](feature-map/F21-automations-dashboard.md) | pass | — |
| [F21.run-now](feature-map/F21-automations-dashboard.md) | blocked | — |
| [F21.run-now-error](feature-map/F21-automations-dashboard.md) | fail | [OpenHands/OpenHands#18062](https://github.com/OpenHands/OpenHands/issues/18062) |
| [F21.toggle](feature-map/F21-automations-dashboard.md) | pass | — |
| [F21.edit](feature-map/F21-automations-dashboard.md) | pass | — |
| [F21.export](feature-map/F21-automations-dashboard.md) | pass | — |
| [F21.delete](feature-map/F21-automations-dashboard.md) | pass | — |
| [F21.phone](feature-map/F21-automations-dashboard.md) | pass | — |

## F22

| Map ID | Result | Triage |
| --- | --- | --- |
| [F22.templates-page](feature-map/F22-automation-creation.md) | pass | — |
| [F22.template-card](feature-map/F22-automation-creation.md) | pass | — |
| [F22.templates-search](feature-map/F22-automation-creation.md) | fail | [OpenHands/OpenHands#17959](https://github.com/OpenHands/OpenHands/issues/17959) |
| [F22.template-launch-setup](feature-map/F22-automation-creation.md) | pass | — |
| [F22.template-launch-conversation](feature-map/F22-automation-creation.md) | pass | — |
| [F22.template-mcp-install-queue](feature-map/F22-automation-creation.md) | blocked | — |
| [F22.responder-deployment-choice](feature-map/F22-automation-creation.md) | pass | — |
| [F22.setup-prerequisites](feature-map/F22-automation-creation.md) | pass | — |
| [F22.setup-action-kind](feature-map/F22-automation-creation.md) | pass | — |
| [F22.setup-form-validation](feature-map/F22-automation-creation.md) | pass | — |
| [F22.setup-agent-profile](feature-map/F22-automation-creation.md) | pass | — |
| [F22.setup-plugin-create](feature-map/F22-automation-creation.md) | pass | — |
| [F22.setup-tarball-create](feature-map/F22-automation-creation.md) | pass | — |
| [F22.setup-repository-list](feature-map/F22-automation-creation.md) | pass | — |
| [F22.setup-review-confirm](feature-map/F22-automation-creation.md) | pass | — |
| [F22.setup-bundle-create](feature-map/F22-automation-creation.md) | pass | — |
| [F22.setup-unsupported](feature-map/F22-automation-creation.md) | pass | — |
| [F22.setup-fallback-conversation](feature-map/F22-automation-creation.md) | pass | — |
| [F22.setup-close](feature-map/F22-automation-creation.md) | pass | — |
| [F22.setup-unknown-id](feature-map/F22-automation-creation.md) | pass | — |
| [F22.create-helper-conversation](feature-map/F22-automation-creation.md) | pass | — |
| [F22.import-picker](feature-map/F22-automation-creation.md) | pass | — |
| [F22.import-validation](feature-map/F22-automation-creation.md) | pass | — |
| [F22.import-preview](feature-map/F22-automation-creation.md) | pass | — |
| [F22.import-confirm](feature-map/F22-automation-creation.md) | pass | — |
| [F22.import-choose-file](feature-map/F22-automation-creation.md) | pass | — |
| [F22.import-dropzone](feature-map/F22-automation-creation.md) | pass | — |
| [F22.phone](feature-map/F22-automation-creation.md) | pass | — |

## F23

| Map ID | Result | Triage |
| --- | --- | --- |
| [F23.open-detail](feature-map/F23-automation-detail.md) | pass | — |
| [F23.back-link](feature-map/F23-automation-detail.md) | pass | — |
| [F23.loading](feature-map/F23-automation-detail.md) | pass | — |
| [F23.not-found](feature-map/F23-automation-detail.md) | fail | [OpenHands/OpenHands#17948](https://github.com/OpenHands/OpenHands/issues/17948) |
| [F23.backend-unavailable](feature-map/F23-automation-detail.md) | pass | — |
| [F23.load-error](feature-map/F23-automation-detail.md) | pass | — |
| [F23.header](feature-map/F23-automation-detail.md) | pass | — |
| [F23.toggle](feature-map/F23-automation-detail.md) | pass | — |
| [F23.disabled-reason](feature-map/F23-automation-detail.md) | pass | — |
| [F23.run-now](feature-map/F23-automation-detail.md) | blocked | — |
| [F23.run-now-error](feature-map/F23-automation-detail.md) | fail | [OpenHands/OpenHands#18062](https://github.com/OpenHands/OpenHands/issues/18062) |
| [F23.run-status-polling](feature-map/F23-automation-detail.md) | blocked | — |
| [F23.activity-log](feature-map/F23-automation-detail.md) | pass | — |
| [F23.run-task-outcome](feature-map/F23-automation-detail.md) | blocked | — |
| [F23.activity-log-more](feature-map/F23-automation-detail.md) | blocked | — |
| [F23.run-no-conversation](feature-map/F23-automation-detail.md) | blocked | — |
| [F23.run-linked-conversation](feature-map/F23-automation-detail.md) | blocked | — |
| [F23.run-logs-modal](feature-map/F23-automation-detail.md) | blocked | — |
| [F23.debug-with-openhands](feature-map/F23-automation-detail.md) | blocked | — |
| [F23.activity-log-export](feature-map/F23-automation-detail.md) | blocked | — |
| [F23.run-deeplink](feature-map/F23-automation-detail.md) | blocked | — |
| [F23.prompt-collapse](feature-map/F23-automation-detail.md) | pass | — |
| [F23.configuration](feature-map/F23-automation-detail.md) | pass | — |
| [F23.configuration-event](feature-map/F23-automation-detail.md) | pass | — |
| [F23.plugins-repo](feature-map/F23-automation-detail.md) | pass | — |
| [F23.script-section](feature-map/F23-automation-detail.md) | pass | — |
| [F23.export](feature-map/F23-automation-detail.md) | pass | — |
| [F23.tarball](feature-map/F23-automation-detail.md) | fail | [OpenHands/OpenHands#17948](https://github.com/OpenHands/OpenHands/issues/17948) |
| [F23.delete](feature-map/F23-automation-detail.md) | pass | — |
| [F23.edit-open](feature-map/F23-automation-detail.md) | pass | — |
| [F23.edit-close](feature-map/F23-automation-detail.md) | fail | [OpenHands/OpenHands#17563](https://github.com/OpenHands/OpenHands/issues/17563) |
| [F23.edit-validation](feature-map/F23-automation-detail.md) | pass | — |
| [F23.edit-save](feature-map/F23-automation-detail.md) | pass | — |
| [F23.edit-schedule](feature-map/F23-automation-detail.md) | pass | — |
| [F23.edit-time-cleared](feature-map/F23-automation-detail.md) | fail | [OpenHands/OpenHands#17948](https://github.com/OpenHands/OpenHands/issues/17948) |
| [F23.edit-cron](feature-map/F23-automation-detail.md) | pass | — |
| [F23.edit-event-readonly](feature-map/F23-automation-detail.md) | pass | — |
| [F23.edit-save-error](feature-map/F23-automation-detail.md) | pass | — |
| [F23.phone](feature-map/F23-automation-detail.md) | fail | [OpenHands/OpenHands#17563](https://github.com/OpenHands/OpenHands/issues/17563) |

## F24

| Map ID | Result | Triage |
| --- | --- | --- |
| [F24.open](feature-map/F24-git-sync.md) | pass | — |
| [F24.loading](feature-map/F24-git-sync.md) | pass | — |
| [F24.overview-unconfigured](feature-map/F24-git-sync.md) | pass | — |
| [F24.form-dirty](feature-map/F24-git-sync.md) | pass | — |
| [F24.check-failure](feature-map/F24-git-sync.md) | pass | — |
| [F24.save-and-sync](feature-map/F24-git-sync.md) | pass | — |
| [F24.sync-now](feature-map/F24-git-sync.md) | pass | — |
| [F24.sync-failure](feature-map/F24-git-sync.md) | pass | — |
| [F24.encryption](feature-map/F24-git-sync.md) | pass | — |
| [F24.token](feature-map/F24-git-sync.md) | pass | — |
| [F24.author](feature-map/F24-git-sync.md) | pass | — |
| [F24.interval](feature-map/F24-git-sync.md) | pass | — |
| [F24.pause](feature-map/F24-git-sync.md) | pass | — |
| [F24.sync-disabled-error](feature-map/F24-git-sync.md) | pass | — |
| [F24.background-sync](feature-map/F24-git-sync.md) | pass | — |
| [F24.repo-link](feature-map/F24-git-sync.md) | pass | — |
| [F24.repo-link-ssh](feature-map/F24-git-sync.md) | pass | — |
| [F24.sync-delete](feature-map/F24-git-sync.md) | pass | — |
| [F24.field-defaults](feature-map/F24-git-sync.md) | pass | — |
| [F24.clear-repo](feature-map/F24-git-sync.md) | pass | — |
| [F24.backend-down](feature-map/F24-git-sync.md) | pass | — |
| [F24.phone](feature-map/F24-git-sync.md) | pass | — |
| [F24.unsupported](feature-map/F24-git-sync.md) | blocked | — |
| [F24.no-access](feature-map/F24-git-sync.md) | blocked | — |
| [F24.conflict](feature-map/F24-git-sync.md) | blocked | — |
| [F24.error-state](feature-map/F24-git-sync.md) | pass | — |

## F25

| Map ID | Result | Triage |
| --- | --- | --- |
| [F25.selector-dropdown](feature-map/F25-backends-and-cloud.md) | pass | — |
| [F25.add-backend-modal](feature-map/F25-backends-and-cloud.md) | pass | — |
| [F25.cloud-advanced-host](feature-map/F25-backends-and-cloud.md) | pass | — |
| [F25.cloud-device-flow](feature-map/F25-backends-and-cloud.md) | fail | [OpenHands/OpenHands#17953](https://github.com/OpenHands/OpenHands/issues/17953) |
| [F25.agent-server-guidance](feature-map/F25-backends-and-cloud.md) | pass | — |
| [F25.add-agent-server](feature-map/F25-backends-and-cloud.md) | pass | — |
| [F25.switch-backend](feature-map/F25-backends-and-cloud.md) | pass | — |
| [F25.backend-pinned-url](feature-map/F25-backends-and-cloud.md) | pass | — |
| [F25.per-tab-backend](feature-map/F25-backends-and-cloud.md) | pass | — |
| [F25.manage-backends](feature-map/F25-backends-and-cloud.md) | pass | — |
| [F25.manage-add](feature-map/F25-backends-and-cloud.md) | pass | — |
| [F25.manage-select](feature-map/F25-backends-and-cloud.md) | pass | — |
| [F25.edit-backend](feature-map/F25-backends-and-cloud.md) | pass | — |
| [F25.edit-cancel](feature-map/F25-backends-and-cloud.md) | pass | — |
| [F25.edit-escape](feature-map/F25-backends-and-cloud.md) | fail | [OpenHands/OpenHands#17953](https://github.com/OpenHands/OpenHands/issues/17953) |
| [F25.health-status](feature-map/F25-backends-and-cloud.md) | pass | — |
| [F25.recovery-gate](feature-map/F25-backends-and-cloud.md) | pass | — |
| [F25.remove-backend](feature-map/F25-backends-and-cloud.md) | pass | — |
| [F25.collapsed-entry](feature-map/F25-backends-and-cloud.md) | pass | — |
| [F25.phone](feature-map/F25-backends-and-cloud.md) | pass | — |
| [F25.device-verify-page](feature-map/F25-backends-and-cloud.md) | pass | — |
| [F25.shared-conversation-view](feature-map/F25-backends-and-cloud.md) | fail | [OpenHands/OpenHands#17953](https://github.com/OpenHands/OpenHands/issues/17953) |
| [F25.share-publicly](feature-map/F25-backends-and-cloud.md) | blocked | — |
| [F25.cloud-org-rows](feature-map/F25-backends-and-cloud.md) | blocked | — |
| [F25.cloud-log-back-in](feature-map/F25-backends-and-cloud.md) | blocked | — |
| [F25.cloud-settings-link](feature-map/F25-backends-and-cloud.md) | blocked | — |
| [F25.cloud-sandbox-states](feature-map/F25-backends-and-cloud.md) | blocked | — |
| [F25.locked-cloud](feature-map/F25-backends-and-cloud.md) | blocked | — |

## F26

| Map ID | Result | Triage |
| --- | --- | --- |
| [F26.cli-version](feature-map/F26-runtime-variants.md) | pass | — |
| [F26.cli-info](feature-map/F26-runtime-variants.md) | pass | — |
| [F26.cli-help](feature-map/F26-runtime-variants.md) | pass | — |
| [F26.cli-flag-conflicts](feature-map/F26-runtime-variants.md) | pass | — |
| [F26.cli-public-needs-key](feature-map/F26-runtime-variants.md) | pass | — |
| [F26.cli-missing-build](feature-map/F26-runtime-variants.md) | pass | — |
| [F26.cli-port-in-use](feature-map/F26-runtime-variants.md) | pass | — |
| [F26.loopback-bind-default](feature-map/F26-runtime-variants.md) | pass | — |
| [F26.session-key-rotated](feature-map/F26-runtime-variants.md) | pass | — |
| [F26.session-key-persist](feature-map/F26-runtime-variants.md) | pass | — |
| [F26.session-key-pinned](feature-map/F26-runtime-variants.md) | pass | — |
| [F26.frontend-only](feature-map/F26-runtime-variants.md) | pass | — |
| [F26.frontend-only-returning](feature-map/F26-runtime-variants.md) | fail | [OpenHands/OpenHands#18160](https://github.com/OpenHands/OpenHands/issues/18160) |
| [F26.backend-only](feature-map/F26-runtime-variants.md) | pass | — |
| [F26.cross-connect](feature-map/F26-runtime-variants.md) | pass | — |
| [F26.remote-backend](feature-map/F26-runtime-variants.md) | pass | — |
| [F26.seeded-local-backend](feature-map/F26-runtime-variants.md) | pass | — |
| [F26.host-bind-lan](feature-map/F26-runtime-variants.md) | blocked | — |
| [F26.host-bind-lan-optin](feature-map/F26-runtime-variants.md) | blocked | — |
| [F26.host-bind-env](feature-map/F26-runtime-variants.md) | blocked | — |
| [F26.runtime-services](feature-map/F26-runtime-variants.md) | pass | — |
| [F26.lib-build](feature-map/F26-runtime-variants.md) | pass | — |
| [F26.lib-style-scope](feature-map/F26-runtime-variants.md) | pass | — |
| [F26.lib-host-app](feature-map/F26-runtime-variants.md) | blocked | — |
| [F26.docker-image](feature-map/F26-runtime-variants.md) | blocked | — |
| [F26.docker-conversation-runtime](feature-map/F26-runtime-variants.md) | blocked | — |
| [F26.helm-chart](feature-map/F26-runtime-variants.md) | blocked | — |
| [F26.desktop-boot-splash](feature-map/F26-runtime-variants.md) | blocked | — |
| [F26.desktop-main-window](feature-map/F26-runtime-variants.md) | blocked | — |
| [F26.desktop-external-links](feature-map/F26-runtime-variants.md) | blocked | — |

## F27

| Map ID | Result | Triage |
| --- | --- | --- |
| [F27.terminal-empty](feature-map/F27-workspace-tools.md) | blocked | — |
| [F27.terminal-waiting](feature-map/F27-workspace-tools.md) | pass | — |
| [F27.terminal-output](feature-map/F27-workspace-tools.md) | blocked | — |
| [F27.terminal-history-reload](feature-map/F27-workspace-tools.md) | blocked | — |
| [F27.terminal-live-append](feature-map/F27-workspace-tools.md) | blocked | — |
| [F27.browser-empty](feature-map/F27-workspace-tools.md) | pass | — |
| [F27.browser-screenshot](feature-map/F27-workspace-tools.md) | blocked | — |
| [F27.planner-empty](feature-map/F27-workspace-tools.md) | pass | — |
| [F27.planner-create](feature-map/F27-workspace-tools.md) | pass | — |
| [F27.planner-plan](feature-map/F27-workspace-tools.md) | blocked | — |
| [F27.planner-build](feature-map/F27-workspace-tools.md) | blocked | — |
| [F27.tasklist](feature-map/F27-workspace-tools.md) | blocked | — |
| [F27.usage-metrics](feature-map/F27-workspace-tools.md) | blocked | — |
| [F27.usage-empty](feature-map/F27-workspace-tools.md) | blocked | — |
| [F27.usage-compact](feature-map/F27-workspace-tools.md) | blocked | — |
| [F27.usage-provider-balance](feature-map/F27-workspace-tools.md) | pass | — |
| [F27.composer-context-meter](feature-map/F27-workspace-tools.md) | blocked | — |
| [F27.overview-peek](feature-map/F27-workspace-tools.md) | pass | — |
| [F27.overview-pin](feature-map/F27-workspace-tools.md) | pass | — |
| [F27.overview-changes-link](feature-map/F27-workspace-tools.md) | pass | — |
| [F27.overview-menu-open](feature-map/F27-workspace-tools.md) | pass | — |
| [F27.overview-identity](feature-map/F27-workspace-tools.md) | blocked | — |
| [F27.overview-drawer](feature-map/F27-workspace-tools.md) | pass | — |
| [F27.canvas-ui-open-tab](feature-map/F27-workspace-tools.md) | blocked | — |
| [F27.canvas-ui-navigate-file](feature-map/F27-workspace-tools.md) | blocked | — |
| [F27.canvas-ui-phone](feature-map/F27-workspace-tools.md) | blocked | — |
| [F27.launch-child](feature-map/F27-workspace-tools.md) | blocked | — |
| [F27.info-modals-phone](feature-map/F27-workspace-tools.md) | pass | — |
| [F27.plugins-modal](feature-map/F27-workspace-tools.md) | blocked | — |
| [F27.phone](feature-map/F27-workspace-tools.md) | pass | — |
