import React from "react";
import { useTranslation } from "react-i18next";
import {
  Archive,
  Bot,
  CalendarArrowDown,
  Clock3,
  ClockArrowDown,
  Eye,
  EyeOff,
  Folder,
  GitBranch,
  MessageCircle,
  MousePointerClick,
  Star,
  Tag,
  Trash2,
  Workflow,
} from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import type { BackendKind } from "#/api/backend-registry/types";
import {
  UNNAMED_AUTOMATION_FACET,
  type AutomationFilterMode,
  type ConversationSortField,
  type OrganizeMode,
  type ThreadScope,
} from "./conversation-panel-list-helpers";
import { MenuHeading } from "./menu-heading";
import { MenuSeparator } from "./menu-separator";
import { MenuRow } from "./menu-row";

const capitalizeLabel = (label: string) =>
  label.length > 0 ? label.charAt(0).toUpperCase() + label.slice(1) : label;

export interface ConversationPanelDisplaySettingsProps {
  backendKind: BackendKind;
  organizeMode: OrganizeMode;
  setOrganizeMode: (mode: OrganizeMode) => void;
  conversationSort: ConversationSortField;
  setConversationSort: (sort: ConversationSortField) => void;
  threadScope: ThreadScope;
  setThreadScope: (scope: ThreadScope) => void;
  automationFilterMode: AutomationFilterMode;
  setAutomationFilterMode: (mode: AutomationFilterMode) => void;
  selectedAutomationNames: string[];
  onToggleAutomationName: (name: string) => void;
  automationNameFacets: string[];
  showOlderConversations: boolean;
  showArchivedConversations: boolean;
  toggleShowArchivedConversations: () => void;
  toggleShowOlderConversations: () => void;
  showRepoBranchMetadata: boolean;
  toggleShowRepoBranchMetadata: () => void;
  showLlmProfiles: boolean;
  toggleShowLlmProfiles: () => void;
  showTagsMetadata: boolean;
  toggleShowTagsMetadata: () => void;
  showHoverMetadata: boolean;
  toggleShowHoverMetadata: () => void;
  totalConversationsCount: number;
  onRequestDeleteAll: () => void;
}

interface ConversationPanelAdvancedOptionsProps extends ConversationPanelDisplaySettingsProps {
  onClose: () => void;
}

export function ConversationPanelAdvancedOptions({
  backendKind,
  organizeMode,
  setOrganizeMode,
  conversationSort,
  setConversationSort,
  threadScope,
  setThreadScope,
  automationFilterMode,
  setAutomationFilterMode,
  selectedAutomationNames,
  onToggleAutomationName,
  automationNameFacets,
  showOlderConversations,
  showArchivedConversations,
  toggleShowArchivedConversations,
  toggleShowOlderConversations,
  showRepoBranchMetadata,
  toggleShowRepoBranchMetadata,
  showLlmProfiles,
  toggleShowLlmProfiles,
  showTagsMetadata,
  toggleShowTagsMetadata,
  showHoverMetadata,
  toggleShowHoverMetadata,
  totalConversationsCount,
  onRequestDeleteAll,
  onClose,
}: ConversationPanelAdvancedOptionsProps) {
  const { t } = useTranslation("openhands");

  const groupedLabel =
    backendKind === "local"
      ? t(I18nKey.CONVERSATION_PANEL$BY_WORKSPACE)
      : t(I18nKey.CONVERSATION_PANEL$BY_REPOSITORY);

  const applyAndClose = (action: () => void) => {
    action();
    onClose();
  };

  const handleDeleteAll = () => {
    if (totalConversationsCount === 0) return;
    onClose();
    onRequestDeleteAll();
  };

  return (
    <>
      <MenuHeading>{t(I18nKey.CONVERSATION_PANEL$ORGANIZE)}</MenuHeading>
      <MenuRow
        icon={Folder}
        label={groupedLabel}
        selected={organizeMode === "grouped"}
        onClick={() => applyAndClose(() => setOrganizeMode("grouped"))}
      />
      <MenuRow
        icon={Clock3}
        label={t(I18nKey.CONVERSATION_PANEL$CHRONOLOGICAL)}
        selected={organizeMode === "chronological"}
        onClick={() => applyAndClose(() => setOrganizeMode("chronological"))}
      />

      <MenuSeparator />
      <MenuHeading>{t(I18nKey.CONVERSATION_PANEL$SORT_BY)}</MenuHeading>
      <MenuRow
        icon={CalendarArrowDown}
        label={t(I18nKey.CONVERSATION_PANEL$SORT_CREATED)}
        selected={conversationSort === "created"}
        onClick={() => applyAndClose(() => setConversationSort("created"))}
      />
      <MenuRow
        icon={ClockArrowDown}
        label={t(I18nKey.CONVERSATION_PANEL$SORT_UPDATED)}
        selected={conversationSort === "updated"}
        onClick={() => applyAndClose(() => setConversationSort("updated"))}
      />

      <MenuSeparator />
      <MenuHeading>{t(I18nKey.CONVERSATION_PANEL$SHOW)}</MenuHeading>
      <MenuRow
        icon={MessageCircle}
        label={t(I18nKey.CONVERSATION_PANEL$ALL_THREADS)}
        selected={threadScope === "all"}
        onClick={() => applyAndClose(() => setThreadScope("all"))}
      />
      <MenuRow
        icon={Star}
        label={t(I18nKey.CONVERSATION_PANEL$RELEVANT_THREADS)}
        selected={threadScope === "relevant"}
        onClick={() => applyAndClose(() => setThreadScope("relevant"))}
      />
      <MenuRow
        icon={Archive}
        label={t(I18nKey.CONVERSATION_PANEL$SHOW_ARCHIVED)}
        selected={showArchivedConversations}
        testId="toggle-show-archived"
        onClick={() => applyAndClose(toggleShowArchivedConversations)}
      />

      <MenuSeparator />
      <MenuHeading>{t(I18nKey.CONVERSATION_PANEL$AUTOMATIONS)}</MenuHeading>
      <MenuRow
        icon={MessageCircle}
        label={t(I18nKey.CONVERSATION_PANEL$AUTOMATIONS_ALL)}
        selected={automationFilterMode === "all"}
        testId="automation-filter-all"
        onClick={() => applyAndClose(() => setAutomationFilterMode("all"))}
      />
      <MenuRow
        icon={EyeOff}
        label={t(I18nKey.CONVERSATION_PANEL$AUTOMATIONS_HIDE)}
        selected={automationFilterMode === "hide-automations"}
        testId="automation-filter-hide"
        onClick={() =>
          applyAndClose(() => setAutomationFilterMode("hide-automations"))
        }
      />
      <MenuRow
        icon={Workflow}
        label={t(I18nKey.CONVERSATION_PANEL$AUTOMATIONS_ONLY)}
        selected={automationFilterMode === "only-automations"}
        testId="automation-filter-only"
        onClick={() =>
          applyAndClose(() => setAutomationFilterMode("only-automations"))
        }
      />
      {automationFilterMode === "only-automations"
        ? automationNameFacets.map((facet) => (
            <MenuRow
              key={facet}
              icon={Tag}
              label={
                facet === UNNAMED_AUTOMATION_FACET
                  ? t(I18nKey.CONVERSATION_PANEL$AUTOMATION_UNNAMED)
                  : facet
              }
              selected={selectedAutomationNames.includes(facet)}
              testId={`automation-name-filter-${facet}`}
              onClick={() => applyAndClose(() => onToggleAutomationName(facet))}
            />
          ))
        : null}

      <MenuSeparator />
      <MenuHeading>{t(I18nKey.CONVERSATION_PANEL$METADATA)}</MenuHeading>
      <MenuRow
        icon={GitBranch}
        label={t(I18nKey.CONVERSATION_PANEL$REPO_BRANCH)}
        selected={showRepoBranchMetadata}
        testId="toggle-repo-branch-metadata"
        onClick={() => applyAndClose(toggleShowRepoBranchMetadata)}
      />
      <MenuRow
        icon={Bot}
        label={t(I18nKey.CONVERSATION_PANEL$LLM_MODEL)}
        selected={showLlmProfiles}
        testId="toggle-llm-profiles"
        onClick={() => applyAndClose(toggleShowLlmProfiles)}
      />
      <MenuRow
        icon={Tag}
        label={t(I18nKey.CONVERSATION_PANEL$TAGS)}
        selected={showTagsMetadata}
        testId="toggle-tags-metadata"
        onClick={() => applyAndClose(toggleShowTagsMetadata)}
      />
      <MenuRow
        icon={MousePointerClick}
        label={t(I18nKey.CONVERSATION_PANEL$HOVER_METADATA)}
        selected={showHoverMetadata}
        testId="toggle-hover-metadata"
        onClick={() => applyAndClose(toggleShowHoverMetadata)}
      />

      <MenuSeparator />
      <MenuHeading
        suffix={
          <span className="shrink-0 text-right text-[10px] font-medium normal-case tracking-normal text-[var(--oh-muted)]/70">
            {t(I18nKey.CONVERSATION_PANEL$OLDER_OVER_ONE_HOUR)}
          </span>
        }
      >
        {t(I18nKey.CONVERSATION_PANEL$OLDER_SECTION)}
      </MenuHeading>
      <MenuRow
        testId="toggle-older-conversations"
        icon={showOlderConversations ? EyeOff : Eye}
        label={
          showOlderConversations
            ? capitalizeLabel(t(I18nKey.CONVERSATION$HIDE))
            : capitalizeLabel(t(I18nKey.CONVERSATION$SHOW_ALL))
        }
        onClick={() => applyAndClose(toggleShowOlderConversations)}
      />

      <MenuSeparator />
      <MenuRow
        testId="delete-all-conversations"
        icon={Trash2}
        label={capitalizeLabel(t(I18nKey.CONVERSATION$DELETE_ALL))}
        disabled={totalConversationsCount === 0}
        onClick={handleDeleteAll}
      />
    </>
  );
}
