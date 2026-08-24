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
  Workflow,
} from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import type { BackendKind } from "#/api/backend-registry/types";
import { useConversationPanelPreferencesStore } from "#/stores/conversation-panel-preferences-store";
import {
  BaseModalDescription,
  BaseModalTitle,
} from "#/components/shared/modals/confirmation-modals/base-modal";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { BrandButton } from "#/components/features/settings/brand-button";
import { MenuHeading } from "./menu-heading";
import { MenuSeparator } from "./menu-separator";
import { MenuRow } from "./menu-row";
import { UNNAMED_AUTOMATION_FACET } from "./conversation-panel-list-helpers";

export interface AdvancedConversationOptionsModalProps {
  open: boolean;
  onClose: () => void;
  backendKind: BackendKind;
  /**
   * Distinct automation names among the loaded conversations. Rendered as
   * selectable rows under the Automations section in `only-automations`
   * mode — the only control for `selectedAutomationNames`, which is
   * persisted and narrows the list on its own.
   */
  automationNameFacets: readonly string[];
}

/**
 * The full preference surface, promoted from the old hamburger filter menu
 * into its own modal (design from the 2026-08-14 PM walkthrough video).
 * Rows apply immediately and stay open — the modal closes only via Close,
 * Escape, or backdrop click.
 */
export function AdvancedConversationOptionsModal({
  open,
  onClose,
  backendKind,
  automationNameFacets,
}: AdvancedConversationOptionsModalProps) {
  const { t } = useTranslation("openhands");
  const preferences = useConversationPanelPreferencesStore();

  if (!open) return null;

  const groupedLabel =
    backendKind === "local"
      ? t(I18nKey.CONVERSATION_PANEL$BY_WORKSPACE)
      : t(I18nKey.CONVERSATION_PANEL$BY_REPOSITORY);

  return (
    <ModalBackdrop onClose={onClose}>
      <ModalBody
        className="items-start border border-[var(--oh-border)]"
        testID="advanced-conversation-options-modal"
      >
        <div className="flex w-full flex-col gap-2">
          <BaseModalTitle
            title={t(I18nKey.CONVERSATION_PANEL$ADVANCED_OPTIONS)}
          />
          <BaseModalDescription>
            {t(I18nKey.CONVERSATION_PANEL$ADVANCED_OPTIONS_DESCRIPTION)}
          </BaseModalDescription>
        </div>

        <div
          role="menu"
          aria-orientation="vertical"
          aria-label={t(I18nKey.CONVERSATION_PANEL$ADVANCED_OPTIONS)}
          // Same idiom as the old filter menu: focusable for jsx-a11y but
          // out of the natural Tab order (the rows are tabbable buttons).
          tabIndex={-1}
          className="custom-scrollbar-always flex max-h-[60vh] w-full flex-col overflow-y-auto px-1"
          onClick={(event) => event.stopPropagation()}
        >
          <MenuHeading>{t(I18nKey.CONVERSATION_PANEL$ORGANIZE)}</MenuHeading>
          <MenuRow
            icon={Folder}
            label={groupedLabel}
            selected={preferences.organizeMode === "grouped"}
            testId="organize-grouped"
            onClick={() => preferences.setOrganizeMode("grouped")}
          />
          <MenuRow
            icon={Clock3}
            label={t(I18nKey.CONVERSATION_PANEL$CHRONOLOGICAL)}
            selected={preferences.organizeMode === "chronological"}
            testId="organize-chronological"
            onClick={() => preferences.setOrganizeMode("chronological")}
          />

          <MenuSeparator />
          <MenuHeading>{t(I18nKey.CONVERSATION_PANEL$SORT_BY)}</MenuHeading>
          <MenuRow
            icon={CalendarArrowDown}
            label={t(I18nKey.CONVERSATION_PANEL$SORT_CREATED)}
            selected={preferences.conversationSort === "created"}
            testId="sort-created"
            onClick={() => preferences.setConversationSort("created")}
          />
          <MenuRow
            icon={ClockArrowDown}
            label={t(I18nKey.CONVERSATION_PANEL$SORT_UPDATED)}
            selected={preferences.conversationSort === "updated"}
            testId="sort-updated"
            onClick={() => preferences.setConversationSort("updated")}
          />

          <MenuSeparator />
          <MenuHeading>{t(I18nKey.CONVERSATION_PANEL$SHOW)}</MenuHeading>
          <MenuRow
            icon={MessageCircle}
            label={t(I18nKey.CONVERSATION_PANEL$ALL_THREADS)}
            selected={preferences.threadScope === "all"}
            testId="scope-all"
            onClick={() => preferences.setThreadScope("all")}
          />
          <MenuRow
            icon={Star}
            label={t(I18nKey.CONVERSATION_PANEL$RELEVANT_THREADS)}
            selected={preferences.threadScope === "relevant"}
            testId="scope-relevant"
            onClick={() => preferences.setThreadScope("relevant")}
          />
          <MenuRow
            icon={Archive}
            label={t(I18nKey.CONVERSATION_PANEL$SHOW_ARCHIVED)}
            selected={preferences.showArchivedConversations}
            variant="toggle"
            testId="toggle-show-archived"
            onClick={preferences.toggleShowArchivedConversations}
          />
          <MenuRow
            icon={preferences.showOlderConversations ? Eye : EyeOff}
            label={t(I18nKey.CONVERSATION_PANEL$SHOW_OLDER_CONVERSATIONS)}
            sublabel={t(I18nKey.CONVERSATION_PANEL$OLDER_OVER_ONE_HOUR)}
            selected={preferences.showOlderConversations}
            variant="toggle"
            testId="toggle-older-conversations"
            onClick={preferences.toggleShowOlderConversations}
          />
          <MenuRow
            icon={Tag}
            label={t(I18nKey.CONVERSATION_PANEL$TAG_FILTERS)}
            selected={preferences.tagFiltersEnabled}
            variant="toggle"
            testId="toggle-tag-filters"
            onClick={preferences.toggleTagFiltersEnabled}
          />

          <MenuSeparator />
          <MenuHeading>{t(I18nKey.CONVERSATION_PANEL$AUTOMATIONS)}</MenuHeading>
          <MenuRow
            icon={MessageCircle}
            label={t(I18nKey.CONVERSATION_PANEL$AUTOMATIONS_ALL)}
            selected={preferences.automationFilterMode === "all"}
            testId="automation-filter-all"
            onClick={() => preferences.setAutomationFilterMode("all")}
          />
          <MenuRow
            icon={EyeOff}
            label={t(I18nKey.CONVERSATION_PANEL$AUTOMATIONS_HIDE)}
            selected={preferences.automationFilterMode === "hide-automations"}
            testId="automation-filter-hide"
            onClick={() =>
              preferences.setAutomationFilterMode("hide-automations")
            }
          />
          <MenuRow
            icon={Workflow}
            label={t(I18nKey.CONVERSATION_PANEL$AUTOMATIONS_ONLY)}
            selected={preferences.automationFilterMode === "only-automations"}
            testId="automation-filter-only"
            onClick={() =>
              preferences.setAutomationFilterMode("only-automations")
            }
          />
          {/* Name rows only make sense while the list is scoped to
              automations — and `setAutomationFilterMode` clears the selection
              on the way out, so a hidden row can never keep narrowing. */}
          {preferences.automationFilterMode === "only-automations"
            ? automationNameFacets.map((facet) => (
                <MenuRow
                  key={facet}
                  icon={Tag}
                  label={
                    facet === UNNAMED_AUTOMATION_FACET
                      ? t(I18nKey.CONVERSATION_PANEL$AUTOMATION_UNNAMED)
                      : facet
                  }
                  selected={preferences.selectedAutomationNames.includes(facet)}
                  variant="toggle"
                  testId={`automation-name-row-${facet}`}
                  onClick={() => preferences.toggleAutomationName(facet)}
                />
              ))
            : null}

          <MenuSeparator />
          <MenuHeading>{t(I18nKey.CONVERSATION_PANEL$METADATA)}</MenuHeading>
          <MenuRow
            icon={GitBranch}
            label={t(I18nKey.CONVERSATION_PANEL$REPO_BRANCH)}
            selected={preferences.showRepoBranchMetadata}
            variant="toggle"
            testId="toggle-repo-branch-metadata"
            onClick={preferences.toggleShowRepoBranchMetadata}
          />
          <MenuRow
            icon={Bot}
            label={t(I18nKey.CONVERSATION_PANEL$LLM_MODEL)}
            selected={preferences.showLlmProfiles}
            variant="toggle"
            testId="toggle-llm-profiles"
            onClick={preferences.toggleShowLlmProfiles}
          />
          <MenuRow
            icon={Tag}
            label={t(I18nKey.CONVERSATION_PANEL$TAG_CHIPS)}
            selected={preferences.showTagsMetadata}
            variant="toggle"
            testId="toggle-tags-metadata"
            onClick={preferences.toggleShowTagsMetadata}
          />
          <MenuRow
            icon={MousePointerClick}
            label={t(I18nKey.CONVERSATION_PANEL$HOVER_METADATA)}
            selected={preferences.showHoverMetadata}
            variant="toggle"
            testId="toggle-hover-metadata"
            onClick={preferences.toggleShowHoverMetadata}
          />
        </div>

        <div
          className="flex w-full justify-end"
          onClick={(event) => event.stopPropagation()}
        >
          <BrandButton
            type="button"
            variant="primary"
            onClick={onClose}
            testId="advanced-options-close"
          >
            {t(I18nKey.BUTTON$CLOSE)}
          </BrandButton>
        </div>
      </ModalBody>
    </ModalBackdrop>
  );
}
