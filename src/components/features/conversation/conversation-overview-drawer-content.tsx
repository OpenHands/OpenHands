import React, { useEffect } from "react";
import { useTranslation } from "react-i18next";
import XMarkIcon from "#/icons/x-mark.svg?react";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { useConversationOverviewDrawer } from "./conversation-overview-drawer-context";
import {
  CONVERSATION_OVERVIEW_DRAWER_SECTION,
  type ConversationOverviewDrawerSection,
} from "./conversation-overview-drawer.types";
import { ConversationOverviewAutomationsPanel } from "./conversation-overview-automations-panel";
import { ConversationOverviewSkillsPanel } from "./conversation-overview-skills-panel";
import { ConversationOverviewMcpPanel } from "./conversation-overview-mcp-panel";
import { ConversationOverviewSecretsPanel } from "./conversation-overview-secrets-panel";
import {
  ConversationOverviewGitItemsHeaderLink,
  ConversationOverviewGitItemsPanel,
} from "./conversation-overview-git-items-panel";
import { useConversationOverviewStats } from "#/hooks/use-conversation-overview-stats";
import {
  CONVERSATION_SECONDARY_DRAWER_CLOSE_BUTTON_CLASSNAME,
  CONVERSATION_SECONDARY_DRAWER_HEADER_ACTION_CLASSNAME,
  CONVERSATION_SECONDARY_DRAWER_HEADER_CLASSNAME,
} from "./conversation-secondary-drawer.classes";

interface ConversationOverviewDrawerContentProps {
  className?: string;
}

function getSectionTitleKey(section: ConversationOverviewDrawerSection) {
  switch (section) {
    case CONVERSATION_OVERVIEW_DRAWER_SECTION.automations:
      return I18nKey.CONVERSATION_PANEL$AUTOMATIONS;
    case CONVERSATION_OVERVIEW_DRAWER_SECTION.skills:
      return I18nKey.SETTINGS$NAV_SKILLS;
    case CONVERSATION_OVERVIEW_DRAWER_SECTION.mcp:
      return I18nKey.CONVERSATION$OVERVIEW_MCP;
    case CONVERSATION_OVERVIEW_DRAWER_SECTION.secrets:
      return I18nKey.SETTINGS$NAV_SECRETS;
    case CONVERSATION_OVERVIEW_DRAWER_SECTION.pull_requests:
      return I18nKey.CONVERSATION$OVERVIEW_PULL_REQUESTS;
    case CONVERSATION_OVERVIEW_DRAWER_SECTION.issues:
      return I18nKey.CONVERSATION$OVERVIEW_ISSUES;
    default: {
      const _exhaustive: never = section;
      return _exhaustive;
    }
  }
}

function ConversationOverviewDrawerHeaderAddAction({
  section,
}: {
  section: ConversationOverviewDrawerSection;
}) {
  const { t } = useTranslation("openhands");
  const { requestAdd } = useConversationOverviewDrawer();

  switch (section) {
    case CONVERSATION_OVERVIEW_DRAWER_SECTION.automations:
      return (
        <BrandButton
          type="button"
          variant="primary"
          testId="conversation-overview-automations-add"
          className={CONVERSATION_SECONDARY_DRAWER_HEADER_ACTION_CLASSNAME}
          onClick={requestAdd}
        >
          {t(I18nKey.AUTOMATIONS$ADD_AUTOMATION)}
        </BrandButton>
      );
    case CONVERSATION_OVERVIEW_DRAWER_SECTION.skills:
      return (
        <BrandButton
          type="button"
          variant="primary"
          testId="conversation-overview-skills-add-skill-button"
          className={CONVERSATION_SECONDARY_DRAWER_HEADER_ACTION_CLASSNAME}
          onClick={requestAdd}
        >
          {t(I18nKey.SETTINGS$SKILLS_ADD_BUTTON)}
        </BrandButton>
      );
    case CONVERSATION_OVERVIEW_DRAWER_SECTION.mcp:
      return (
        <BrandButton
          type="button"
          variant="secondary"
          testId="conversation-overview-mcp-add-server"
          className={CONVERSATION_SECONDARY_DRAWER_HEADER_ACTION_CLASSNAME}
          onClick={requestAdd}
        >
          {t(I18nKey.MCP$ADD_CUSTOM)}
        </BrandButton>
      );
    case CONVERSATION_OVERVIEW_DRAWER_SECTION.secrets:
      return (
        <BrandButton
          type="button"
          variant="primary"
          testId="conversation-overview-secrets-add-button"
          className={CONVERSATION_SECONDARY_DRAWER_HEADER_ACTION_CLASSNAME}
          onClick={requestAdd}
        >
          {t(I18nKey.SECRETS$ADD_NEW_SECRET)}
        </BrandButton>
      );
    case CONVERSATION_OVERVIEW_DRAWER_SECTION.pull_requests:
      return <ConversationOverviewGitItemsHeaderLink kind="pull_requests" />;
    case CONVERSATION_OVERVIEW_DRAWER_SECTION.issues:
      return <ConversationOverviewGitItemsHeaderLink kind="issues" />;
    default:
      return null;
  }
}

function ConversationOverviewDrawerBody({
  section,
  openAdd,
}: {
  section: ConversationOverviewDrawerSection;
  openAdd: boolean;
}) {
  switch (section) {
    case CONVERSATION_OVERVIEW_DRAWER_SECTION.automations:
      return <ConversationOverviewAutomationsPanel openAdd={openAdd} />;
    case CONVERSATION_OVERVIEW_DRAWER_SECTION.skills:
      return <ConversationOverviewSkillsPanel openAdd={openAdd} />;
    case CONVERSATION_OVERVIEW_DRAWER_SECTION.mcp:
      return <ConversationOverviewMcpPanel openAdd={openAdd} />;
    case CONVERSATION_OVERVIEW_DRAWER_SECTION.secrets:
      return <ConversationOverviewSecretsPanel openAdd={openAdd} />;
    case CONVERSATION_OVERVIEW_DRAWER_SECTION.pull_requests:
      return <ConversationOverviewGitItemsPanel kind="pull_requests" />;
    case CONVERSATION_OVERVIEW_DRAWER_SECTION.issues:
      return <ConversationOverviewGitItemsPanel kind="issues" />;
    default: {
      const _exhaustive: never = section;
      return _exhaustive;
    }
  }
}

export function ConversationOverviewDrawerContent({
  className,
}: ConversationOverviewDrawerContentProps) {
  const { t } = useTranslation("openhands");
  const { section, openAdd, closeDrawer } = useConversationOverviewDrawer();
  const { workspaceName } = useConversationOverviewStats();

  useEffect(() => {
    if (!section) {
      return undefined;
    }

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        closeDrawer();
      }
    };

    document.addEventListener("keydown", handleEscape);
    return () => {
      document.removeEventListener("keydown", handleEscape);
    };
  }, [closeDrawer, section]);

  if (!section) {
    return null;
  }

  const isFullBleedBody =
    section === CONVERSATION_OVERVIEW_DRAWER_SECTION.pull_requests ||
    section === CONVERSATION_OVERVIEW_DRAWER_SECTION.issues ||
    section === CONVERSATION_OVERVIEW_DRAWER_SECTION.secrets;

  return (
    <aside
      data-testid="conversation-overview-drawer-content"
      aria-label={t(getSectionTitleKey(section))}
      className={cn(
        "flex h-full w-full min-h-0 flex-col overflow-hidden bg-base-secondary",
        className,
      )}
    >
      <header className={CONVERSATION_SECONDARY_DRAWER_HEADER_CLASSNAME}>
        <button
          type="button"
          data-testid="conversation-overview-drawer-close"
          aria-label={t(I18nKey.BUTTON$CLOSE)}
          onClick={closeDrawer}
          className={CONVERSATION_SECONDARY_DRAWER_CLOSE_BUTTON_CLASSNAME}
        >
          <XMarkIcon className="size-4" aria-hidden />
        </button>
        <div className="flex min-w-0 flex-1 items-center gap-2">
          <h2 className="truncate text-sm font-medium text-content">
            {t(getSectionTitleKey(section))}
          </h2>
          {workspaceName ? (
            <p
              data-testid="conversation-overview-drawer-workspace"
              className="truncate text-xs text-muted"
            >
              {workspaceName}
            </p>
          ) : null}
        </div>
        <div className="shrink-0">
          <ConversationOverviewDrawerHeaderAddAction section={section} />
        </div>
      </header>

      <div
        className={cn(
          "min-h-0 flex-1 custom-scrollbar-always",
          isFullBleedBody
            ? "flex flex-col overflow-hidden"
            : "overflow-y-auto px-4 py-4",
        )}
      >
        <ConversationOverviewDrawerBody section={section} openAdd={openAdd} />
      </div>
    </aside>
  );
}
