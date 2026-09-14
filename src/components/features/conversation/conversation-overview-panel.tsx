import React, { useRef, useState } from "react";
import { Laptop } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useConversationOverviewStats } from "#/hooks/use-conversation-overview-stats";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useConversationPrimaryRepository } from "#/hooks/use-conversation-primary-repository";
import { useConversationLocalStorageState } from "#/utils/conversation-local-storage";
import { ConversationOverviewDiffsRow } from "./conversation-overview-diffs-row";
import { ConversationOverviewContextMenu } from "./conversation-overview-context-menu";
import { ConversationOverviewGitSection } from "./conversation-overview-git-section";
import { EllipsisButton } from "#/components/features/conversation-panel/ellipsis-button";
import { Divider } from "#/ui/divider";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import {
  CONVERSATION_OVERVIEW_GIT_PART,
  CONVERSATION_OVERVIEW_SECTION,
  CONVERSATION_OVERVIEW_SECTION_GROUPS,
  isOverviewGitPartPinned,
  isOverviewSectionPinned,
  type ConversationOverviewSection,
} from "./conversation-overview-sections";

const PANEL_CLASSNAME = cn(
  "w-full max-w-[240px] rounded-xl border border-[var(--oh-border)]",
  "bg-[var(--oh-surface)] pb-1",
);

const ROW_CLASSNAME = cn(
  "flex items-center gap-2 rounded-md px-2 py-1.5",
  "transition-colors hover:bg-white/5",
);

const ROW_ICON_CLASSNAME = "size-4 shrink-0 text-[var(--oh-muted)]";

interface OverviewRowProps {
  icon: React.ReactNode;
  label: string;
  value: React.ReactNode;
  testId: string;
}

function OverviewRow({ icon, label, value, testId }: OverviewRowProps) {
  return (
    <li data-testid={testId} className={ROW_CLASSNAME}>
      {icon}
      <span className="min-w-0 flex-1 truncate text-sm text-[var(--oh-foreground)]">
        {label}
      </span>
      <span className="max-w-[45%] shrink-0 truncate text-right text-sm text-[var(--oh-muted)]">
        {value}
      </span>
    </li>
  );
}

function OverviewSectionDivider() {
  // Match dropdown/context-menu separators: `px-1` host + `inset="menu"`
  // Divider (12px slot, line bleeds into the horizontal padding).
  return (
    <div className="px-1">
      <Divider inset="menu" />
    </div>
  );
}

type OverviewPanelBlock =
  | {
      kind: "sections";
      key: string;
      sections: ConversationOverviewSection[];
    }
  | { kind: "git"; key: "git" };

export function ConversationOverviewPanel() {
  const { t } = useTranslation("openhands");
  const stats = useConversationOverviewStats();
  const { conversationId } = useConversationId();
  const { state } = useConversationLocalStorageState(conversationId);
  const { isConnected: isGitConnected } = useConversationPrimaryRepository();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const menuAnchorRef = useRef<HTMLButtonElement>(null);

  const isPinned = (section: ConversationOverviewSection) =>
    isOverviewSectionPinned(section, state.unpinnedOverviewSections ?? []);

  const showChanges = isOverviewGitPartPinned(
    CONVERSATION_OVERVIEW_GIT_PART.changes,
    state.unpinnedOverviewGitParts ?? [],
  );

  const showGitBlock =
    isPinned(CONVERSATION_OVERVIEW_SECTION.git) &&
    (showChanges || isGitConnected);

  const workspaceSections = CONVERSATION_OVERVIEW_SECTION_GROUPS.flatMap(
    (group) => group.sections.filter((section) => isPinned(section)),
  );

  // Git content first (untitled); Workspace sits below it.
  const panelBlocks: OverviewPanelBlock[] = [];
  if (showGitBlock) {
    panelBlocks.push({ kind: "git", key: "git" });
  }
  if (workspaceSections.length > 0) {
    panelBlocks.push({
      kind: "sections",
      key: workspaceSections.join("-"),
      sections: workspaceSections,
    });
  }

  const renderSection = (section: ConversationOverviewSection) => {
    switch (section) {
      case CONVERSATION_OVERVIEW_SECTION.workspace:
        return (
          <OverviewRow
            key={section}
            testId="conversation-overview-workspace"
            icon={<Laptop className={ROW_ICON_CLASSNAME} />}
            label={t(I18nKey.CONVERSATION$OVERVIEW_WORKSPACE)}
            value={stats.workspaceName ?? t(I18nKey.CONVERSATION$OVERVIEW_NONE)}
          />
        );
      case CONVERSATION_OVERVIEW_SECTION.git:
        // Rendered as its own panel block (not a group row).
        return null;
      default: {
        const _exhaustive: never = section;
        return _exhaustive;
      }
    }
  };

  return (
    <aside
      data-testid="conversation-overview-panel"
      aria-label={t(I18nKey.CONVERSATION$OVERVIEW)}
      className={PANEL_CLASSNAME}
    >
      <div className="flex items-center justify-between px-4 pb-0.5 pt-2.5">
        <span className="text-xs font-medium text-[var(--oh-muted)]">
          {t(I18nKey.CONVERSATION$OVERVIEW)}
        </span>
        <div className="relative shrink-0">
          <EllipsisButton
            ref={menuAnchorRef}
            testId="conversation-overview-ellipsis"
            onClick={() => setIsMenuOpen((open) => !open)}
            ariaLabel={t(I18nKey.COMMON$MORE_OPTIONS)}
          />
          <ConversationOverviewContextMenu
            isOpen={isMenuOpen}
            onClose={() => setIsMenuOpen(false)}
            ignoreOutsideClickRef={menuAnchorRef}
            anchorRef={menuAnchorRef}
          />
        </div>
      </div>

      {panelBlocks.map((block, blockIndex) => (
        <React.Fragment key={block.key}>
          {blockIndex > 0 ? <OverviewSectionDivider /> : null}
          {block.kind === "git" ? (
            <div data-testid="conversation-overview-git-block">
              {showChanges ? (
                <ul className="px-2">
                  <ConversationOverviewDiffsRow />
                </ul>
              ) : null}
              {isGitConnected ? <ConversationOverviewGitSection /> : null}
            </div>
          ) : (
            <ul className="px-2">
              {block.sections.map((section) => renderSection(section))}
            </ul>
          )}
        </React.Fragment>
      ))}
    </aside>
  );
}
