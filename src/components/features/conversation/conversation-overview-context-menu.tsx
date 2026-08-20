import React, { useLayoutEffect, useState } from "react";
import ReactDOM from "react-dom";
import {
  FolderGit2,
  GitBranch,
  GitCommitHorizontal,
  Laptop,
} from "lucide-react";
import { LuFileDiff } from "react-icons/lu";
import { useTranslation } from "react-i18next";
import { ContextMenu } from "#/ui/context-menu";
import { Divider } from "#/ui/divider";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useConversationLocalStorageState } from "#/utils/conversation-local-storage";
import { useSelectConversationTab } from "#/hooks/use-select-conversation-tab";
import { I18nKey } from "#/i18n/declaration";
import PrIcon from "#/icons/u-pr.svg?react";
import PillIcon from "#/icons/pill.svg?react";
import PillFillIcon from "#/icons/pill-fill.svg?react";
import { cn } from "#/utils/utils";
import {
  dropdownInstantColorClassName,
  dropdownMenuRowIconWrapperClassName,
} from "#/utils/dropdown-classes";
import { useConversationOverviewDrawerOptional } from "./conversation-overview-drawer-context";
import {
  CONVERSATION_OVERVIEW_DRAWER_SECTION,
  type ConversationOverviewDrawerSection,
} from "./conversation-overview-drawer.types";
import {
  CONVERSATION_OVERVIEW_GIT_PART,
  CONVERSATION_OVERVIEW_SECTION,
  CONVERSATION_OVERVIEW_SECTION_GROUPS,
  isOverviewGitPartPinned,
  isOverviewSectionPinned,
  type ConversationOverviewGitPart,
  type ConversationOverviewSection,
} from "./conversation-overview-sections";

interface ConversationOverviewContextMenuProps {
  isOpen: boolean;
  onClose: () => void;
  ignoreOutsideClickRef?: React.RefObject<HTMLElement | null>;
  /** Portal anchor so the menu is not clipped by the overview panel. */
  anchorRef?: React.RefObject<HTMLElement | null>;
}

type OverviewMenuItem = {
  section: ConversationOverviewSection;
  icon: React.ComponentType<{ className?: string }>;
  i18nKey: I18nKey;
};

type GitPartMenuItem = {
  part: ConversationOverviewGitPart;
  icon: React.ComponentType<{ className?: string }>;
  i18nKey: I18nKey;
  drawerSection?: ConversationOverviewDrawerSection;
  opensChanges?: boolean;
  opensCommits?: boolean;
};

const OVERVIEW_MENU_ITEMS_BY_SECTION: Record<
  Exclude<ConversationOverviewSection, "git">,
  OverviewMenuItem
> = {
  [CONVERSATION_OVERVIEW_SECTION.workspace]: {
    section: CONVERSATION_OVERVIEW_SECTION.workspace,
    icon: Laptop,
    i18nKey: I18nKey.CONVERSATION$OVERVIEW_WORKSPACE,
  },
};

const GIT_PART_MENU_ITEMS: GitPartMenuItem[] = [
  {
    part: CONVERSATION_OVERVIEW_GIT_PART.changes,
    icon: LuFileDiff,
    i18nKey: I18nKey.COMMON$CHANGES,
    opensChanges: true,
  },
  {
    part: CONVERSATION_OVERVIEW_GIT_PART.repository,
    icon: FolderGit2,
    i18nKey: I18nKey.CONVERSATION$REPOSITORY,
  },
  {
    part: CONVERSATION_OVERVIEW_GIT_PART.branch,
    icon: GitBranch,
    i18nKey: I18nKey.CONVERSATION$BRANCH,
  },
  {
    part: CONVERSATION_OVERVIEW_GIT_PART.commits,
    icon: GitCommitHorizontal,
    i18nKey: I18nKey.DIFF_VIEWER$COMMITS,
    opensCommits: true,
  },
  {
    part: CONVERSATION_OVERVIEW_GIT_PART.pull_requests,
    icon: PrIcon,
    i18nKey: I18nKey.CONVERSATION$OVERVIEW_PULL_REQUESTS,
    drawerSection: CONVERSATION_OVERVIEW_DRAWER_SECTION.pull_requests,
  },
];

function OverviewPinButton({
  testId,
  pinned,
  pinnedLabel,
  unpinnedLabel,
  onClick,
}: {
  testId: string;
  pinned: boolean;
  pinnedLabel: string;
  unpinnedLabel: string;
  onClick: (event: React.MouseEvent) => void;
}) {
  return (
    <button
      type="button"
      data-testid={testId}
      className={cn(
        "flex shrink-0 cursor-pointer items-center justify-center rounded-r px-2 text-white",
        dropdownInstantColorClassName,
        "hover:bg-white/10",
      )}
      aria-pressed={pinned}
      aria-label={pinned ? pinnedLabel : unpinnedLabel}
      onClick={onClick}
    >
      <span
        className={cn(
          "ml-auto overflow-hidden",
          dropdownMenuRowIconWrapperClassName,
        )}
        aria-hidden
      >
        {pinned ? (
          <PillFillIcon className="size-4" width={16} height={16} />
        ) : (
          <PillIcon className="size-4" width={16} height={16} />
        )}
      </span>
    </button>
  );
}

export function ConversationOverviewContextMenu({
  isOpen,
  onClose,
  ignoreOutsideClickRef,
  anchorRef,
}: ConversationOverviewContextMenuProps) {
  const ref = useClickOutsideElement<HTMLUListElement>(
    onClose,
    ignoreOutsideClickRef,
  );
  const [portalStyle, setPortalStyle] = useState<React.CSSProperties>();
  const { t } = useTranslation("openhands");
  const { conversationId } = useConversationId();
  const { state, setUnpinnedOverviewSections, setUnpinnedOverviewGitParts } =
    useConversationLocalStorageState(conversationId);
  const { navigateToChanges, navigateToCommits } = useSelectConversationTab();
  const overviewDrawer = useConversationOverviewDrawerOptional();

  useLayoutEffect(() => {
    if (!isOpen || !anchorRef?.current) {
      setPortalStyle(undefined);
      return undefined;
    }

    const updatePosition = () => {
      const rect = anchorRef.current?.getBoundingClientRect();
      if (!rect) return;

      const gap = 8;
      setPortalStyle({
        position: "fixed",
        zIndex: 9999,
        top: rect.bottom + gap,
        right: window.innerWidth - rect.right,
      });
    };

    updatePosition();
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);
    return () => {
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
    };
  }, [isOpen, anchorRef]);

  const handleOpenGitPart = (item: GitPartMenuItem) => {
    if (item.opensChanges) {
      overviewDrawer?.closeDrawer();
      navigateToChanges();
      onClose();
      return;
    }

    if (item.opensCommits) {
      overviewDrawer?.closeDrawer();
      navigateToCommits();
      onClose();
      return;
    }

    if (!item.drawerSection) return;
    overviewDrawer?.openSection(item.drawerSection);
    onClose();
  };

  const handlePinToggle = (
    section: ConversationOverviewSection,
    event: React.MouseEvent,
  ) => {
    event.preventDefault();
    event.stopPropagation();

    const unpinnedSections = state.unpinnedOverviewSections ?? [];
    if (unpinnedSections.includes(section)) {
      setUnpinnedOverviewSections?.(
        unpinnedSections.filter((item) => item !== section),
      );
      return;
    }

    setUnpinnedOverviewSections?.([...unpinnedSections, section]);
  };

  const handleGitPartPinToggle = (
    part: ConversationOverviewGitPart,
    event: React.MouseEvent,
  ) => {
    event.preventDefault();
    event.stopPropagation();

    const unpinnedGitParts = state.unpinnedOverviewGitParts ?? [];
    if (unpinnedGitParts.includes(part)) {
      setUnpinnedOverviewGitParts?.(
        unpinnedGitParts.filter((item) => item !== part),
      );
      return;
    }

    setUnpinnedOverviewGitParts?.([...unpinnedGitParts, part]);
  };

  if (!isOpen) return null;

  const isPortaled = Boolean(anchorRef?.current && portalStyle);
  const gitPinned = isOverviewSectionPinned(
    CONVERSATION_OVERVIEW_SECTION.git,
    state.unpinnedOverviewSections ?? [],
  );
  const pinnedLabel = t(I18nKey.CONVERSATION$UNPIN_OVERVIEW_SECTION);
  const unpinnedLabel = t(I18nKey.CONVERSATION$PIN_OVERVIEW_SECTION);

  const renderMenuRow = ({
    key,
    testIdOpen,
    testIdPin,
    icon: Icon,
    label,
    pinned,
    isActionable,
    indent = false,
    onOpen,
    onPinToggle,
  }: {
    key: string;
    testIdOpen: string;
    testIdPin: string;
    icon: React.ComponentType<{ className?: string }>;
    label: string;
    pinned: boolean;
    isActionable: boolean;
    indent?: boolean;
    onOpen?: () => void;
    onPinToggle: (event: React.MouseEvent) => void;
  }) => {
    const labelContent = (
      <>
        <span className={dropdownMenuRowIconWrapperClassName} aria-hidden>
          <Icon className="h-4 w-4" />
        </span>
        <span className="text-sm">{label}</span>
      </>
    );

    return (
      <li key={key} className="list-none">
        <div
          className={cn(
            "group flex h-[30px] w-full min-w-0 items-stretch rounded hover:bg-[var(--oh-interactive-hover)]",
            indent && "pl-4",
          )}
        >
          {isActionable ? (
            <button
              type="button"
              data-testid={testIdOpen}
              className={cn(
                "flex min-w-0 flex-1 cursor-pointer items-center gap-2 rounded-l p-2 text-start text-white",
                dropdownInstantColorClassName,
              )}
              onClick={onOpen}
            >
              {labelContent}
            </button>
          ) : (
            <div
              data-testid={testIdOpen}
              className={cn(
                "flex min-w-0 flex-1 items-center gap-2 rounded-l p-2 text-start text-white",
                dropdownInstantColorClassName,
              )}
            >
              {labelContent}
            </div>
          )}
          <OverviewPinButton
            testId={testIdPin}
            pinned={pinned}
            pinnedLabel={pinnedLabel}
            unpinnedLabel={unpinnedLabel}
            onClick={onPinToggle}
          />
        </div>
      </li>
    );
  };

  const renderSectionRow = (section: ConversationOverviewSection) => {
    if (section === CONVERSATION_OVERVIEW_SECTION.git) return null;
    const item = OVERVIEW_MENU_ITEMS_BY_SECTION[section];
    return renderMenuRow({
      key: section,
      testIdOpen: `conversation-overview-menu-open-${section}`,
      testIdPin: `conversation-overview-menu-pin-${section}`,
      icon: item.icon,
      label: t(item.i18nKey),
      pinned: isOverviewSectionPinned(
        section,
        state.unpinnedOverviewSections ?? [],
      ),
      isActionable: false,
      onPinToggle: (event) => handlePinToggle(section, event),
    });
  };

  const menu = (
    <ContextMenu
      ref={ref}
      testId="conversation-overview-context-menu"
      theme={isPortaled ? "popover" : "default"}
      position={isPortaled ? "none" : "bottom"}
      alignment={isPortaled ? "none" : "right"}
      spacing={isPortaled ? "none" : "default"}
      className={cn("z-[9999] w-fit", isPortaled ? "mt-0" : "mt-2")}
    >
      {CONVERSATION_OVERVIEW_SECTION_GROUPS.map((group, groupIndex) => (
        <React.Fragment key={group.sections.join("-")}>
          {groupIndex > 0 ? (
            <Divider
              testId={`conversation-overview-menu-divider-${groupIndex}`}
              inset="menu"
            />
          ) : null}
          {group.sections.map((section) => renderSectionRow(section))}
        </React.Fragment>
      ))}

      <Divider testId="conversation-overview-menu-divider-git" inset="menu" />

      {renderMenuRow({
        key: CONVERSATION_OVERVIEW_SECTION.git,
        testIdOpen: `conversation-overview-menu-open-${CONVERSATION_OVERVIEW_SECTION.git}`,
        testIdPin: `conversation-overview-menu-pin-${CONVERSATION_OVERVIEW_SECTION.git}`,
        icon: GitBranch,
        label: t(I18nKey.CONVERSATION$OVERVIEW_GIT),
        pinned: gitPinned,
        isActionable: false,
        onPinToggle: (event) =>
          handlePinToggle(CONVERSATION_OVERVIEW_SECTION.git, event),
      })}

      {GIT_PART_MENU_ITEMS.map((item) =>
        renderMenuRow({
          key: item.part,
          testIdOpen: `conversation-overview-menu-open-git-${item.part}`,
          testIdPin: `conversation-overview-menu-pin-git-${item.part}`,
          icon: item.icon,
          label: t(item.i18nKey),
          pinned: isOverviewGitPartPinned(
            item.part,
            state.unpinnedOverviewGitParts ?? [],
          ),
          isActionable: Boolean(
            item.drawerSection || item.opensCommits || item.opensChanges,
          ),
          indent: true,
          onOpen: () => handleOpenGitPart(item),
          onPinToggle: (event) => handleGitPartPinToggle(item.part, event),
        }),
      )}
    </ContextMenu>
  );

  if (isPortaled && portalStyle && typeof document !== "undefined") {
    return ReactDOM.createPortal(
      <div style={portalStyle}>{menu}</div>,
      document.body,
    );
  }

  return menu;
}
