import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";
import { useTranslation } from "react-i18next";
import { LayoutGroup } from "framer-motion";
import { Gauge, Globe, ListTodo, SquareChevronRight } from "lucide-react";
import { LuFileDiff } from "react-icons/lu";
import DocumentIcon from "#/icons/document.svg?react";
import DoubleCheckIcon from "#/icons/double-check.svg?react";
import { EllipsisButton } from "#/components/features/conversation-panel/ellipsis-button";
import { cn } from "#/utils/utils";
import { useConversationLocalStorageState } from "#/utils/conversation-local-storage";
import { CONVERSATION_TAB_LABEL_KEYS } from "./conversation-tab-ids";
import { ConversationTabNav } from "./conversation-tab-nav";
import { ChatActionTooltip } from "../../chat/chat-action-tooltip";
import { I18nKey } from "#/i18n/declaration";
import { useConversationStore } from "#/stores/conversation-store";
import { ConversationTabsContextMenu } from "./conversation-tabs-context-menu";
import { ConversationPlannerBuildBar } from "./conversation-planner-build-bar";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useSelectConversationTab } from "#/hooks/use-select-conversation-tab";
import { useTaskList } from "#/hooks/use-task-list";
import { mobileTopBarIconClassName } from "#/utils/mobile-top-bar-icon-button-classes";

export function ConversationTabs({
  variant = "default",
  isPanelResizing = false,
}: {
  variant?: "default" | "compact";
  /** True while the desktop drawer gripper is being dragged. */
  isPanelResizing?: boolean;
}) {
  const { conversationId } = useConversationId();
  const { setSelectedTab } = useConversationStore();

  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const { state: persistedState } =
    useConversationLocalStorageState(conversationId);

  const { hasTaskList } = useTaskList();

  const {
    selectTab,
    isTabActive,
    onTabChange,
    selectedTab,
    isRightPanelShown,
  } = useSelectConversationTab();

  // Restore the most-recently-used tab from localStorage so users don't
  // lose their tab selection across reloads.
  //
  // Note: we deliberately do NOT mirror `rightPanelShown` from
  // localStorage. The drawer's open/closed state is session-only — see
  // the comment in `useConversationStore` and the schema note in
  // `conversation-local-storage.ts` for the rationale.
  useEffect(() => {
    setSelectedTab(persistedState.selectedTab);
  }, [setSelectedTab, persistedState.selectedTab]);

  useEffect(() => {
    const handlePanelVisibilityChange = () => {
      if (isRightPanelShown) {
        // If no tab is selected, default to files tab
        if (!selectedTab) {
          onTabChange("files");
        }
      }
    };

    handlePanelVisibilityChange();
  }, [isRightPanelShown, selectedTab, onTabChange]);

  const { t, i18n } = useTranslation("openhands");

  // `files` is intentionally the leftmost tab — it's the primary entry
  // point for inspecting agent output (workspace files + git diff).
  const tabs = [
    {
      tabValue: "files",
      isActive: isTabActive("files"),
      icon: DocumentIcon,
      onClick: () => selectTab("files"),
      tooltipContent: t(CONVERSATION_TAB_LABEL_KEYS.files),
      tooltipAriaLabel: t(CONVERSATION_TAB_LABEL_KEYS.files),
      label: t(CONVERSATION_TAB_LABEL_KEYS.files),
    },
    {
      tabValue: "commits",
      isActive: isTabActive("commits"),
      icon: LuFileDiff,
      onClick: () => selectTab("commits"),
      tooltipContent: t(CONVERSATION_TAB_LABEL_KEYS.commits),
      tooltipAriaLabel: t(CONVERSATION_TAB_LABEL_KEYS.commits),
      label: t(CONVERSATION_TAB_LABEL_KEYS.commits),
    },
    {
      tabValue: "planner",
      isActive: isTabActive("planner"),
      icon: ListTodo,
      onClick: () => selectTab("planner"),
      tooltipContent: t(CONVERSATION_TAB_LABEL_KEYS.planner),
      tooltipAriaLabel: t(CONVERSATION_TAB_LABEL_KEYS.planner),
      label: t(CONVERSATION_TAB_LABEL_KEYS.planner),
    },
    {
      tabValue: "terminal",
      isActive: isTabActive("terminal"),
      icon: SquareChevronRight,
      onClick: () => selectTab("terminal"),
      tooltipContent: t(CONVERSATION_TAB_LABEL_KEYS.terminal),
      tooltipAriaLabel: t(CONVERSATION_TAB_LABEL_KEYS.terminal),
      label: t(CONVERSATION_TAB_LABEL_KEYS.terminal),
      className: "pl-2",
    },
    {
      tabValue: "browser",
      isActive: isTabActive("browser"),
      icon: Globe,
      onClick: () => selectTab("browser"),
      tooltipContent: t(CONVERSATION_TAB_LABEL_KEYS.browser),
      tooltipAriaLabel: t(CONVERSATION_TAB_LABEL_KEYS.browser),
      label: t(CONVERSATION_TAB_LABEL_KEYS.browser),
    },
    {
      tabValue: "usage",
      isActive: isTabActive("usage"),
      icon: Gauge,
      onClick: () => selectTab("usage"),
      tooltipContent: t(CONVERSATION_TAB_LABEL_KEYS.usage),
      tooltipAriaLabel: t(CONVERSATION_TAB_LABEL_KEYS.usage),
      label: t(CONVERSATION_TAB_LABEL_KEYS.usage),
    },
  ];

  if (hasTaskList) {
    // Insert after the file-related tabs.
    tabs.splice(2, 0, {
      tabValue: "tasklist",
      isActive: isTabActive("tasklist"),
      icon: DoubleCheckIcon,
      onClick: () => selectTab("tasklist"),
      tooltipContent: t(CONVERSATION_TAB_LABEL_KEYS.tasklist),
      tooltipAriaLabel: t(CONVERSATION_TAB_LABEL_KEYS.tasklist),
      label: t(CONVERSATION_TAB_LABEL_KEYS.tasklist),
    });
  }

  // Pinned tabs always show in the bar. Unpinned tabs stay hidden unless the
  // user has that tab selected — then it appears while active so the bar
  // matches the open panel.
  const visibleTabs = tabs.filter((tab) => {
    if (!persistedState.unpinnedTabs.includes(tab.tabValue)) return true;
    return selectedTab === tab.tabValue;
  });

  const unpinnedSignature = persistedState.unpinnedTabs.join(",");

  const tabsRowInnerRef = useRef<HTMLDivElement>(null);
  const measureRowRef = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const anchorRef = useRef<HTMLButtonElement>(null);
  const [inlineTabCount, setInlineTabCount] = useState(visibleTabs.length);

  useLayoutEffect(() => {
    const rowInner = tabsRowInnerRef.current;
    const measureRow = measureRowRef.current;
    const menuEl = menuRef.current;
    if (!rowInner || !measureRow || !menuEl) return undefined;

    const measure = () => {
      const measureButtons = measureRow.querySelectorAll<HTMLButtonElement>(
        '[data-tab-measure="true"]',
      );
      const tabCount = measureButtons.length;

      const rowWidth = rowInner.getBoundingClientRect().width;
      if (rowWidth === 0) {
        setInlineTabCount(tabCount);
        return;
      }

      const widths = Array.from(measureButtons).map(
        (button) => button.getBoundingClientRect().width,
      );

      if (widths.length !== tabCount || tabCount === 0) {
        setInlineTabCount(Math.max(0, tabCount));
        return;
      }

      const menuWidth = menuEl.getBoundingClientRect().width;
      const gapCss =
        getComputedStyle(rowInner).columnGap || getComputedStyle(rowInner).gap;
      const gapPx = parseFloat(gapCss) || 6;

      let nextCount = 0;
      for (let k = tabCount; k >= 0; k -= 1) {
        let total = menuWidth;
        for (let i = 0; i < k; i += 1) {
          total += widths[i] ?? 0;
        }
        if (k > 0) {
          total += k * gapPx;
        }
        if (total <= rowWidth + 0.5) {
          nextCount = k;
          break;
        }
      }

      setInlineTabCount((prev) => (prev === nextCount ? prev : nextCount));
    };

    measure();
    if (typeof ResizeObserver === "undefined") return undefined;
    const ro = new ResizeObserver(measure);
    ro.observe(rowInner);
    return () => ro.disconnect();
  }, [
    unpinnedSignature,
    visibleTabs.length,
    hasTaskList,
    selectedTab,
    isRightPanelShown,
    i18n.language,
  ]);

  const safeInlineTabCount = Math.min(inlineTabCount, visibleTabs.length);
  const inlineTabs = visibleTabs.slice(0, safeInlineTabCount);

  // Roving tabindex: the strip is a single tab stop. It sits on the selected
  // tab — drawer open or not, since the strip stays in the accessibility tree
  // while the drawer is collapsed — until the user arrows elsewhere.
  const selectedTabIndex = inlineTabs.findIndex(
    (tab) => tab.tabValue === selectedTab,
  );
  const [rovingTabIndex, setRovingTabIndex] = useState(0);
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);

  useEffect(() => {
    if (selectedTabIndex >= 0) setRovingTabIndex(selectedTabIndex);
  }, [selectedTabIndex]);

  const tabStopIndex = Math.min(
    rovingTabIndex,
    Math.max(inlineTabs.length - 1, 0),
  );

  // Arrow keys move focus only; Enter/Space still activate, which keeps the
  // drawer from thrashing through tabs as the user scans the strip.
  const handleTabKeyDown = (
    event: KeyboardEvent<HTMLButtonElement>,
    index: number,
  ) => {
    const lastIndex = inlineTabs.length - 1;
    if (lastIndex < 0) return;

    let nextIndex: number;
    if (event.key === "ArrowRight") {
      nextIndex = index === lastIndex ? 0 : index + 1;
    } else if (event.key === "ArrowLeft") {
      nextIndex = index === 0 ? lastIndex : index - 1;
    } else if (event.key === "Home") {
      nextIndex = 0;
    } else if (event.key === "End") {
      nextIndex = lastIndex;
    } else {
      return;
    }

    event.preventDefault();
    setRovingTabIndex(nextIndex);
    tabRefs.current[nextIndex]?.focus();
  };

  return (
    <>
      <div
        className={cn(
          "relative w-full min-w-0",
          variant === "compact"
            ? "flex h-full min-h-0 items-center py-0 pl-0 pr-1"
            : "min-h-10 p-1",
        )}
      >
        <div
          ref={measureRowRef}
          aria-hidden
          className="pointer-events-none absolute top-0 left-[-10000px] flex flex-nowrap items-center gap-1.5"
        >
          {visibleTabs.map(
            (
              {
                tabValue,
                icon,
                isActive,
                tooltipContent,
                tooltipAriaLabel,
                label,
                className: tabClassName,
              },
              index,
            ) => (
              <ChatActionTooltip
                key={`measure-${tabValue}-${index}`}
                tooltip={tooltipContent}
                ariaLabel={tooltipAriaLabel}
              >
                <ConversationTabNav
                  tabValue={tabValue}
                  icon={icon}
                  onClick={() => {}}
                  isActive={isActive}
                  label={label}
                  className={cn(tabClassName, "shrink-0")}
                  measureOnly
                />
              </ChatActionTooltip>
            ),
          )}
        </div>
        <div
          ref={tabsRowInnerRef}
          className="flex w-full min-w-0 flex-nowrap items-center justify-start"
        >
          <div className="flex min-w-0 flex-1 items-center justify-start overflow-hidden">
            <div className="flex w-fit max-w-full min-w-0 items-center gap-1.5">
              <LayoutGroup id="conversation-drawer-tabs">
                <div
                  // A narrow drawer can push every tab into the overflow menu,
                  // and a tablist owning no tabs is not a tablist.
                  role={inlineTabs.length > 0 ? "tablist" : undefined}
                  aria-label={t(I18nKey.CONVERSATION$TABS_LABEL)}
                  className="flex w-fit max-w-full min-w-0 flex-nowrap items-center gap-1.5 overflow-x-hidden"
                >
                  {inlineTabs.map(
                    (
                      {
                        tabValue,
                        icon,
                        onClick,
                        isActive,
                        tooltipContent,
                        tooltipAriaLabel,
                        label,
                        className: tabClassName,
                      },
                      index,
                    ) => (
                      <ChatActionTooltip
                        key={`${tabValue}-${index}`}
                        tooltip={tooltipContent}
                        ariaLabel={tooltipAriaLabel}
                      >
                        <ConversationTabNav
                          tabValue={tabValue}
                          icon={icon}
                          onClick={onClick}
                          isActive={isActive}
                          isSelected={selectedTab === tabValue}
                          label={label}
                          className={cn(tabClassName, "shrink-0")}
                          suppressLayoutAnimation={isPanelResizing}
                          tabIndex={index === tabStopIndex ? 0 : -1}
                          onKeyDown={(event) => handleTabKeyDown(event, index)}
                          buttonRef={(node) => {
                            tabRefs.current[index] = node;
                          }}
                        />
                      </ChatActionTooltip>
                    ),
                  )}
                </div>
              </LayoutGroup>
              <div ref={menuRef} className="relative shrink-0">
                <EllipsisButton
                  ref={anchorRef}
                  onClick={() => setIsMenuOpen(!isMenuOpen)}
                  ariaLabel={t(I18nKey.CONVERSATION$CUSTOMIZE_TABS)}
                  iconClassName={
                    variant === "compact"
                      ? mobileTopBarIconClassName
                      : undefined
                  }
                />
                <ConversationTabsContextMenu
                  isOpen={isMenuOpen}
                  onClose={() => setIsMenuOpen(false)}
                  ignoreOutsideClickRef={anchorRef}
                  anchorRef={anchorRef}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
      {/* At phone width the panel page renders this below its fixed-height
          top bar, where the compact tab row lives. */}
      {variant === "default" && <ConversationPlannerBuildBar />}
    </>
  );
}
