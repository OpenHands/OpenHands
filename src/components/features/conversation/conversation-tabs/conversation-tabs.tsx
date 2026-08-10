import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { LayoutGroup } from "framer-motion";
import { Globe, ListTodo, SquareChevronRight } from "lucide-react";
import DocumentIcon from "#/icons/document.svg?react";
import DoubleCheckIcon from "#/icons/double-check.svg?react";
import { EllipsisButton } from "#/components/features/conversation-panel/ellipsis-button";
import { cn } from "#/utils/utils";
import { useConversationLocalStorageState } from "#/utils/conversation-local-storage";
import { ConversationTabNav } from "./conversation-tab-nav";
import { DrawerVSCodeLink } from "./drawer-vscode-link";
import { ChatActionTooltip } from "../../chat/chat-action-tooltip";
import { I18nKey } from "#/i18n/declaration";
import { useConversationStore } from "#/stores/conversation-store";
import { ConversationTabsContextMenu } from "./conversation-tabs-context-menu";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useSelectConversationTab } from "#/hooks/use-select-conversation-tab";
import { useTaskList } from "#/hooks/use-task-list";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useHandleBuildPlanClick } from "#/hooks/use-handle-build-plan-click";
import { useAgentState } from "#/hooks/use-agent-state";
import { AgentState } from "#/types/agent-state";
import { Typography } from "#/ui/typography";
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
  const { setSelectedTab, planContent } = useConversationStore();

  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const { state: persistedState } =
    useConversationLocalStorageState(conversationId);

  const { hasTaskList } = useTaskList();
  const { backend } = useActiveBackend();

  const { handleBuildPlanClick } = useHandleBuildPlanClick();
  const { curAgentState } = useAgentState();

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
      tooltipContent: t(I18nKey.COMMON$FILES),
      tooltipAriaLabel: t(I18nKey.COMMON$FILES),
      label: t(I18nKey.COMMON$FILES),
    },
    {
      tabValue: "planner",
      isActive: isTabActive("planner"),
      icon: ListTodo,
      onClick: () => selectTab("planner"),
      tooltipContent: t(I18nKey.COMMON$PLANNER),
      tooltipAriaLabel: t(I18nKey.COMMON$PLANNER),
      label: t(I18nKey.COMMON$PLANNER),
    },
    {
      tabValue: "terminal",
      isActive: isTabActive("terminal"),
      icon: SquareChevronRight,
      onClick: () => selectTab("terminal"),
      tooltipContent: t(I18nKey.COMMON$TERMINAL),
      tooltipAriaLabel: t(I18nKey.COMMON$TERMINAL),
      label: t(I18nKey.COMMON$TERMINAL),
      className: "pl-2",
    },
    {
      tabValue: "browser",
      isActive: isTabActive("browser"),
      icon: Globe,
      onClick: () => selectTab("browser"),
      tooltipContent: t(I18nKey.COMMON$BROWSER),
      tooltipAriaLabel: t(I18nKey.COMMON$BROWSER),
      label: t(I18nKey.COMMON$BROWSER),
    },
  ];

  if (hasTaskList) {
    // Insert after `files` so the leftmost slot stays Files.
    tabs.splice(1, 0, {
      tabValue: "tasklist",
      isActive: isTabActive("tasklist"),
      icon: DoubleCheckIcon,
      onClick: () => selectTab("tasklist"),
      tooltipContent: t(I18nKey.COMMON$TASK_LIST),
      tooltipAriaLabel: t(I18nKey.COMMON$TASK_LIST),
      label: t(I18nKey.COMMON$TASK_LIST),
    });
  }

  // Pinned tabs always show in the bar. Unpinned tabs stay hidden unless the
  // user has that tab selected — then it appears while active so the bar
  // matches the open panel. Hide Planner on local backends — the planning
  // agent isn't supported locally.
  const visibleTabs = tabs.filter((tab) => {
    if (tab.tabValue === "planner" && backend.kind !== "cloud") return false;
    if (!persistedState.unpinnedTabs.includes(tab.tabValue)) return true;
    return selectedTab === tab.tabValue;
  });

  const unpinnedSignature = persistedState.unpinnedTabs.join(",");

  const isAgentRunning =
    curAgentState === AgentState.RUNNING ||
    curAgentState === AgentState.LOADING;
  const isBuildDisabled = isAgentRunning || !planContent;

  const tabsRowInnerRef = useRef<HTMLDivElement>(null);
  const measureRowRef = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const vscodeButtonRef = useRef<HTMLDivElement>(null);
  const anchorRef = useRef<HTMLButtonElement>(null);
  const [inlineTabCount, setInlineTabCount] = useState(visibleTabs.length);

  useLayoutEffect(() => {
    const rowInner = tabsRowInnerRef.current;
    const measureRow = measureRowRef.current;
    const menuEl = menuRef.current;
    const vscodeEl = vscodeButtonRef.current;
    if (!rowInner || !measureRow || !menuEl || !vscodeEl) return undefined;

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
      const vscodeWidth = vscodeEl.getBoundingClientRect().width;
      const gapCss =
        getComputedStyle(rowInner).columnGap || getComputedStyle(rowInner).gap;
      const gapPx = parseFloat(gapCss) || 6;

      let nextCount = 0;
      for (let k = tabCount; k >= 0; k -= 1) {
        let total = menuWidth + vscodeWidth;
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
    backend.kind,
    selectedTab,
    isRightPanelShown,
    i18n.language,
  ]);

  const safeInlineTabCount = Math.min(inlineTabCount, visibleTabs.length);

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
                <div className="flex w-fit max-w-full min-w-0 flex-nowrap items-center gap-1.5 overflow-x-hidden">
                  {visibleTabs
                    .slice(0, safeInlineTabCount)
                    .map(
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
                            label={label}
                            className={cn(tabClassName, "shrink-0")}
                            suppressLayoutAnimation={isPanelResizing}
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
                  ariaLabel={t(I18nKey.COMMON$MORE_OPTIONS)}
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
          {/* Keep the ref'd wrapper mounted on local backends too — the
              overflow measurement effect above bails if it's missing. */}
          <div
            ref={vscodeButtonRef}
            className={cn(
              "ml-auto shrink-0",
              backend.kind === "cloud" && "pr-1",
            )}
          >
            {backend.kind === "cloud" && <DrawerVSCodeLink />}
          </div>
        </div>
      </div>
      {isTabActive("planner") && (
        <div
          className={cn(
            "flex h-10 min-h-10 shrink-0 items-center border-t border-[var(--oh-border)] pl-[10px] pr-1",
          )}
        >
          <button
            type="button"
            onClick={handleBuildPlanClick}
            disabled={isBuildDisabled}
            className={cn(
              "flex h-5 min-w-17 items-center justify-center rounded bg-white px-2 transition-opacity",
              isBuildDisabled
                ? "cursor-not-allowed opacity-50"
                : "cursor-pointer hover:opacity-90",
            )}
            data-testid="planner-tab-build-button"
          >
            <Typography.Text className="text-[11px] font-normal leading-5 text-black">
              {/* eslint-disable-next-line i18next/no-literal-string */}
              {t(I18nKey.COMMON$BUILD)} ⌘↩
            </Typography.Text>
          </button>
        </div>
      )}
    </>
  );
}
