import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { ArrowLeft, PanelLeftClose, PanelLeftOpen } from "lucide-react";
import { cn } from "#/utils/utils";
import { ChatInterfaceWrapper } from "./chat-interface-wrapper";
import { ConversationTabContent } from "../conversation-tabs/conversation-tab-content/conversation-tab-content";
import { ConversationNameWithStatus } from "../conversation-name-with-status";
import { ConversationTabs } from "../conversation-tabs/conversation-tabs";
import { ResizeHandle } from "../../../ui/resize-handle";
import { useResizablePanels } from "#/hooks/use-resizable-panels";
import { useConversationStore } from "#/stores/conversation-store";
import { AutomationSetupPanel } from "#/components/features/automations/setup/automation-setup-panel";
import {
  clearAutomationSetupDraft,
  getAutomationSetupDraft,
  subscribeAutomationSetupDraft,
  type AutomationSetupDraft,
} from "#/api/automation-setup-draft-store";
import { useConversationId } from "#/hooks/use-conversation-id";
import {
  useBreakpoint,
  SIDEBAR_RAIL_COLLAPSE_MAX_WIDTH,
} from "#/hooks/use-breakpoint";
import { SidebarMobileMenuToggle } from "#/components/features/sidebar/sidebar-mobile-menu-toggle";
import { ConversationOverviewDrawer } from "../conversation-overview-drawer";
import { useConversationOverviewDrawerOptional } from "../conversation-overview-drawer-context";
import { useNavigation } from "#/context/navigation-context";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { I18nKey } from "#/i18n/declaration";
import { formControlTransitionClassName } from "#/utils/form-control-classes";

const SPLASH_ROUTE = "/";

function getDesktopTabPanelClass(isRightPanelShown: boolean) {
  return isRightPanelShown
    ? "translate-x-0 opacity-100"
    : "w-0 translate-x-full opacity-0";
}

export function ConversationMain() {
  const { t } = useTranslation("openhands");
  const { navigate } = useNavigation();
  const { conversationId } = useConversationId();
  const { data: conversation } = useActiveConversation();
  const isMobile = useBreakpoint();
  const isSidebarRailHidden = useBreakpoint(SIDEBAR_RAIL_COLLAPSE_MAX_WIDTH);
  const { isRightPanelShown, setHasRightPanelToggled, setIsRightPanelShown } =
    useConversationStore();
  const [automationSetupDraft, setAutomationSetupDraftState] =
    useState<AutomationSetupDraft | null>(() =>
      getAutomationSetupDraft(conversationId),
    );
  const [automationToolbarElement, setAutomationToolbarElement] =
    useState<HTMLDivElement | null>(null);
  const [isAutomationAgentHidden, setIsAutomationAgentHidden] = useState(false);
  const overviewDrawer = useConversationOverviewDrawerOptional();
  const isSecondaryDrawerOpen = Boolean(overviewDrawer?.section);
  const isAutomationSetupMode = Boolean(automationSetupDraft);

  const { leftWidth, rightWidth, isDragging, containerRef, handleMouseDown } =
    useResizablePanels({
      defaultLeftWidth: 50,
      minLeftWidth: 30,
      maxLeftWidth: 80,
      storageKey: "desktop-layout-panel-width",
    });

  useEffect(() => {
    setAutomationSetupDraftState(getAutomationSetupDraft(conversationId));
    if (!conversationId) return undefined;
    return subscribeAutomationSetupDraft(
      conversationId,
      setAutomationSetupDraftState,
    );
  }, [conversationId]);

  useEffect(() => {
    if (!automationSetupDraft) return;
    setHasRightPanelToggled(true);
    setIsRightPanelShown(true);
  }, [automationSetupDraft, setHasRightPanelToggled, setIsRightPanelShown]);

  useEffect(() => {
    if (!automationSetupDraft) {
      setIsAutomationAgentHidden(false);
    }
  }, [automationSetupDraft]);

  const closeAutomationSetup = () => {
    if (conversationId) clearAutomationSetupDraft(conversationId);
    setAutomationSetupDraftState(null);
    setIsRightPanelShown(false);
  };

  const handleBackToSplash = () => {
    navigate(SPLASH_ROUTE);
  };

  const agentToggleLabel = isAutomationAgentHidden
    ? t(I18nKey.AUTOMATION_SETUP$SHOW_AGENT)
    : t(I18nKey.AUTOMATION_SETUP$HIDE_AGENT);

  return (
    <div
      className={cn(
        isMobile
          ? "relative min-h-0 flex-1 flex flex-col"
          : "h-full flex flex-col overflow-hidden",
      )}
    >
      {isAutomationSetupMode ? (
        <header
          data-testid="automation-setup-topbar"
          className="flex h-10 min-h-10 shrink-0 items-center gap-2 border-b border-[var(--oh-border)] bg-base px-3"
        >
          {isSidebarRailHidden ? <SidebarMobileMenuToggle /> : null}
          <button
            type="button"
            data-testid="automation-setup-back"
            aria-label={t(I18nKey.AUTOMATION_SETUP$BACK_LABEL)}
            onClick={handleBackToSplash}
            className={cn(
              "flex size-7 items-center justify-center rounded-lg text-[var(--oh-muted)] hover:bg-white/10 hover:text-white",
              formControlTransitionClassName,
            )}
          >
            <ArrowLeft className="size-4" aria-hidden />
          </button>
          <h1
            data-testid="automation-setup-conversation-title"
            className="min-w-0 truncate text-sm font-semibold text-white"
          >
            {conversation?.title || t(I18nKey.AUTOMATION_SETUP$TITLE)}
          </h1>
          <div className="ml-auto flex min-w-0 shrink-0 items-center gap-2">
            <div
              ref={setAutomationToolbarElement}
              data-testid="automation-setup-toolbar"
              className="flex min-w-0 shrink-0 items-center gap-2"
            />
            {!isMobile ? (
              <button
                type="button"
                data-testid="automation-setup-agent-toggle"
                aria-label={agentToggleLabel}
                title={agentToggleLabel}
                onClick={() =>
                  setIsAutomationAgentHidden((previous) => !previous)
                }
                className={cn(
                  "flex size-7 items-center justify-center rounded-lg text-[var(--oh-muted)] hover:bg-white/10 hover:text-white",
                  formControlTransitionClassName,
                )}
              >
                {isAutomationAgentHidden ? (
                  <PanelLeftOpen className="size-4" aria-hidden />
                ) : (
                  <PanelLeftClose className="size-4" aria-hidden />
                )}
              </button>
            ) : null}
          </div>
        </header>
      ) : null}

      <div
        ref={containerRef}
        className={cn(
          "flex flex-1 overflow-hidden",
          isMobile ? "flex-col" : "transition-all duration-300 ease-in-out",
        )}
        // transition toggled at runtime based on drag state
        style={
          !isMobile
            ? { transitionProperty: isDragging ? "none" : "all" }
            : undefined
        }
      >
        {/* Chat Panel - always mounted, styled differently for mobile/desktop.
            Owns its own header (name + status) and gets bottom padding so the
            chat input doesn't slam the floor. */}
        <div
          data-testid="conversation-chat-panel"
          className={cn(
            "flex flex-col bg-base overflow-hidden",
            isMobile
              ? "flex-1"
              : cn(
                  "min-w-0",
                  isAutomationSetupMode &&
                    isAutomationAgentHidden &&
                    "pointer-events-none opacity-0",
                  !isSecondaryDrawerOpen &&
                    "transition-[width] duration-300 ease-in-out",
                ),
          )}
          // panel width computed at runtime by resize hook; transition toggled by drag state
          style={
            !isMobile
              ? {
                  width:
                    isAutomationSetupMode && isAutomationAgentHidden
                      ? "0%"
                      : isRightPanelShown
                        ? `${leftWidth}%`
                        : "100%",
                  transitionProperty:
                    isDragging || isSecondaryDrawerOpen ? "none" : "width",
                }
              : undefined
          }
        >
          {!isAutomationSetupMode ? (
            <div
              data-testid="chat-pane-header"
              className={cn(
                "flex h-10 min-h-10 shrink-0 items-center",
                isSidebarRailHidden && "gap-2 pl-2.5",
              )}
            >
              {isSidebarRailHidden ? <SidebarMobileMenuToggle /> : null}
              <div className="min-w-0 flex-1">
                <ConversationNameWithStatus />
              </div>
            </div>
          ) : null}
          <div className="flex-1 min-h-0 flex flex-col">
            <ChatInterfaceWrapper
              isRightPanelShown={!isMobile && isRightPanelShown}
            />
          </div>
        </div>

        {/* Resize Handle - only shown on desktop when right panel is visible */}
        {!isMobile &&
          isRightPanelShown &&
          !(isAutomationSetupMode && isAutomationAgentHidden) && (
            <ResizeHandle
              onMouseDown={handleMouseDown}
              isDragging={isDragging}
            />
          )}

        {/* Right panel: desktop side drawer. Mobile opens Files/Tools via /panel route. */}
        {!isMobile && (
          <div
            data-testid="conversation-right-panel"
            className={cn(
              "transition-all duration-300 ease-in-out overflow-hidden",
              getDesktopTabPanelClass(isRightPanelShown),
            )}
            style={{
              width: isRightPanelShown
                ? isAutomationSetupMode && isAutomationAgentHidden
                  ? "100%"
                  : `${rightWidth}%`
                : "0%",
              transitionProperty: isDragging ? "opacity, transform" : "all",
            }}
          >
            <div className="flex h-full w-full flex-col">
              <div
                className={cn(
                  "flex flex-col flex-1 min-h-0 bg-[var(--oh-surface)] overflow-hidden",
                  !(isAutomationSetupMode && isAutomationAgentHidden) &&
                    "border-l border-[var(--oh-border)]",
                )}
              >
                {automationSetupDraft ? (
                  <AutomationSetupPanel
                    draft={automationSetupDraft}
                    conversationId={conversationId}
                    toolbarPortal={automationToolbarElement}
                    showInlineHeader={false}
                    onClose={closeAutomationSetup}
                  />
                ) : (
                  <>
                    <div
                      data-testid="tabs-pane-header"
                      className="flex shrink-0 flex-col border-b border-[var(--oh-border)]"
                    >
                      <ConversationTabs isPanelResizing={isDragging} />
                    </div>
                    <div className="flex-1 min-h-0 flex flex-col">
                      <ConversationTabContent />
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        )}
        <ConversationOverviewDrawer
          isMobile={isMobile}
          resizeContainerRef={containerRef}
        />
      </div>
    </div>
  );
}
