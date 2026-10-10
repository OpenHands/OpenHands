import React from "react";
import { useTranslation } from "react-i18next";
import { useAgentState } from "#/hooks/use-agent-state";
import { useTaskPolling } from "#/hooks/query/use-task-polling";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useUnifiedPauseConversation } from "#/hooks/mutation/use-unified-stop-conversation";
import { useUnifiedResumeConversation } from "#/hooks/mutation/use-unified-start-conversation";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useCloseOnEscape } from "#/hooks/use-close-on-escape";
import { useUserProviders } from "#/hooks/use-user-providers";
import { getStatusColor, cn } from "#/utils/utils";
import { AgentState } from "#/types/agent-state";
import DebugStackframeDot from "#/icons/debug-stackframe-dot.svg?react";
import { ServerStatusContextMenu } from "../controls/server-status-context-menu";
import { ConversationName } from "./conversation-name";
import { ConversationGitActionsToggle } from "./conversation-git-actions-toggle";
import { ConversationOverviewToggle } from "./conversation-overview-toggle";
import { RightPanelToggle } from "./right-panel-toggle";
import { getDisplayConversationTags } from "#/api/agent-server-adapter";
import {
  formatConversationTagTooltip,
  truncateTagChipValue,
} from "../conversation-panel/conversation-card/conversation-tag-display";
import {
  getConversationTagIcon,
  type ConversationTagIcon,
} from "../conversation-panel/conversation-card/conversation-tag-icons";
import {
  CONVERSATION_CARD_META_CHIP_CLASSNAME,
  CONVERSATION_CARD_META_CHIP_ICON_CLASSNAME,
  CONVERSATION_CARD_META_CHIP_ICON_SLOT_CLASSNAME,
} from "../conversation-panel/conversation-card/conversation-card-meta-chip";
import {
  isExecutionActive,
  isExecutionErrored,
  isExecutionPaused,
} from "#/utils/status";
import { I18nKey } from "#/i18n/declaration";

function ChannelOriginIconSlot({
  icon: Icon,
  tagKey,
}: {
  icon: ConversationTagIcon;
  tagKey: string;
}) {
  return (
    <span
      className={CONVERSATION_CARD_META_CHIP_ICON_SLOT_CLASSNAME}
      aria-hidden
    >
      <Icon
        data-testid="conversation-channel-origin-icon"
        data-tag-key={tagKey}
        className={CONVERSATION_CARD_META_CHIP_ICON_CLASSNAME}
        aria-hidden
      />
    </span>
  );
}

export function ConversationNameWithStatus() {
  const { t } = useTranslation("openhands");
  const { conversationId } = useConversationId();
  const { data: conversation } = useActiveConversation();
  const { curAgentState } = useAgentState();
  const { isTask, taskStatus } = useTaskPolling();
  const { mutate: pauseConversation } = useUnifiedPauseConversation();
  const { mutate: resumeConversation } = useUnifiedResumeConversation();
  const { providers } = useUserProviders();
  const [menuOpen, setMenuOpen] = React.useState(false);
  const [hoveredOpen, setHoveredOpen] = React.useState(false);
  const triggerRef = React.useRef<HTMLButtonElement>(null);

  const executionStatus = conversation?.execution_status ?? null;
  const isStartingStatus =
    curAgentState === AgentState.LOADING || curAgentState === AgentState.INIT;
  const isStopStatus = isExecutionErrored(executionStatus);

  const statusColor = getStatusColor({
    isPausing: false,
    isTask,
    taskStatus,
    isStartingStatus,
    isStopStatus,
    curAgentState,
  });

  const isMenuVisible = menuOpen || hoveredOpen;

  const displayTags = React.useMemo(
    () => getDisplayConversationTags(conversation?.tags),
    [conversation?.tags],
  );
  const channelOriginTag = React.useMemo(() => {
    return displayTags.find(([key]) => {
      const normalized = key.trim().toLowerCase();
      return normalized === "origin" || normalized === "source";
    });
  }, [displayTags]);

  const closeMenu = () => {
    setMenuOpen(false);
    setHoveredOpen(false);
  };

  // A menu opened by click or Enter returns focus to the dot; one opened by
  // hover closes without moving focus.
  useCloseOnEscape(isMenuVisible, closeMenu, menuOpen ? triggerRef : undefined);

  const handleStopServer = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    if (conversationId) {
      pauseConversation({ conversationId });
    }
    closeMenu();
  };

  const handleStartServer = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    if (conversationId) {
      resumeConversation({ conversationId, providers });
    }
    closeMenu();
  };

  const handleStatusClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    // Click toggles for touch; on desktop it also dismisses a hover-opened menu
    // instead of "pinning" it open under the cursor.
    if (isMenuVisible) {
      closeMenu();
      return;
    }
    setMenuOpen(true);
  };

  return (
    <div className="flex items-center justify-between w-full">
      <div className="flex items-center min-w-0">
        <div
          className="relative shrink-0"
          onPointerEnter={(event) => {
            // Gate on this event's pointer, not device-primary matchMedia — hybrid
            // laptops can emit compatibility mouseenter for touchscreen taps.
            if (event.pointerType === "mouse") {
              setHoveredOpen(true);
            }
          }}
          onPointerLeave={(event) => {
            if (event.pointerType === "mouse") {
              setHoveredOpen(false);
            }
          }}
        >
          <button
            ref={triggerRef}
            type="button"
            data-testid="server-status-menu-trigger"
            aria-label={t(I18nKey.COMMON$SERVER_STATUS)}
            aria-expanded={isMenuVisible}
            aria-haspopup="menu"
            onClick={handleStatusClick}
            className={cn(
              "flex items-center justify-center rounded-md",
              "text-muted hover:bg-contrast/10",
            )}
          >
            <DebugStackframeDot
              className="ml-[3.5px] w-6 h-6 cursor-pointer"
              color={statusColor}
              aria-hidden
            />
          </button>
          {isMenuVisible ? (
            <ServerStatusContextMenu
              onClose={closeMenu}
              ignoreOutsideClickRef={triggerRef}
              onStopServer={
                isExecutionActive(executionStatus)
                  ? handleStopServer
                  : undefined
              }
              onStartServer={
                isExecutionPaused(executionStatus)
                  ? handleStartServer
                  : undefined
              }
              executionStatus={executionStatus}
              position="bottom"
              className="bottom-full left-0 mt-0 min-h-fit"
              isPausing={false}
            />
          ) : null}
        </div>
        <ConversationName />
        {channelOriginTag ? (
          <span
            data-testid="conversation-channel-origin-indicator"
            title={formatConversationTagTooltip(
              channelOriginTag[0],
              channelOriginTag[1],
              (k) => t(k),
            )}
            className={cn(CONVERSATION_CARD_META_CHIP_CLASSNAME, "ml-2")}
          >
            <ChannelOriginIconSlot
              icon={getConversationTagIcon(
                channelOriginTag[0],
                channelOriginTag[1],
              )}
              tagKey={channelOriginTag[0]}
            />
            <span className="truncate">
              {formatConversationTagTooltip(
                channelOriginTag[0],
                truncateTagChipValue(channelOriginTag[1]),
                (k) => t(k),
              )}
            </span>
          </span>
        ) : null}
      </div>
      <div className="mr-2 flex shrink-0 items-center gap-1">
        <ConversationGitActionsToggle />
        <ConversationOverviewToggle />
        <RightPanelToggle />
      </div>
    </div>
  );
}
