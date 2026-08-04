import React from "react";
import ReactDOM from "react-dom";
import { Bell } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useAgentNotifications } from "#/hooks/chat/use-agent-notifications";
import { useDetectAgentNotifications } from "#/hooks/chat/use-detect-agent-notifications";
import { usePopoverFixedPlacement } from "#/hooks/use-popover-fixed-placement";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import {
  formControlTransitionClassName,
  formControlMutedHoverClassName,
} from "#/utils/form-control-classes";
import { AgentNotificationsList } from "../chat/agent-notifications-list";
import { AgentNotificationsTitle } from "../chat/agent-notifications-title";
import { AgentNotificationsDropdownEmptyState } from "./agent-notifications-dropdown-empty-state";
import { AgentNotificationsDetectButton } from "./agent-notifications-detect-button";

const DROPDOWN_WIDTH_PX = 400;

interface AgentNotificationsBellProps {
  conversationId: string;
}

export function AgentNotificationsBell({
  conversationId,
}: AgentNotificationsBellProps) {
  const { t } = useTranslation("openhands");
  const [isOpen, setIsOpen] = React.useState(false);
  const bellRef = React.useRef<HTMLButtonElement>(null);
  const dropdownRef = React.useRef<HTMLDivElement>(null);

  const {
    agentNotifications: recentAgentNotifications,
    history,
    createAll,
    remove,
    isCreating,
  } = useAgentNotifications({
    conversationId,
    enabled: true,
  });
  const { detectNow } = useDetectAgentNotifications(conversationId);

  const placement = usePopoverFixedPlacement(bellRef, {
    open: isOpen,
    enabled: true,
    targetWidth: DROPDOWN_WIDTH_PX,
    horizontalAlign: "center",
  });

  React.useEffect(() => {
    if (!isOpen) {
      return undefined;
    }

    const handlePointerDownOutside = (event: MouseEvent) => {
      const target = event.target as Node;
      if (
        bellRef.current?.contains(target) ||
        dropdownRef.current?.contains(target)
      ) {
        return;
      }
      setIsOpen(false);
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", handlePointerDownOutside);
    document.addEventListener("keydown", handleEscape);
    return () => {
      document.removeEventListener("mousedown", handlePointerDownOutside);
      document.removeEventListener("keydown", handleEscape);
    };
  }, [isOpen]);

  const handleToggle = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsOpen((current) => !current);
  };

  const handleCreateAll = (selectedIds: string[]) => {
    createAll(selectedIds);
    setIsOpen(false);
  };

  const handleDetect = (event?: React.MouseEvent<HTMLButtonElement>) => {
    event?.preventDefault();
    event?.stopPropagation();

    const result = detectNow();
    if (result.added > 0) {
      displaySuccessToast(
        t(I18nKey.CHAT_INTERFACE$AGENT_NOTIFICATIONS_DETECT_ADDED, {
          count: result.added,
        }),
      );
      return;
    }

    if (result.found > 0) {
      displaySuccessToast(
        t(I18nKey.CHAT_INTERFACE$AGENT_NOTIFICATIONS_DETECT_UP_TO_DATE),
      );
      return;
    }

    displayErrorToast(
      t(I18nKey.CHAT_INTERFACE$AGENT_NOTIFICATIONS_DETECT_NONE),
    );
  };

  return (
    <>
      <button
        ref={bellRef}
        type="button"
        data-testid="agent-notifications-bell"
        aria-label={t(I18nKey.CHAT_INTERFACE$AGENT_NOTIFICATIONS_TITLE)}
        aria-expanded={isOpen}
        aria-haspopup="dialog"
        onClick={handleToggle}
        className={cn(
          "relative shrink-0 p-1 rounded-md cursor-pointer",
          formControlTransitionClassName,
          "text-[var(--oh-muted)]",
          formControlMutedHoverClassName,
          "flex items-center justify-center",
        )}
      >
        <Bell className="w-4 h-4" aria-hidden />
        {recentAgentNotifications.length > 0 && (
          <span
            aria-hidden
            data-testid="agent-notifications-bell-badge"
            className="absolute right-0.5 top-0.5 block size-1.5 rounded-full bg-primary"
          />
        )}
      </button>

      {isOpen && placement && typeof document !== "undefined"
        ? ReactDOM.createPortal(
            <div
              ref={dropdownRef}
              data-testid="agent-notifications-bell-dropdown"
              style={{
                position: "fixed",
                top: placement.top,
                left: placement.left,
                width: placement.width,
                zIndex: 9999,
              }}
              className={cn(
                "rounded-md border border-[var(--oh-border-subtle)]",
                "bg-tertiary p-3 shadow-lg",
              )}
            >
              <div className="mb-3 flex items-center gap-3">
                <div className="min-w-0 flex-1">
                  <AgentNotificationsTitle infoTestId="agent-notifications-bell-info" />
                </div>
                <AgentNotificationsDetectButton
                  onDetect={handleDetect}
                  disabled={isCreating}
                  testId="agent-notifications-bell-detect"
                />
              </div>
              {history.length > 0 ? (
                <AgentNotificationsList
                  agentNotifications={history}
                  onSubmit={handleCreateAll}
                  onRemove={remove}
                  isSubmitting={isCreating}
                  submitTestId="agent-notifications-bell-create-all"
                  listItemTestIdPrefix="agent-notifications-bell-item"
                />
              ) : (
                <AgentNotificationsDropdownEmptyState />
              )}
            </div>,
            document.body,
          )
        : null}
    </>
  );
}
