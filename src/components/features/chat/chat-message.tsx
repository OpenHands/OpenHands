import React from "react";
import { useTranslation } from "react-i18next";
import { cn } from "#/utils/utils";
import { CopyToClipboardButton } from "#/components/shared/buttons/copy-to-clipboard-button";
import type { SourceType } from "#/types/agent-server/core/base/common";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { I18nKey } from "#/i18n/declaration";
import { TextShimmer } from "#/components/shared/text-shimmer";
import { MarkdownRenderer } from "../markdown/markdown-renderer";
import { PendingStopIcon } from "./pending-stop-icon";
import {
  UserMessageBody,
  chatBubbleMarkdownComponents,
  USER_MESSAGE_LINE_HEIGHT_PX,
} from "./user-message-body";

export type ChatMessagePendingStatus = "sending" | "error";

interface ChatMessageProps {
  type: SourceType;
  message: string;
  actions?: Array<{
    icon: React.ReactNode;
    onClick: () => void;
    tooltip?: string;
  }>;
  isFromPlanningAgent?: boolean;
  pendingStatus?: ChatMessagePendingStatus;
  onRetry?: () => void;
  onStop?: () => void;
}

export function ChatMessage({
  type,
  message,
  children,
  actions,
  isFromPlanningAgent = false,
  pendingStatus,
  onRetry,
  onStop,
}: React.PropsWithChildren<ChatMessageProps>) {
  const { t } = useTranslation("openhands");
  const [isHovering, setIsHovering] = React.useState(false);
  const [isCopy, setIsCopy] = React.useState(false);
  const [isExpanded, setIsExpanded] = React.useState(false);
  const [isTruncatable, setIsTruncatable] = React.useState(false);
  const [isSingleLinePendingMessage, setIsSingleLinePendingMessage] =
    React.useState(true);
  const pendingMessageContentRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    setIsExpanded(false);
  }, [message]);

  const handleCopyToClipboard = async () => {
    await navigator.clipboard.writeText(message);
    setIsCopy(true);
  };

  React.useEffect(() => {
    let timeout: NodeJS.Timeout;

    if (isCopy) {
      timeout = setTimeout(() => {
        setIsCopy(false);
      }, 2000);
    }

    return () => {
      clearTimeout(timeout);
    };
  }, [isCopy]);

  const isPendingUserMessage =
    type === "user" &&
    (pendingStatus === "error" || pendingStatus === "sending");
  const canStopPendingMessage = pendingStatus === "sending" && onStop != null;
  const showStopButton = canStopPendingMessage && isHovering;
  const useTruncatedUserBody = type === "user" && pendingStatus == null;
  const isCollapsed = useTruncatedUserBody && isTruncatable && !isExpanded;
  const hasBubbleChildren = React.Children.count(children) > 0;

  React.useLayoutEffect(() => {
    if (!canStopPendingMessage || useTruncatedUserBody) {
      return undefined;
    }

    const content = pendingMessageContentRef.current;
    if (!content) {
      return undefined;
    }

    const measure = () => {
      const lineHeight = Number.parseFloat(
        getComputedStyle(content).lineHeight,
      );
      const linePx =
        Number.isFinite(lineHeight) && lineHeight > 0
          ? lineHeight
          : USER_MESSAGE_LINE_HEIGHT_PX;

      setIsSingleLinePendingMessage(content.scrollHeight <= linePx + 1);
    };

    measure();

    const observer = new ResizeObserver(measure);
    observer.observe(content);

    return () => observer.disconnect();
  }, [message, canStopPendingMessage, useTruncatedUserBody]);

  const messageContent = useTruncatedUserBody ? (
    <UserMessageBody
      message={message}
      isHovering={isHovering}
      isExpanded={isExpanded}
      onTruncatableChange={setIsTruncatable}
    />
  ) : (
    <div
      ref={pendingMessageContentRef}
      className="min-w-0 text-sm leading-6 whitespace-normal [word-break:break-word]"
    >
      <MarkdownRenderer
        includeStandard
        includeHeadings
        allowHtml={type !== "user"}
        components={chatBubbleMarkdownComponents}
      >
        {message}
      </MarkdownRenderer>
    </div>
  );

  const renderedMessageContent = messageContent;

  const messageBubble = (
    <article
      data-testid={`${type}-message`}
      data-pending-status={pendingStatus}
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
      className={cn(
        "rounded-xl relative w-fit max-w-full flex flex-col",
        hasBubbleChildren && "gap-2",
        type === "user" && "mt-6 bg-tertiary self-end px-4 py-2.5",
        type === "agent" && "mt-6 w-full max-w-full bg-transparent",
        isFromPlanningAgent &&
          type === "agent" &&
          "border border-[#597ff4] bg-tertiary p-4 mt-2",
        pendingStatus === "error" &&
          "border border-[var(--oh-status-error)]/40",
        !isPendingUserMessage && "last:mb-4",
      )}
    >
      <div
        className={cn(
          "absolute -top-2.5 -right-2.5 z-10",
          !isHovering || pendingStatus === "sending" ? "hidden" : "flex",
          "items-center gap-1",
        )}
        onClick={(event) => event.stopPropagation()}
      >
        {actions?.map((action, index) =>
          action.tooltip ? (
            <StyledTooltip key={index} content={action.tooltip} placement="top">
              <button
                type="button"
                onClick={action.onClick}
                className="button-base p-1 cursor-pointer"
                aria-label={action.tooltip}
              >
                {action.icon}
              </button>
            </StyledTooltip>
          ) : (
            <button
              key={index}
              type="button"
              onClick={action.onClick}
              className="button-base p-1 cursor-pointer"
              aria-label={`Action ${index + 1}`}
            >
              {action.icon}
            </button>
          ),
        )}

        <CopyToClipboardButton
          isHidden={!isHovering}
          isDisabled={isCopy}
          onClick={handleCopyToClipboard}
          mode={isCopy ? "copied" : "copy"}
        />
      </div>

      {renderedMessageContent}

      {canStopPendingMessage ? (
        <button
          type="button"
          onClick={(event) => {
            event.stopPropagation();
            onStop?.();
          }}
          data-testid="chat-message-stop"
          aria-label={t(I18nKey.BUTTON$STOP)}
          aria-hidden={!showStopButton}
          tabIndex={showStopButton ? 0 : -1}
          className={cn(
            "group absolute z-10 inline-flex h-7 w-7 cursor-pointer items-center justify-center rounded-full bg-[var(--oh-color-tertiary)] text-[var(--oh-foreground)] transition-opacity duration-150",
            isSingleLinePendingMessage
              ? "right-3 top-1/2 -translate-y-1/2"
              : "right-3 bottom-2.5",
            showStopButton ? "opacity-100" : "pointer-events-none opacity-0",
          )}
        >
          <PendingStopIcon className="block h-7 w-7 max-w-none" />
        </button>
      ) : null}

      {isCollapsed ? (
        <button
          type="button"
          data-testid="chat-message-expand"
          aria-expanded={false}
          aria-label={t(I18nKey.COMMON$VIEW_MORE)}
          className="absolute inset-0 z-[1] cursor-pointer rounded-xl border-0 bg-transparent p-0"
          onClick={() => setIsExpanded(true)}
        />
      ) : null}

      {children}
    </article>
  );

  if (type === "user" && pendingStatus === "error") {
    return (
      <div className="flex w-fit max-w-full flex-col items-end gap-1.5 self-end last:mb-4">
        {messageBubble}
        <div
          role="alert"
          data-testid="chat-message-error"
          className="flex items-center gap-2 text-xs text-[var(--oh-status-error)]"
        >
          <span>{t(I18nKey.CHAT_INTERFACE$MESSAGE_SEND_FAILED)}</span>
          {onRetry ? (
            <button
              type="button"
              onClick={onRetry}
              className="cursor-pointer rounded-md border border-[var(--oh-border)] px-2 py-1 text-xs font-normal text-[var(--oh-foreground)] hover:bg-[var(--oh-interactive-hover)]"
              data-testid="chat-message-retry"
            >
              {t(I18nKey.CHAT_INTERFACE$MESSAGE_RETRY)}
            </button>
          ) : null}
        </div>
      </div>
    );
  }

  if (type === "user" && pendingStatus === "sending") {
    return (
      <div className="flex w-full max-w-full flex-col last:mb-4">
        {messageBubble}
        <div className="my-1 w-full py-1 text-sm">
          <TextShimmer
            as="p"
            role="status"
            aria-live="polite"
            data-testid="chat-message-sending"
            className="block w-full text-sm font-normal"
            duration={1}
            spread={2}
          >
            {t(I18nKey.CHAT_INTERFACE$MESSAGE_SENDING)}
          </TextShimmer>
        </div>
      </div>
    );
  }

  return messageBubble;
}
