import React from "react";
import { useTranslation } from "react-i18next";
import { cn } from "#/utils/utils";
import { CopyToClipboardButton } from "#/components/shared/buttons/copy-to-clipboard-button";
import { OpenHandsSourceType } from "#/types/core/base";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { I18nKey } from "#/i18n/declaration";
import { MarkdownRenderer } from "../markdown/markdown-renderer";

interface ChatMessageProps {
  type: OpenHandsSourceType;
  message: string;
  actions?: Array<{
    icon: React.ReactNode;
    onClick: () => void;
    tooltip?: string;
  }>;
  isFromPlanningAgent?: boolean;
  /**
   * When true, renders a "Delivering..." status indicator beneath the
   * message bubble. Used while a user message has been queued server-side
   * and is awaiting WebSocket delivery (issue #14181).
   */
  isPendingDelivery?: boolean;
}

export function ChatMessage({
  type,
  message,
  children,
  actions,
  isFromPlanningAgent = false,
  isPendingDelivery = false,
}: React.PropsWithChildren<ChatMessageProps>) {
  const { t } = useTranslation();
  const [isHovering, setIsHovering] = React.useState(false);
  const [isCopy, setIsCopy] = React.useState(false);

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

  return (
    <article
      data-testid={`${type}-message`}
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
      className={cn(
        "rounded-xl relative w-fit max-w-full last:mb-4",
        "flex flex-col gap-2",
        type === "user" && "p-4 bg-tertiary self-end",
        type === "agent" && "mt-6 w-full max-w-full bg-transparent",
        isFromPlanningAgent &&
          type === "agent" &&
          "border border-[#597ff4] bg-tertiary p-4 mt-2",
      )}
    >
      <div
        className={cn(
          "absolute -top-2.5 -right-2.5",
          !isHovering ? "hidden" : "flex",
          "items-center gap-1",
        )}
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

      <div
        className="text-sm"
        style={{
          whiteSpace: "normal",
          wordBreak: "break-word",
        }}
      >
        <MarkdownRenderer includeStandard>{message}</MarkdownRenderer>
      </div>

      {isPendingDelivery && (
        <div
          data-testid="delivering-indicator"
          role="status"
          aria-live="polite"
          className={cn(
            "flex items-center gap-1 text-xs italic",
            "text-content-2/70",
          )}
        >
          <span>{t(I18nKey.CHAT_INTERFACE$MESSAGE_DELIVERING)}</span>
          <span className="inline-flex items-end gap-0.5" aria-hidden="true">
            <span className="animate-[pulse_1.2s_ease-in-out_infinite]">.</span>
            <span className="animate-[pulse_1.2s_ease-in-out_0.2s_infinite]">
              .
            </span>
            <span className="animate-[pulse_1.2s_ease-in-out_0.4s_infinite]">
              .
            </span>
          </span>
        </div>
      )}

      {children}
    </article>
  );
}
