import React from "react";
import { cn } from "#/utils/utils";
import { CopyToClipboardButton } from "#/components/shared/buttons/copy-to-clipboard-button";
import { OpenHandsSourceType } from "#/types/core/base";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
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
}

export function ChatMessage({
  type,
  message,
  children,
  actions,
  isFromPlanningAgent = false,
}: React.PropsWithChildren<ChatMessageProps>) {
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
        isFromPlanningAgent && "border border-[#597ff4] bg-tertiary p-4 mt-2",
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

      {children}
    </article>
  );
}
