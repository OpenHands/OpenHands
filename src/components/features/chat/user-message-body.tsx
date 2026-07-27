import React from "react";
import { useTranslation } from "react-i18next";
import { cn } from "#/utils/utils";
import { I18nKey } from "#/i18n/declaration";
import { MarkdownRenderer } from "../markdown/markdown-renderer";

const USER_MESSAGE_MAX_LINES = 5;
const USER_MESSAGE_LENGTH_THRESHOLD = 360;
export const USER_MESSAGE_LINE_HEIGHT_PX = 24;

export const chatBubbleMarkdownComponents = {
  p: ({ children }: React.ComponentProps<"p">) => (
    <p className="m-0 leading-6">{children}</p>
  ),
};

export function UserMessageBody({
  message,
  isHovering,
  isExpanded,
  onTruncatableChange,
}: {
  message: string;
  isHovering: boolean;
  isExpanded: boolean;
  onTruncatableChange: (truncatable: boolean) => void;
}) {
  const { t } = useTranslation("openhands");
  const contentRef = React.useRef<HTMLDivElement>(null);
  const [isTruncatable, setIsTruncatable] = React.useState(false);

  React.useEffect(() => {
    onTruncatableChange(isTruncatable);
  }, [isTruncatable, onTruncatableChange]);

  React.useLayoutEffect(() => {
    const content = contentRef.current;
    if (!content || isExpanded) {
      setIsTruncatable(false);
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
      const maxHeight = USER_MESSAGE_MAX_LINES * linePx;
      const newlineCount = (message.match(/\n/g) ?? []).length;

      const truncatable =
        content.scrollHeight > maxHeight + 1 ||
        newlineCount >= USER_MESSAGE_MAX_LINES ||
        message.trim().length > USER_MESSAGE_LENGTH_THRESHOLD;

      setIsTruncatable(truncatable);
    };

    measure();

    const observer = new ResizeObserver(measure);
    observer.observe(content);

    return () => observer.disconnect();
  }, [message, isExpanded]);

  const isCollapsed = isTruncatable && !isExpanded;

  return (
    <div className="relative min-w-0">
      <div
        ref={contentRef}
        className={cn(
          "text-sm leading-6 whitespace-normal [word-break:break-word]",
          isCollapsed && "line-clamp-5",
        )}
      >
        <MarkdownRenderer
          includeStandard
          includeHeadings
          allowHtml={false}
          components={chatBubbleMarkdownComponents}
        >
          {message}
        </MarkdownRenderer>
      </div>

      {isCollapsed ? (
        <>
          <div
            aria-hidden
            data-testid="chat-message-truncation-gradient"
            className="pointer-events-none absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-tertiary to-transparent"
          />
          <span
            data-testid="chat-message-view-more"
            className={cn(
              "pointer-events-none absolute bottom-1 left-1/2 z-10 inline-flex -translate-x-1/2 items-center rounded-full border border-[var(--oh-border-subtle)] bg-[var(--oh-surface-raised)] px-2.5 py-0.5 text-xs font-normal text-[var(--oh-foreground)] transition-opacity duration-150",
              isHovering ? "opacity-100" : "opacity-0",
            )}
          >
            {t(I18nKey.COMMON$VIEW_MORE)}
          </span>
        </>
      ) : null}
    </div>
  );
}
