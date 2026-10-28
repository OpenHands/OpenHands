import React from "react";
import { useTranslation } from "react-i18next";
import { X } from "lucide-react";
import { MarkdownRenderer } from "#/components/features/markdown/markdown-renderer";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import type { ReplyQuote } from "#/utils/message-quote";

interface QuoteCardProps {
  quote: ReplyQuote;
  /** Jump to the source message; omitted until the jump hook is wired. */
  onJump?: (sourceEventId: string) => void;
  onDismiss?: () => void;
}

/**
 * The quoted-source card shared by the composer slot and message bubbles:
 * localized source label, the quoted body (markdown-rendered, clamped),
 * a truncation label when flagged, a dismiss button, and optional
 * click-to-jump to the source message.
 */
export function QuoteCard({ quote, onJump, onDismiss }: QuoteCardProps) {
  const { t } = useTranslation();

  const sourceLabel =
    quote.source === "user"
      ? t(I18nKey.CHAT_INTERFACE$REPLY_YOU)
      : t(I18nKey.CHAT_INTERFACE$REPLY_AGENT);

  const handleJump = () => {
    onJump?.(quote.sourceEventId);
  };

  return (
    <div
      data-testid="quote-card"
      className={cn(
        "bg-tertiary border border-tertiary flex w-full flex-col gap-1 rounded-lg px-3 py-2 text-left",
        onJump && "cursor-pointer",
      )}
      {...(onJump
        ? {
            role: "button",
            tabIndex: 0,
            onClick: handleJump,
            onKeyDown: (event: React.KeyboardEvent) => {
              if (event.key === "Enter") handleJump();
            },
            "aria-label": `${sourceLabel}: ${quote.body}`,
          }
        : {})}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-muted">{sourceLabel}</span>
        <div className="flex items-center gap-1">
          {quote.truncated && (
            <span className="text-xs text-muted">
              {t(I18nKey.CHAT_INTERFACE$REPLY_TRUNCATED)}
            </span>
          )}
          {onDismiss && (
            <button
              type="button"
              aria-label={t(I18nKey.CHAT_INTERFACE$REPLY_DISMISS)}
              onClick={(event) => {
                event.stopPropagation();
                onDismiss();
              }}
              className="cursor-pointer text-muted hover:text-foreground"
            >
              <X size={14} />
            </button>
          )}
        </div>
      </div>
      <div className="line-clamp-3 text-sm text-cool-grey-300">
        <MarkdownRenderer>{quote.body}</MarkdownRenderer>
      </div>
    </div>
  );
}
