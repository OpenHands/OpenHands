import React from "react";
import { Trans, useTranslation } from "react-i18next";
import { Link } from "react-router";
import { X } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";

interface ErrorMessageBannerProps {
  message: string;
  onDismiss?: () => void;
}

const DEFAULT_MAX_COLLAPSED_CHARS = 220;

export function ErrorMessageBanner({
  message,
  onDismiss,
}: ErrorMessageBannerProps) {
  const { t, i18n } = useTranslation();
  const [isExpanded, setIsExpanded] = React.useState(false);

  const isI18nKey = i18n.exists(message);
  const displayTextForLength = isI18nKey ? String(t(message)) : message;
  const shouldShowToggle =
    displayTextForLength.length > DEFAULT_MAX_COLLAPSED_CHARS;

  const isCollapsed = shouldShowToggle && !isExpanded;
  const clampStyle: React.CSSProperties | undefined = isCollapsed
    ? {
        display: "-webkit-box",
        WebkitLineClamp: 3,
        WebkitBoxOrient: "vertical",
        overflow: "hidden",
      }
    : undefined;

  return (
    <div
      className="w-full rounded-lg p-2 text-black border border-red-800 bg-red-500 flex gap-2 items-start"
      data-testid="error-message-banner"
    >
      <div className="min-w-0 flex-1">
        <div
          className="whitespace-pre-wrap break-words"
          style={clampStyle}
          data-testid="error-message-banner-content"
        >
          {isI18nKey ? (
            <Trans
              i18nKey={message}
              components={{
                a: (
                  <Link
                    className="underline font-bold cursor-pointer"
                    to="/settings/billing"
                  >
                    link
                  </Link>
                ),
              }}
            />
          ) : (
            message
          )}
        </div>

        {shouldShowToggle && (
          <button
            type="button"
            className="mt-1 text-xs underline font-semibold cursor-pointer"
            onClick={() => setIsExpanded((prev) => !prev)}
            data-testid="error-message-banner-toggle"
          >
            {isExpanded
              ? t(I18nKey.COMMON$VIEW_LESS)
              : t(I18nKey.COMMON$VIEW_MORE)}
          </button>
        )}
      </div>

      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          className="shrink-0 rounded-md p-1 hover:bg-black/10"
          aria-label={t(I18nKey.BUTTON$CLOSE)}
          data-testid="error-message-banner-dismiss"
        >
          <X className="h-4 w-4" />
        </button>
      )}
    </div>
  );
}
