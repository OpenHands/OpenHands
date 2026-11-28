import React from "react";
import { useTranslation } from "react-i18next";
import ArrowDown from "#/icons/angle-down-solid.svg?react";
import ArrowUp from "#/icons/angle-up-solid.svg?react";
import i18n from "#/i18n";
import { I18nKey } from "#/i18n/declaration";
import { MarkdownRenderer } from "../markdown/markdown-renderer";
import { useEventExpansionKey } from "./row-expansion-context";
import { usePersistentExpansion } from "#/stores/message-expansion-store";

interface ErrorMessageProps {
  errorId?: string;
  defaultMessage: string;
  onRetry?: () => void;
  isRetrying?: boolean;
}

export function ErrorMessage({
  errorId,
  defaultMessage,
  onRetry,
  isRetrying = false,
}: ErrorMessageProps) {
  const { t } = useTranslation("openhands");
  const expansionKey = useEventExpansionKey("error-details");
  const [showDetails, toggleDetails] = usePersistentExpansion(
    expansionKey,
    false,
  );

  const hasValidTranslationId = !!errorId && i18n.exists(errorId);
  const errorKey = hasValidTranslationId
    ? errorId
    : "CHAT_INTERFACE$AGENT_ERROR_MESSAGE";

  return (
    <div className="flex flex-col gap-2 my-2 py-2 text-sm w-full">
      <div className="flex items-center justify-between gap-2">
        <div
          className="font-bold text-danger flex items-center"
          data-testid="error-message-title"
        >
          {t(errorKey)}
          <button
            type="button"
            onClick={() => toggleDetails()}
            className="cursor-pointer text-left"
            data-testid="error-message-toggle"
            aria-label={
              showDetails
                ? t(I18nKey.COMMON$VIEW_LESS)
                : t(I18nKey.COMMON$VIEW_MORE)
            }
          >
            {showDetails ? (
              <ArrowUp className="h-4 w-4 ml-2 inline fill-danger" />
            ) : (
              <ArrowDown className="h-4 w-4 ml-2 inline fill-danger" />
            )}
          </button>
        </div>

        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            disabled={isRetrying}
            className="cursor-pointer rounded-md border border-border px-2.5 py-1 text-xs font-medium text-foreground hover:bg-interactive-hover disabled:cursor-not-allowed disabled:opacity-50"
            data-testid="error-message-retry"
          >
            {t(I18nKey.CHAT_INTERFACE$MESSAGE_RETRY)}
          </button>
        )}
      </div>

      {showDetails && <MarkdownRenderer>{defaultMessage}</MarkdownRenderer>}
    </div>
  );
}
