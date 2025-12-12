import React from "react";
import { useTranslation } from "react-i18next";
import { cn } from "#/utils/utils";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { MarkdownRenderer } from "#/components/features/markdown/markdown-renderer";

interface PendingMessageProps {
  message: string;
}

export function PendingMessage({ message }: PendingMessageProps) {
  const { t } = useTranslation();

  return (
    <article
      data-testid="pending-message"
      className={cn(
        "rounded-xl relative w-fit max-w-full last:mb-4",
        "flex flex-col gap-2",
        "p-4 bg-tertiary self-end",
        "opacity-60",
      )}
    >
      <div
        className="text-sm"
        style={{
          whiteSpace: "normal",
          wordBreak: "break-word",
        }}
      >
        <MarkdownRenderer includeStandard>{message}</MarkdownRenderer>
      </div>

      <div className="flex items-center gap-2 text-xs text-muted mt-1">
        <LoadingSpinner size="small" />
        <span>{t("PENDING_MESSAGE$WAITING_TEXT")}</span>
      </div>
    </article>
  );
}
