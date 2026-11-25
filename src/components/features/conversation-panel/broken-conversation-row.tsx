import { ChevronRight } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { parseDateAsUTC } from "#/utils/format-time-delta";
import { statusToneBadgeClassName } from "#/utils/status-tone-classes";
import { cn } from "#/utils/utils";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";

interface BrokenConversationRowProps {
  conversation: AppConversation;
  onDelete: () => void;
}

export function BrokenConversationRow({
  conversation,
  onDelete,
}: BrokenConversationRowProps) {
  const { t, i18n } = useTranslation("openhands");
  const {
    id,
    title,
    updated_at: updatedAt,
    broken_reason: reason,
  } = conversation;
  return (
    <details className="group rounded-md px-2 py-1 hover:bg-surface">
      <summary className="flex cursor-pointer items-center gap-2 rounded-md focus-visible:outline-2">
        <ChevronRight
          className="size-3 shrink-0 group-open:rotate-90"
          aria-hidden
        />
        <span className="min-w-0 flex-1 truncate text-sm group-open:break-words group-open:whitespace-normal">
          {title}
        </span>
        <span
          className={cn(
            "shrink-0 rounded px-1.5 py-0.5 text-xs",
            statusToneBadgeClassName.danger,
          )}
        >
          {t(I18nKey.CONVERSATION_PANEL$CORRUPTED)}
        </span>
      </summary>
      <div className="flex flex-col items-start gap-2 py-2 text-xs text-muted">
        <p>
          {reason === "metadata"
            ? t(I18nKey.CONVERSATION_PANEL$CORRUPTED_METADATA)
            : t(I18nKey.CONVERSATION_PANEL$CORRUPTED_STATE)}
        </p>
        <code className="break-all select-text">{id}</code>
        <time dateTime={updatedAt}>
          {parseDateAsUTC(updatedAt).toLocaleString(i18n.language, {
            dateStyle: "medium",
            timeStyle: "short",
          })}
        </time>
        <button
          type="button"
          onClick={onDelete}
          className="rounded px-1.5 py-1 text-danger hover:bg-danger/15"
        >
          {t(I18nKey.COMMON$DELETE_CONVERSATION)}
        </button>
      </div>
    </details>
  );
}
