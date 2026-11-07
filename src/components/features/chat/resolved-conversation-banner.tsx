import { useTranslation } from "react-i18next";
import { CircleCheck } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useUpdateConversationTagFlags } from "#/hooks/mutation/use-update-conversation-tag-flags";
import {
  RESOLVED_CONVERSATION_TAG_KEY,
  getConversationResolution,
} from "#/utils/conversation-tag-flags";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

/**
 * Notice above the composer for a conversation that a project marked
 * resolved (`resolved=user|auto` tag). The composer stays usable: sending a
 * message reopens the thread on the server, and Reopen removes the tag
 * without sending anything.
 */
export function ResolvedConversationBanner() {
  const { t } = useTranslation("openhands");
  const { data: conversation } = useActiveConversation();
  const { mutate: updateTagFlags, isPending } = useUpdateConversationTagFlags();

  if (!conversation || !getConversationResolution(conversation.tags)) {
    return null;
  }

  const handleReopen = () => {
    if (isPending) return;
    updateTagFlags(
      {
        conversationId: conversation.id,
        currentTags: conversation.tags,
        changes: { [RESOLVED_CONVERSATION_TAG_KEY]: false },
      },
      {
        onError: () =>
          displayErrorToast(t(I18nKey.CHAT_INTERFACE$RESOLVED_REOPEN_FAILED)),
      },
    );
  };

  return (
    <div
      role="status"
      className="flex w-full items-start gap-2 rounded-lg border border-border-subtle bg-surface-raised px-3 py-2 text-foreground"
      data-testid="resolved-conversation-banner"
    >
      <CircleCheck className="mt-0.5 size-4 shrink-0 text-accent" aria-hidden />
      <div className="min-w-0 flex-1">
        <p className="text-sm text-foreground">
          {t(I18nKey.CHAT_INTERFACE$RESOLVED_TITLE)}
        </p>
        <p className="text-xs text-muted mt-0.5">
          {t(I18nKey.CHAT_INTERFACE$RESOLVED_DESCRIPTION)}
        </p>
      </div>
      <button
        type="button"
        onClick={handleReopen}
        disabled={isPending}
        className="shrink-0 cursor-pointer rounded-md border border-border px-2 py-1 text-xs font-normal text-foreground hover:bg-interactive-hover disabled:cursor-not-allowed disabled:opacity-50"
        data-testid="resolved-conversation-reopen"
      >
        {t(I18nKey.CHAT_INTERFACE$RESOLVED_REOPEN)}
      </button>
    </div>
  );
}
