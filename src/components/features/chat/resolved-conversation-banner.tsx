import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { CircleCheck } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { I18nKey } from "#/i18n/declaration";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useUpdateConversationTagFlags } from "#/hooks/mutation/use-update-conversation-tag-flags";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useEventStore } from "#/stores/use-event-store";
import { isUserMessageEvent } from "#/types/agent-server/type-guards";
import type { OHEvent } from "#/stores/use-event-store";
import {
  RESOLVED_CONVERSATION_TAG_KEY,
  getConversationResolution,
} from "#/utils/conversation-tag-flags";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

const COMPOSER_INPUT_SELECTOR = "[contenteditable='true']";

function latestUserMessageId(events: OHEvent[]): string | null {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const event = events[i];
    if (isUserMessageEvent(event)) return String(event.id);
  }
  return null;
}

/**
 * Notice above the composer for a project thread the Agent Server marked
 * resolved (`resolved=user|auto` tag). The composer stays usable.
 *
 * - Reopen re-reads the tags and PATCHes them without `resolved`, so tags
 *   the server wrote since the last poll are kept (the local PATCH replaces
 *   the whole map).
 * - Sending a message reopens the thread on the server. When the server
 *   echoes a new user message, the conversation is refetched so the banner
 *   follows the server rather than guessing.
 *
 * Local Agent Server only: Cloud tags come from the app server, which does
 * not carry the runtime's `resolved` tag.
 */
export function ResolvedConversationBanner() {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
  const { backend } = useActiveBackend();
  const { data: conversation } = useActiveConversation();
  const { mutate: updateTagFlags, isPending: isUpdating } =
    useUpdateConversationTagFlags();
  const [isReading, setIsReading] = useState(false);
  const bannerRef = useRef<HTMLDivElement>(null);

  const conversationId = conversation?.id ?? null;
  const resolution =
    backend.kind === "local"
      ? getConversationResolution(conversation?.tags)
      : null;

  const userMessageId = useEventStore((state) =>
    latestUserMessageId(state.events),
  );
  const seenUserMessageId = useRef(userMessageId);
  useEffect(() => {
    if (userMessageId === seenUserMessageId.current) return;
    seenUserMessageId.current = userMessageId;
    if (resolution && conversationId) {
      queryClient.invalidateQueries({
        queryKey: ["user", "conversation", conversationId],
      });
      // The sidebar list shows the tag too.
      queryClient.invalidateQueries({ queryKey: ["user", "conversations"] });
    }
  }, [userMessageId, resolution, conversationId, queryClient]);

  if (!conversation || !resolution) {
    return null;
  }

  const isPending = isReading || isUpdating;

  const handleReopen = async () => {
    if (isPending) return;
    // The banner unmounts once reopened; keep keyboard focus in the chat.
    const composer =
      bannerRef.current?.parentElement?.querySelector<HTMLElement>(
        COMPOSER_INPUT_SELECTOR,
      ) ?? null;

    setIsReading(true);
    let currentTags = conversation.tags;
    try {
      const [fresh] =
        await AgentServerConversationService.batchGetAppConversations([
          conversation.id,
        ]);
      if (fresh) {
        currentTags = fresh.tags;
        queryClient.setQueriesData<AppConversation | null | undefined>(
          { queryKey: ["user", "conversation", conversation.id] },
          (old) => (old ? { ...old, tags: fresh.tags } : old),
        );
      }
    } catch {
      // Fall back to the cached tags; the PATCH reports any real failure.
    } finally {
      setIsReading(false);
    }

    updateTagFlags(
      {
        conversationId: conversation.id,
        currentTags,
        changes: { [RESOLVED_CONVERSATION_TAG_KEY]: false },
      },
      {
        onError: () =>
          displayErrorToast(t(I18nKey.CHAT_INTERFACE$RESOLVED_REOPEN_FAILED)),
      },
    );
    composer?.focus();
  };

  return (
    <div
      ref={bannerRef}
      className="flex w-full items-start gap-2 rounded-lg border border-border-subtle bg-surface-raised px-3 py-2 text-foreground"
      data-testid="resolved-conversation-banner"
    >
      <CircleCheck className="mt-0.5 size-4 shrink-0 text-accent" aria-hidden />
      <div role="status" className="min-w-0 flex-1">
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
