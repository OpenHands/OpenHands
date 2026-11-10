import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { CircleCheck } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { I18nKey } from "#/i18n/declaration";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useEventStore } from "#/stores/use-event-store";
import { isUserMessageEvent } from "#/types/agent-server/type-guards";
import type { OHEvent } from "#/stores/use-event-store";
import {
  RESOLVED_CONVERSATION_TAG_KEY,
  RESOLVE_NOTE_TAG_KEY,
  getConversationResolution,
  parseAutoResolveNote,
} from "#/utils/conversation-tag-flags";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import ProjectsService from "#/api/projects-service/projects-service.api";
import { projectsQueryRoot } from "#/hooks/query/use-projects";

/** Agent Server tags a project's member with the project's id. */
const PROJECT_TAG_KEY = "project";

const COMPOSER_INPUT_SELECTOR = "[contenteditable='true']";

/**
 * Matches this conversation's cached detail whichever id form keyed it: the
 * route uses the plain hex id, the server's `id` has dashes.
 */
const conversationDetailFilter = (id: string) => {
  const key = id.replace(/-/g, "").toLowerCase();
  return {
    queryKey: ["user", "conversation"],
    predicate: (query: { queryKey: readonly unknown[] }) =>
      String(query.queryKey[2] ?? "")
        .replace(/-/g, "")
        .toLowerCase() === key,
  };
};

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
 * - Reopen uses the project's reopen route; the server owns the tags.
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
      queryClient.invalidateQueries(conversationDetailFilter(conversationId));
      // The sidebar list shows the tag too.
      queryClient.invalidateQueries({ queryKey: ["user", "conversations"] });
    }
  }, [userMessageId, resolution, conversationId, queryClient]);

  // Resolving is a project's board state: only members carry it.
  const projectId = conversation?.tags?.[PROJECT_TAG_KEY];
  if (!conversation || !resolution || !projectId) {
    return null;
  }

  const isPending = isReading;
  const autoNote =
    resolution === "auto"
      ? parseAutoResolveNote(conversation.tags?.[RESOLVE_NOTE_TAG_KEY])
      : null;

  const handleReopen = async () => {
    if (isPending) return;
    // The banner unmounts once reopened; keep keyboard focus in the chat.
    const composer =
      bannerRef.current?.parentElement?.querySelector<HTMLElement>(
        COMPOSER_INPUT_SELECTOR,
      ) ?? null;

    setIsReading(true);
    try {
      await ProjectsService.setConversationResolved(
        projectId,
        conversation.id,
        false,
      );
    } catch {
      setIsReading(false);
      displayErrorToast(t(I18nKey.CHAT_INTERFACE$RESOLVED_REOPEN_FAILED));
      return;
    }
    // Hide at once; the refetch below confirms from the server.
    queryClient.setQueriesData<AppConversation | null | undefined>(
      conversationDetailFilter(conversation.id),
      (old) => {
        if (!old?.tags) return old;
        const tags = { ...old.tags };
        delete tags[RESOLVED_CONVERSATION_TAG_KEY];
        delete tags[RESOLVE_NOTE_TAG_KEY];
        return { ...old, tags };
      },
    );
    setIsReading(false);
    composer?.focus();
    await Promise.all([
      queryClient.invalidateQueries(conversationDetailFilter(conversation.id)),
      queryClient.invalidateQueries({ queryKey: ["user", "conversations"] }),
      queryClient.invalidateQueries({ queryKey: projectsQueryRoot(backend) }),
    ]);
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
        {autoNote ? (
          <p
            className="text-xs text-muted mt-0.5"
            data-testid="resolved-conversation-auto-note"
          >
            {t(
              autoNote.source === "critic"
                ? I18nKey.PROJECTS$AUTO_RESOLVED_CRITIC
                : I18nKey.PROJECTS$AUTO_RESOLVED,
              { percent: autoNote.percent },
            )}
          </p>
        ) : null}
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
