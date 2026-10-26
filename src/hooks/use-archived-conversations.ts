import React from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ARCHIVED_TAG_KEY } from "#/api/agent-server-adapter";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { patchConversationInCache } from "#/hooks/mutation/conversation-mutation-utils";
import { useArchivedConversationsStore } from "#/stores/archived-conversations-store";

const EMPTY_ARCHIVED_IDS: readonly string[] = [];

export interface UseArchivedConversationsResult {
  archivedIdSet: ReadonlySet<string>;
  archiveConversation: (conversationId: string) => void;
  unarchiveConversation: (conversationId: string) => void;
  /** Drops any archive state left behind by a deleted conversation. */
  forgetConversation: (conversationId: string) => void;
}

function hasArchivedTag(conversation: AppConversation | undefined): boolean {
  const archivedAt = conversation?.tags?.[ARCHIVED_TAG_KEY];
  return typeof archivedAt === "string" && archivedAt.length > 0;
}

function withArchivedTag(
  tags: Record<string, string> | null | undefined,
  archivedAt: string | null,
): Record<string, string> {
  const next = { ...(tags ?? {}) };
  if (archivedAt === null) {
    delete next[ARCHIVED_TAG_KEY];
  } else {
    next[ARCHIVED_TAG_KEY] = archivedAt;
  }
  return next;
}

/**
 * Archive state derived from the `archived` tag on `conversations` (the
 * unfiltered list the caller already loaded), so archives follow the user
 * across devices. Cloud backends stay on the local store because the
 * app-server does not round-trip `tags` yet.
 */
export function useArchivedConversations(
  conversations: readonly AppConversation[],
): UseArchivedConversationsResult {
  const { backend } = useActiveBackend();
  const backendId = backend.id;
  const supportsTags = backend.kind === "local";

  const queryClient = useQueryClient();

  const localArchivedIds = useArchivedConversationsStore(
    (state) => state.archivesByBackendId[backendId] ?? EMPTY_ARCHIVED_IDS,
  );
  const localArchive = useArchivedConversationsStore(
    (state) => state.archiveConversation,
  );
  const localRemove = useArchivedConversationsStore(
    (state) => state.removeArchivedConversation,
  );

  const conversationsById = React.useMemo(
    () =>
      new Map(
        conversations.map((conversation) => [conversation.id, conversation]),
      ),
    [conversations],
  );
  const conversationsByIdRef = React.useRef(conversationsById);
  conversationsByIdRef.current = conversationsById;

  const { mutate: writeArchivedTag } = useMutation({
    mutationFn: (variables: {
      conversationId: string;
      archivedAt: string | null;
    }) =>
      AgentServerConversationService.mergeConversationTags(
        variables.conversationId,
        { [ARCHIVED_TAG_KEY]: variables.archivedAt },
      ),
    onMutate: async ({ conversationId, archivedAt }) => {
      // Optimistic, so the row hides (or returns) on click rather than on the
      // next refetch. A failed write is undone by the refetch in onSettled.
      await queryClient.cancelQueries({ queryKey: ["user", "conversations"] });
      const current = conversationsByIdRef.current.get(conversationId);
      patchConversationInCache(queryClient, conversationId, {
        tags: withArchivedTag(current?.tags, archivedAt),
      });
    },
    onSuccess: (updated, { conversationId }) => {
      patchConversationInCache(queryClient, conversationId, {
        tags: updated.tags,
      });
      // The server now owns this conversation's archive state, so any
      // browser-local entry (a pre-migration archive) is redundant.
      localRemove(backendId, conversationId);
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["user", "conversations"] });
    },
  });

  const taggedArchivedIds = React.useMemo(
    () =>
      supportsTags
        ? conversations
            .filter((conversation) => hasArchivedTag(conversation))
            .map((conversation) => conversation.id)
        : EMPTY_ARCHIVED_IDS,
    [conversations, supportsTags],
  );

  // Local archives still hide their rows until they are replayed below, so an
  // archived conversation never flickers back into the list mid-migration.
  const archivedIdSet = React.useMemo(
    () => new Set([...taggedArchivedIds, ...localArchivedIds]),
    [localArchivedIds, taggedArchivedIds],
  );

  // Replays archives made before this migration. Only conversations on the
  // loaded pages can be replayed, so a local entry is dropped once its write
  // succeeds (see onSuccess) and never for one pagination has not reached yet.
  const attempted = React.useRef<Set<string>>(new Set());
  React.useEffect(() => {
    if (!supportsTags || localArchivedIds.length === 0) return;

    localArchivedIds.forEach((conversationId) => {
      const key = `${backendId}:${conversationId}`;
      const conversation = conversationsById.get(conversationId);
      if (!conversation || attempted.current.has(key)) return;
      attempted.current.add(key);
      if (hasArchivedTag(conversation)) {
        localRemove(backendId, conversationId);
        return;
      }
      writeArchivedTag({
        conversationId,
        archivedAt: new Date().toISOString(),
      });
    });
  }, [
    backendId,
    conversationsById,
    localArchivedIds,
    localRemove,
    supportsTags,
    writeArchivedTag,
  ]);

  const archiveConversation = React.useCallback(
    (conversationId: string) => {
      if (!supportsTags) {
        localArchive(backendId, conversationId);
        return;
      }
      if (hasArchivedTag(conversationsByIdRef.current.get(conversationId))) {
        return;
      }
      writeArchivedTag({
        conversationId,
        archivedAt: new Date().toISOString(),
      });
    },
    [backendId, localArchive, supportsTags, writeArchivedTag],
  );

  const unarchiveConversation = React.useCallback(
    (conversationId: string) => {
      localRemove(backendId, conversationId);
      if (
        !supportsTags ||
        !hasArchivedTag(conversationsByIdRef.current.get(conversationId))
      ) {
        return;
      }
      writeArchivedTag({ conversationId, archivedAt: null });
    },
    [backendId, localRemove, supportsTags, writeArchivedTag],
  );

  // A deleted conversation takes its server-side tags with it; only the
  // browser-local entry can outlive it.
  const forgetConversation = React.useCallback(
    (conversationId: string) => localRemove(backendId, conversationId),
    [backendId, localRemove],
  );

  return {
    archivedIdSet,
    archiveConversation,
    unarchiveConversation,
    forgetConversation,
  };
}
