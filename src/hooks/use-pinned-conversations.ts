import React from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { PINNED_TAG_KEY } from "#/api/agent-server-adapter";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { usePinnedConversationsStore } from "#/stores/pinned-conversations-store";

const EMPTY_PINNED_IDS: readonly string[] = [];

export interface UsePinnedConversationsResult {
  pinnedIds: readonly string[];
  togglePin: (conversationId: string) => void;
  unpinConversation: (conversationId: string) => void;
}

/** Tag values are ISO-8601 instants, so a descending sort is newest pin first. */
function derivePinnedIdsFromTags(
  conversations: readonly AppConversation[],
): readonly string[] {
  return conversations
    .flatMap((conversation) => {
      const pinnedAt = conversation.tags?.[PINNED_TAG_KEY];
      return typeof pinnedAt === "string" && pinnedAt.length > 0
        ? [{ id: conversation.id, pinnedAt }]
        : [];
    })
    .sort((a, b) => b.pinnedAt.localeCompare(a.pinnedAt))
    .map((entry) => entry.id);
}

/**
 * Pin state derived from the `pinned` tag on `conversations` (the list the
 * caller already renders), so pins follow the user across devices. Cloud
 * backends stay on the local store — the app-server has no `tags` yet.
 */
export function usePinnedConversations(
  conversations: readonly AppConversation[],
): UsePinnedConversationsResult {
  const { backend } = useActiveBackend();
  const backendId = backend.id;
  const supportsTags = backend.kind !== "cloud";

  const queryClient = useQueryClient();

  const localPinnedIds = usePinnedConversationsStore(
    (state) => state.pinsByBackendId[backendId] ?? EMPTY_PINNED_IDS,
  );
  const localTogglePin = usePinnedConversationsStore(
    (state) => state.togglePin,
  );
  const localUnpin = usePinnedConversationsStore(
    (state) => state.unpinConversation,
  );

  const { mutate: writePinnedTag } = useMutation({
    mutationFn: (variables: {
      conversationId: string;
      pinnedAt: string | null;
    }) =>
      AgentServerConversationService.mergeConversationTags(
        variables.conversationId,
        { [PINNED_TAG_KEY]: variables.pinnedAt },
      ),
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["user", "conversations"] });
    },
  });

  const taggedPinnedIds = React.useMemo(
    () =>
      supportsTags ? derivePinnedIdsFromTags(conversations) : EMPTY_PINNED_IDS,
    [conversations, supportsTags],
  );

  // Replays pins made before this migration. Only conversations on the loaded
  // pages can be replayed, so a local pin is dropped once its write succeeds —
  // never before, and never for one pagination has not reached yet. Stamps hang
  // off a fixed anchor so a pin replayed from a later page keeps its old rank.
  const attempted = React.useRef<Set<string>>(new Set());
  const anchor = React.useRef<number | null>(null);
  React.useEffect(() => {
    if (!supportsTags || localPinnedIds.length === 0) return;

    const base = (anchor.current ??= Date.now());
    const known = new Set(conversations.map((conversation) => conversation.id));
    const tagged = new Set(taggedPinnedIds);

    localPinnedIds.forEach((conversationId, rank) => {
      const key = `${backendId}:${conversationId}`;
      if (!known.has(conversationId) || attempted.current.has(key)) return;
      attempted.current.add(key);
      if (tagged.has(conversationId)) {
        localUnpin(backendId, conversationId);
        return;
      }
      writePinnedTag(
        { conversationId, pinnedAt: new Date(base - rank).toISOString() },
        { onSuccess: () => localUnpin(backendId, conversationId) },
      );
    });
  }, [
    backendId,
    conversations,
    localPinnedIds,
    localUnpin,
    supportsTags,
    taggedPinnedIds,
    writePinnedTag,
  ]);

  const togglePin = React.useCallback(
    (conversationId: string) => {
      if (!supportsTags) {
        localTogglePin(backendId, conversationId);
        return;
      }
      writePinnedTag({
        conversationId,
        pinnedAt: taggedPinnedIds.includes(conversationId)
          ? null
          : new Date().toISOString(),
      });
    },
    [backendId, localTogglePin, supportsTags, taggedPinnedIds, writePinnedTag],
  );

  const unpinConversation = React.useCallback(
    (conversationId: string) => {
      if (!supportsTags) {
        localUnpin(backendId, conversationId);
        return;
      }
      if (!taggedPinnedIds.includes(conversationId)) return;
      writePinnedTag({ conversationId, pinnedAt: null });
    },
    [backendId, localUnpin, supportsTags, taggedPinnedIds, writePinnedTag],
  );

  return {
    pinnedIds: supportsTags ? taggedPinnedIds : localPinnedIds,
    togglePin,
    unpinConversation,
  };
}
