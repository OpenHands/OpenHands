import React from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { PINNED_TAG_KEY } from "#/api/agent-server-adapter";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { usePinnedConversationsStore } from "#/stores/pinned-conversations-store";

const EMPTY_PINNED_IDS: readonly string[] = [];

export interface UsePinnedConversationsResult {
  /** Pinned conversation ids, most recently pinned first. */
  pinnedIds: readonly string[];
  togglePin: (conversationId: string) => void;
  unpinConversation: (conversationId: string) => void;
}

/**
 * Reads the `pinned` tag off the conversations the sidebar already fetched,
 * newest pin first (tag values are ISO-8601 instants, which sort
 * lexicographically). Conversations the backend no longer returns simply drop
 * out — deleting a conversation takes its tags with it, so there is nothing to
 * prune.
 */
function derivePinnedIdsFromTags(
  conversations: readonly AppConversation[],
): readonly string[] {
  const pinned: Array<{ id: string; pinnedAt: string }> = [];
  for (const conversation of conversations) {
    const pinnedAt = conversation.tags?.[PINNED_TAG_KEY];
    if (typeof pinnedAt === "string" && pinnedAt.length > 0) {
      pinned.push({ id: conversation.id, pinnedAt });
    }
  }
  return pinned
    .sort((a, b) => b.pinnedAt.localeCompare(a.pinnedAt))
    .map((entry) => entry.id);
}

/**
 * Pin state for the conversation panel.
 *
 * Agent-server backends keep pins in the conversation's server-side `pinned`
 * tag, so they follow the user across devices and browsers. The cloud
 * app-server does not round-trip `tags` yet, so cloud backends stay on the
 * browser-local store — see OSS-10028 / the linked cloud follow-up.
 *
 * Callers pass the conversations they already render; no extra fetch happens.
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

  // One-shot replay of pins made before this migration: stamp the tag on the
  // conversations that still exist, then drop the local entries so the server
  // stays the single source of truth. Runs per backend, and only once the
  // conversation list has actually loaded (an empty list would look like
  // "none of these still exist" and silently discard every pin).
  const migratedBackendIds = React.useRef<Set<string>>(new Set());
  React.useEffect(() => {
    if (!supportsTags) return;
    if (localPinnedIds.length === 0) return;
    if (conversations.length === 0) return;
    if (migratedBackendIds.current.has(backendId)) return;
    migratedBackendIds.current.add(backendId);

    const known = new Set(conversations.map((conversation) => conversation.id));
    const alreadyTagged = new Set(taggedPinnedIds);
    // Oldest local pin first, so replayed timestamps preserve the original
    // "most recently pinned first" order.
    const replayed = [...localPinnedIds].reverse();
    replayed.forEach((conversationId, index) => {
      if (known.has(conversationId) && !alreadyTagged.has(conversationId)) {
        writePinnedTag({
          conversationId,
          pinnedAt: new Date(Date.now() + index).toISOString(),
        });
      }
    });
    localPinnedIds.forEach((conversationId) =>
      localUnpin(backendId, conversationId),
    );
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
      const isPinned = taggedPinnedIds.includes(conversationId);
      writePinnedTag({
        conversationId,
        pinnedAt: isPinned ? null : new Date().toISOString(),
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
