import React from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { I18nKey } from "#/i18n/declaration";
import { usePinnedConversationsStore } from "#/stores/pinned-conversations-store";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { resolvePinnedConversations } from "#/components/features/conversation-panel/conversation-panel-list-helpers";
import { useIsAuthed } from "./query/use-is-authed";

const EMPTY_IDS: readonly string[] = [];
const EMPTY_CONVERSATIONS: readonly AppConversation[] = [];

/** Legacy pins being copied to the server, so mounts don't start duplicates. */
const migratingScopes = new Set<string>();

interface UseConversationPinsInput {
  /** Every conversation loaded from the backend, archived ones included. */
  loadedConversations: readonly AppConversation[];
  /** The loaded conversations the user can currently see. */
  visibleConversations: readonly AppConversation[];
  /** Pinned conversations to leave out of the pinned section. */
  hiddenIds: ReadonlySet<string>;
}

/**
 * Pinned conversations, newest pin first.
 *
 * A Cloud server that supports pins keeps them on the conversation record
 * (`pinned_at`), so they follow the user across browsers. Pins made before that
 * lived in browser storage; they are copied to the server once and removed
 * locally. Local backends, and Cloud servers without pin support, keep using
 * browser storage.
 */
export function useConversationPins({
  loadedConversations,
  visibleConversations,
  hiddenIds,
}: UseConversationPinsInput) {
  const active = useActiveBackend();
  const { data: isAuthed } = useIsAuthed();
  const queryClient = useQueryClient();
  const { t } = useTranslation();
  const backendId = active.backend.id;
  const { orgId } = active;

  const serverPinsSupported =
    active.backend.kind === "cloud" &&
    loadedConversations.some(
      (conversation) => conversation.pinned_at !== undefined,
    );

  const localIds = usePinnedConversationsStore(
    (state) => state.pinsByBackendId[backendId] ?? EMPTY_IDS,
  );
  const toggleLocalPin = usePinnedConversationsStore(
    (state) => state.togglePin,
  );
  const unpinLocal = usePinnedConversationsStore(
    (state) => state.unpinConversation,
  );
  const pruneLocal = usePinnedConversationsStore(
    (state) => state.pruneMissingConversations,
  );

  const pinnedQueryKey = React.useMemo(
    () => ["user", "conversations", "pinned", backendId, orgId] as const,
    [backendId, orgId],
  );
  const pinnedQuery = useQuery({
    queryKey: pinnedQueryKey,
    queryFn: () => AgentServerConversationService.searchPinnedConversations(),
    enabled: serverPinsSupported && !!isAuthed && !!orgId,
    refetchInterval: 30_000,
    refetchIntervalInBackground: false,
  });

  const loadedById = React.useMemo(
    () =>
      new Map(
        loadedConversations.map((conversation) => [
          conversation.id,
          conversation,
        ]),
      ),
    [loadedConversations],
  );

  const { mutate: setPinned } = useMutation({
    mutationFn: (variables: { conversationId: string; pinned: boolean }) =>
      AgentServerConversationService.updateConversationPinned(
        variables.conversationId,
        variables.pinned,
      ),
    onMutate: async ({ conversationId, pinned }) => {
      await queryClient.cancelQueries({ queryKey: pinnedQueryKey });
      const previous =
        queryClient.getQueryData<AppConversation[]>(pinnedQueryKey);
      queryClient.setQueryData<AppConversation[]>(
        pinnedQueryKey,
        (current = []) => {
          if (!pinned) {
            return current.filter((c) => c.id !== conversationId);
          }
          const conversation = loadedById.get(conversationId);
          if (!conversation || current.some((c) => c.id === conversationId)) {
            return current;
          }
          return [
            { ...conversation, pinned_at: new Date().toISOString() },
            ...current,
          ];
        },
      );
      return { previous };
    },
    onError: (_error, _variables, context) => {
      queryClient.setQueryData(pinnedQueryKey, context?.previous);
      displayErrorToast(t(I18nKey.ERROR$GENERIC));
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: pinnedQueryKey });
    },
  });

  const serverPinned = pinnedQuery.data ?? EMPTY_CONVERSATIONS;
  const pinnedQueryFetched = pinnedQuery.isSuccess;
  const serverPinnedIds = React.useMemo(
    () => new Set(serverPinned.map((conversation) => conversation.id)),
    [serverPinned],
  );

  React.useEffect(() => {
    if (!serverPinsSupported || !pinnedQueryFetched || localIds.length === 0) {
      return;
    }
    const scope = `${backendId}:${orgId}`;
    if (migratingScopes.has(scope)) {
      return;
    }
    migratingScopes.add(scope);

    const migrate = async () => {
      const legacyIds = [...localIds];
      const found =
        await AgentServerConversationService.batchGetAppConversations(
          legacyIds,
        );
      // The local list is newest first; pin oldest first so the newest pin
      // ends up with the latest timestamp and keeps its place at the top.
      for (let index = legacyIds.length - 1; index >= 0; index -= 1) {
        const id = legacyIds[index];
        const conversation = found[index];
        if (conversation && !serverPinnedIds.has(id)) {
          await AgentServerConversationService.updateConversationPinned(
            id,
            true,
          );
        }
        unpinLocal(backendId, id);
      }
    };

    migrate()
      .catch(() => {
        // Keep the remaining local pins; the next load tries again.
      })
      .finally(() => {
        migratingScopes.delete(scope);
        queryClient.invalidateQueries({ queryKey: pinnedQueryKey });
      });
  }, [
    serverPinsSupported,
    pinnedQueryFetched,
    localIds,
    serverPinnedIds,
    backendId,
    orgId,
    pinnedQueryKey,
    queryClient,
    unpinLocal,
  ]);

  const pinnedConversations = React.useMemo(() => {
    if (!serverPinsSupported) {
      return resolvePinnedConversations(localIds, visibleConversations);
    }
    return serverPinned
      .filter((conversation) => !hiddenIds.has(conversation.id))
      .map((conversation) => loadedById.get(conversation.id) ?? conversation);
  }, [
    serverPinsSupported,
    localIds,
    visibleConversations,
    serverPinned,
    hiddenIds,
    loadedById,
  ]);

  const pinnedIds = React.useMemo(
    () =>
      serverPinsSupported
        ? pinnedConversations.map((conversation) => conversation.id)
        : localIds,
    [serverPinsSupported, pinnedConversations, localIds],
  );

  const togglePin = React.useCallback(
    (conversationId: string) => {
      if (!serverPinsSupported) {
        toggleLocalPin(backendId, conversationId);
        return;
      }
      setPinned({
        conversationId,
        pinned: !serverPinnedIds.has(conversationId),
      });
    },
    [
      serverPinsSupported,
      toggleLocalPin,
      backendId,
      setPinned,
      serverPinnedIds,
    ],
  );

  const unpin = React.useCallback(
    (conversationId: string) => {
      if (!serverPinsSupported) {
        unpinLocal(backendId, conversationId);
        return;
      }
      if (serverPinnedIds.has(conversationId)) {
        setPinned({ conversationId, pinned: false });
      }
    },
    [serverPinsSupported, unpinLocal, backendId, setPinned, serverPinnedIds],
  );

  const pruneMissing = React.useCallback(
    (loadedIds: readonly string[]) => {
      if (!serverPinsSupported) {
        pruneLocal(backendId, loadedIds);
      }
    },
    [serverPinsSupported, pruneLocal, backendId],
  );

  return { pinnedIds, pinnedConversations, togglePin, unpin, pruneMissing };
}
