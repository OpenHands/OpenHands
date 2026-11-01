import { useCallback, useMemo, useSyncExternalStore } from "react";
import {
  hashKey,
  useQueryClient,
  type QueryKey,
  type InfiniteData,
} from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import type {
  AppConversation,
  AppConversationPage,
  AppConversationStartTask,
} from "#/api/conversation-service/agent-server-conversation-service.types";
import { isPlaceholderConversationTitle } from "#/utils/conversation-title";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useConversationLaunchStore } from "#/stores/conversation-launch-store";
import { I18nKey } from "#/i18n/declaration";
import { useOptionalConversationId } from "./use-conversation-id";
import {
  CONVERSATION_QUERY_KEYS,
  DEFAULT_CONVERSATION_PAGE_SIZE,
} from "./query/query-keys";

function useCachedQuery<T>(queryKey: QueryKey) {
  const client = useQueryClient();
  const queryHash = hashKey(queryKey);
  const subscribe = useCallback(
    (onChange: () => void) =>
      client.getQueryCache().subscribe((event) => {
        if (event.query.queryHash === queryHash) onChange();
      }),
    [client, queryHash],
  );
  const getSnapshot = () => client.getQueryState<T>(queryKey);
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}

export function useConversationDisplayTitle() {
  const queryClient = useQueryClient();
  const { t } = useTranslation("openhands");
  const { backend, orgId } = useActiveBackend();
  const { conversationId } = useOptionalConversationId();
  const launches = useConversationLaunchStore((state) => state.launches);
  // Observe existing requests so the header and sidebar change titles together.
  const list = useCachedQuery<InfiniteData<AppConversationPage>>(
    CONVERSATION_QUERY_KEYS.paginated(
      DEFAULT_CONVERSATION_PAGE_SIZE,
      backend.id,
      orgId,
    ),
  );
  const listedById = useMemo(
    () =>
      new Map(
        list?.data?.pages
          .flatMap((page) => page.items)
          .map((entry) => [entry.id, entry]),
      ),
    [list?.data],
  );
  const detail = useCachedQuery<AppConversation | null>(
    CONVERSATION_QUERY_KEYS.detail(conversationId ?? null, backend.id, orgId),
  );
  const taskId = conversationId?.startsWith("task-")
    ? conversationId.slice(5)
    : null;
  const task = useCachedQuery<AppConversationStartTask | null>(
    CONVERSATION_QUERY_KEYS.startTask(taskId),
  );

  return useCallback(
    (id: string | null, fallback?: string | null) => {
      const launch = Object.values(launches).find(
        (entry) =>
          entry.backendId === backend.id &&
          entry.orgId === orgId &&
          (entry.id === id ||
            entry.targetId === id ||
            entry.conversationId === id),
      );
      const activeTask = id === `task-${taskId}` ? task?.data : undefined;
      const realId =
        launch?.conversationId ?? activeTask?.app_conversation_id ?? id;
      const listed = realId ? listedById.get(realId) : undefined;
      const cachedDetail = queryClient.getQueryState<AppConversation | null>(
        CONVERSATION_QUERY_KEYS.detail(realId, backend.id, orgId),
      );
      const active = cachedDetail?.data;
      const candidates = [
        { conversation: active, fetchedAt: cachedDetail?.dataUpdatedAt ?? 0 },
        { conversation: listed, fetchedAt: list?.dataUpdatedAt ?? 0 },
      ].filter(
        ({ conversation }) =>
          !isPlaceholderConversationTitle(conversation?.title, realId ?? ""),
      );
      candidates.sort(
        (a, b) =>
          Date.parse(b.conversation!.updated_at) -
            Date.parse(a.conversation!.updated_at) || b.fetchedAt - a.fetchedAt,
      );
      const serverTitle =
        candidates[0]?.conversation?.title ||
        (!isPlaceholderConversationTitle(fallback, realId ?? "")
          ? fallback
          : undefined);
      const prompt =
        launch?.message ||
        activeTask?.request.initial_message?.content
          .filter((content) => content.type === "text")
          .map((content) => content.text)
          .join(" ");
      return (
        serverTitle?.trim() ||
        activeTask?.request.title?.trim() ||
        prompt?.trim().replace(/\s+/g, " ") ||
        fallback?.trim() ||
        t(I18nKey.SIDEBAR$NEW_CHAT)
      );
    },
    [
      queryClient,
      launches,
      backend.id,
      orgId,
      taskId,
      task?.data,
      listedById,
      list?.dataUpdatedAt,
      detail?.data,
      detail?.dataUpdatedAt,
      t,
    ],
  );
}
