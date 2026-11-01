import {
  useCallback,
  useMemo,
  useSyncExternalStore,
  useEffect,
  useReducer,
} from "react";
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

const OPTIMISTIC_TITLE_TIMEOUT_MS = 8000;

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

export function useConversationDisplayTitleState() {
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

  const [clock, tick] = useReducer((value: number) => value + 1, 0);
  const deadlines = [
    task?.data?.title_generation,
    detail?.data?.title_generation,
    ...[...listedById.values()].map((entry) => entry.title_generation),
  ].flatMap((generation) =>
    generation?.status === "pending"
      ? [Date.parse(generation.deadline_at)]
      : [],
  );
  for (const launch of Object.values(launches)) {
    if (launch.backendId !== backend.id || launch.orgId !== orgId) continue;
    if (launch.startedAt)
      deadlines.push(launch.startedAt + OPTIMISTIC_TITLE_TIMEOUT_MS);
    if (launch.targetId?.startsWith("task-")) {
      const cached = queryClient.getQueryData<AppConversationStartTask>(
        CONVERSATION_QUERY_KEYS.startTask(launch.targetId.slice(5)),
      );
      if (cached?.title_generation?.status === "pending")
        deadlines.push(Date.parse(cached.title_generation.deadline_at));
    }
  }
  const nextDeadline = Math.min(
    ...deadlines.filter((deadline) => deadline > Date.now()),
  );
  useEffect(() => {
    if (!Number.isFinite(nextDeadline)) return;
    const timer = setTimeout(tick, Math.max(0, nextDeadline - Date.now()) + 1);
    return () => clearTimeout(timer);
  }, [nextDeadline, clock]);

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
      const foreignTask =
        !launch &&
        Object.values(launches).some((entry) => entry.targetId === id);
      const matchingTaskId = launch?.targetId?.startsWith("task-")
        ? launch.targetId.slice(5)
        : id?.startsWith("task-")
          ? id.slice(5)
          : null;
      const activeTask =
        !foreignTask && matchingTaskId
          ? queryClient.getQueryData<AppConversationStartTask>(
              CONVERSATION_QUERY_KEYS.startTask(matchingTaskId),
            )
          : undefined;
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
      const conversation = candidates[0]?.conversation;
      const generation =
        conversation?.title_generation?.status === "pending"
          ? (activeTask?.title_generation ?? conversation.title_generation)
          : conversation
            ? conversation.title_generation
            : activeTask?.title_generation;
      const optimisticPending =
        backend.kind === "cloud" &&
        launch &&
        !activeTask &&
        !active &&
        !listed &&
        !launch.error &&
        Date.now() < (launch.startedAt ?? 0) + OPTIMISTIC_TITLE_TIMEOUT_MS;
      const pending = !!(
        (generation?.status === "pending" &&
          Date.now() < Date.parse(generation.deadline_at)) ||
        optimisticPending
      );
      const title = pending
        ? t(I18nKey.SIDEBAR$NEW_CHAT)
        : generation?.title?.trim() ||
          serverTitle?.trim() ||
          activeTask?.request.title?.trim() ||
          prompt?.trim().replace(/\s+/g, " ") ||
          fallback?.trim() ||
          t(I18nKey.SIDEBAR$NEW_CHAT);
      return { title, pending };
    },
    [
      queryClient,
      clock,
      launches,
      backend.id,
      backend.kind,
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

export function useConversationDisplayTitle() {
  const getState = useConversationDisplayTitleState();
  return useCallback(
    (id: string | null, fallback?: string | null) =>
      getState(id, fallback).title,
    [getState],
  );
}
