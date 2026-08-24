import React from "react";
import { useNavigate } from "react-router";
import { useTranslation } from "react-i18next";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import {
  FACTORY_PLAN_WORKSTREAM_ID,
  FACTORY_RUN_ID_TAG_KEY,
  FACTORY_RUN_TAG_KEY,
  FACTORY_WORKSTREAM_ID_TAG_KEY,
} from "#/api/agent-server-adapter";
import { NavigationLink } from "#/components/shared/navigation-link";
import { ConversationRunGraph } from "#/components/features/conversation-run-graph/conversation-run-graph";
import { useConversationId } from "#/hooks/use-conversation-id";
import { usePaginatedConversations } from "#/hooks/query/use-paginated-conversations";
import { useSubConversations } from "#/hooks/query/use-sub-conversations";
import { useUserConversation } from "#/hooks/query/use-user-conversation";
import { I18nKey } from "#/i18n/declaration";

/**
 * Window of recent conversations used to discover factory workstream children.
 * The agent-server has no tag filter on `/api/conversations/search`, and
 * factory workstreams cannot be linked via `sub_conversation_ids` (the server
 * rejects a child whose workspace differs from its parent's), so siblings of a
 * factory plan are found by matching the shared `runid` tag among the most
 * recent conversations — factory runs are minutes old, so their workstreams
 * are always inside this window.
 */
const RUN_GRAPH_DISCOVERY_LIMIT = 100;

/**
 * Merge a parent conversation's children from every source: native
 * `sub_conversation_ids` plus factory workstreams sharing the parent's
 * `runid` tag (deduped, parent excluded).
 */
export function collectRunGraphChildren(
  parent: AppConversation | null | undefined,
  nativeChildren: (AppConversation | null)[] | undefined,
  loadedConversations: readonly AppConversation[],
): AppConversation[] {
  const byId = new Map<string, AppConversation>();
  for (const child of nativeChildren ?? []) {
    if (child) {
      byId.set(child.id, child);
    }
  }
  const runId = parent?.tags?.[FACTORY_RUN_ID_TAG_KEY];
  if (parent && runId) {
    for (const conversation of loadedConversations) {
      if (conversation.id === parent.id) {
        continue;
      }
      const tags = conversation.tags;
      if (
        tags?.[FACTORY_RUN_TAG_KEY] === "1" &&
        tags[FACTORY_RUN_ID_TAG_KEY] === runId &&
        tags[FACTORY_WORKSTREAM_ID_TAG_KEY] !== FACTORY_PLAN_WORKSTREAM_ID
      ) {
        byId.set(conversation.id, conversation);
      }
    }
  }
  return [...byId.values()];
}

export function ConversationGraphRoute() {
  const { t } = useTranslation("openhands");
  const { conversationId } = useConversationId();
  const navigate = useNavigate();

  const { data: parent, isFetched } = useUserConversation(conversationId);
  const { data: nativeChildren } = useSubConversations(
    parent?.sub_conversation_ids ?? [],
  );
  const { data: pages } = usePaginatedConversations(RUN_GRAPH_DISCOVERY_LIMIT);
  const loadedConversations = React.useMemo(
    () => (pages?.pages ?? []).flatMap((page) => page.items ?? []),
    [pages],
  );

  React.useEffect(() => {
    if (isFetched && !parent) {
      navigate("/conversations", { replace: true });
    }
  }, [isFetched, parent, navigate]);

  const children = React.useMemo(
    () => collectRunGraphChildren(parent, nativeChildren, loadedConversations),
    [parent, nativeChildren, loadedConversations],
  );

  return (
    <div
      data-testid="conversation-graph-route"
      className="flex h-full flex-col gap-4 overflow-hidden p-4"
    >
      <header className="flex items-center justify-between gap-3">
        <h1 className="m-0 text-base font-semibold text-[var(--foreground)]">
          {t(I18nKey.CONVERSATION_GRAPH$TITLE)}
        </h1>
        <NavigationLink
          to={`/conversations/${conversationId}`}
          data-testid="run-graph-open-parent"
          className="text-sm text-[var(--oh-link)] hover:underline"
        >
          {t(I18nKey.CONVERSATION_GRAPH$OPEN_CONVERSATION)}
        </NavigationLink>
      </header>

      {!parent ? (
        <div className="flex flex-1 items-center justify-center text-sm text-[var(--oh-muted)]">
          …
        </div>
      ) : children.length === 0 ? (
        <div
          data-testid="run-graph-no-children"
          className="flex flex-1 items-center justify-center text-sm text-[var(--oh-muted)]"
        >
          {t(I18nKey.CONVERSATION_GRAPH$NO_CHILDREN)}
        </div>
      ) : (
        <ConversationRunGraph parent={parent} childConversations={children} />
      )}
    </div>
  );
}

export default ConversationGraphRoute;
