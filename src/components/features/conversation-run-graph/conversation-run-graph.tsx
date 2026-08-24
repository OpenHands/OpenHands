import React from "react";
import { useTranslation } from "react-i18next";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { FACTORY_WORKSTREAM_ID_TAG_KEY } from "#/api/agent-server-adapter";
import { NavigationLink } from "#/components/shared/navigation-link";
import { I18nKey } from "#/i18n/declaration";
import { ExecutionStatus } from "#/types/agent-server/core";
import { cn } from "#/utils/utils";
import {
  computeRunGraphLayout,
  type RunGraphNodePlacement,
} from "./conversation-run-graph-layout";

const STATUS_I18N_KEYS: Record<string, I18nKey> = {
  [ExecutionStatus.IDLE]: I18nKey.CONVERSATION_GRAPH$STATUS_IDLE,
  [ExecutionStatus.RUNNING]: I18nKey.CONVERSATION_GRAPH$STATUS_RUNNING,
  [ExecutionStatus.PAUSED]: I18nKey.CONVERSATION_GRAPH$STATUS_PAUSED,
  [ExecutionStatus.WAITING_FOR_CONFIRMATION]:
    I18nKey.CONVERSATION_GRAPH$STATUS_WAITING,
  [ExecutionStatus.FINISHED]: I18nKey.CONVERSATION_GRAPH$STATUS_FINISHED,
  [ExecutionStatus.ERROR]: I18nKey.CONVERSATION_GRAPH$STATUS_ERROR,
  [ExecutionStatus.STUCK]: I18nKey.CONVERSATION_GRAPH$STATUS_STUCK,
};

function GraphNode({
  conversation,
  placement,
  kind,
}: {
  conversation: AppConversation;
  placement: RunGraphNodePlacement;
  kind: "parent" | "child";
}) {
  const { t } = useTranslation("openhands");
  const isParent = kind === "parent";
  const statusKey = conversation.execution_status
    ? STATUS_I18N_KEYS[conversation.execution_status]
    : undefined;
  const isFactoryWorkstream =
    !isParent && Boolean(conversation.tags?.[FACTORY_WORKSTREAM_ID_TAG_KEY]);

  return (
    <NavigationLink
      to={`/conversations/${conversation.id}`}
      data-testid={isParent ? "run-graph-parent-node" : "run-graph-child-node"}
      title={conversation.title ?? conversation.id}
      className={cn(
        "absolute flex flex-col gap-1 overflow-hidden rounded-lg border p-3",
        "bg-[var(--oh-surface)] text-left transition-colors",
        "hover:bg-[var(--oh-interactive-hover)]",
        isParent
          ? "border-[var(--oh-accent)]"
          : "border-[var(--oh-border-subtle)]",
      )}
      style={{
        left: placement.x,
        top: placement.y,
        width: placement.width,
        height: placement.height,
      }}
    >
      <span className="text-[10px] font-medium uppercase tracking-wide text-[var(--oh-muted)]">
        {isParent
          ? t(I18nKey.CONVERSATION_GRAPH$PARENT)
          : isFactoryWorkstream
            ? t(I18nKey.CONVERSATION_GRAPH$WORKSTREAM)
            : t(I18nKey.CONVERSATION_GRAPH$CHILD)}
      </span>
      <span className="min-w-0 truncate text-sm font-medium leading-4 text-[var(--foreground)]">
        {conversation.title?.trim() || t(I18nKey.CONVERSATION_GRAPH$UNTITLED)}
      </span>
      <span className="mt-auto text-xs text-[var(--oh-muted)]">
        {statusKey
          ? t(statusKey)
          : t(I18nKey.CONVERSATION_GRAPH$STATUS_UNKNOWN)}
      </span>
    </NavigationLink>
  );
}

/**
 * The run graph: the parent conversation on top and its child conversations
 * (native `sub_conversation_ids` and/or factory workstreams) in a row beneath,
 * connected by straight edges. Every node is the normal conversation link — a
 * click opens that conversation's usual single-conversation view.
 */
export function ConversationRunGraph({
  parent,
  childConversations,
}: {
  parent: AppConversation;
  childConversations: AppConversation[];
}) {
  const layout = computeRunGraphLayout(childConversations.length);

  return (
    <div
      data-testid="conversation-run-graph"
      className="relative overflow-auto"
      style={{ width: layout.width, height: layout.height }}
    >
      <svg
        className="pointer-events-none absolute inset-0"
        width={layout.width}
        height={layout.height}
        aria-hidden
      >
        {layout.edges.map((edge, index) => (
          <line
            key={index}
            x1={edge.x1}
            y1={edge.y1}
            x2={edge.x2}
            y2={edge.y2}
            stroke="var(--oh-border)"
            strokeWidth={1.5}
          />
        ))}
      </svg>
      <GraphNode
        conversation={parent}
        placement={layout.parent}
        kind="parent"
      />
      {childConversations.map((child, index) => (
        <GraphNode
          key={child.id}
          conversation={child}
          placement={layout.children[index]}
          kind="child"
        />
      ))}
    </div>
  );
}
