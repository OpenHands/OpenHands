import type { ReactNode } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import type { ConversationNode } from "./conversation-panel-list-helpers";

/** Per-layer indent for the parent-child tree ("2-space tab"). */
export const CONVERSATION_TREE_INDENT_PX = 16;

interface ConversationTreeProps {
  nodes: ConversationNode[];
  collapsedIds: ReadonlySet<string>;
  onToggleCollapsed: (conversationId: string) => void;
  renderConversationCard: (conversation: AppConversation) => ReactNode;
}

interface NodeViewProps {
  node: ConversationNode;
  collapsedIds: ReadonlySet<string>;
  onToggleCollapsed: (conversationId: string) => void;
  renderConversationCard: (conversation: AppConversation) => ReactNode;
}

function NodeView({
  node,
  collapsedIds,
  onToggleCollapsed,
  renderConversationCard,
}: NodeViewProps) {
  const { t } = useTranslation("openhands");
  const collapsed = collapsedIds.has(node.conversation.id);
  const hasChildren = node.hasChildren;

  return (
    <div className="flex flex-col">
      <div
        data-testid={`conversation-tree-row-${node.conversation.id}`}
        className="flex items-start gap-1"
        style={{ paddingLeft: node.depth * CONVERSATION_TREE_INDENT_PX }}
      >
        <button
          type="button"
          data-testid={`conversation-tree-toggle-${node.conversation.id}`}
          aria-label={
            hasChildren
              ? collapsed
                ? t(I18nKey.CONVERSATION_TREE$EXPAND)
                : t(I18nKey.CONVERSATION_TREE$COLLAPSE)
              : t(I18nKey.CONVERSATION_TREE$NO_CHILDREN)
          }
          aria-expanded={hasChildren ? !collapsed : undefined}
          disabled={!hasChildren}
          onClick={() => onToggleCollapsed(node.conversation.id)}
          className={cn(
            "mt-1.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-sm",
            "text-[var(--oh-muted)] hover:text-white",
            !hasChildren && "invisible",
          )}
        >
          {hasChildren ? (
            collapsed ? (
              <ChevronRight
                width={14}
                height={14}
                strokeWidth={2}
                aria-hidden
              />
            ) : (
              <ChevronDown width={14} height={14} strokeWidth={2} aria-hidden />
            )
          ) : null}
        </button>
        <div className="min-w-0 flex-1">
          {renderConversationCard(node.conversation)}
        </div>
      </div>

      {hasChildren && !collapsed ? (
        <div
          role="group"
          aria-label={node.conversation.title ?? node.conversation.id}
        >
          {node.children.map((child) => (
            <NodeView
              key={child.conversation.id}
              node={child}
              collapsedIds={collapsedIds}
              onToggleCollapsed={onToggleCollapsed}
              renderConversationCard={renderConversationCard}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}

/**
 * Recursive parent-child conversation list. Each layer indents by a fixed
 * "2-space tab" per depth, any layer with children is collapsible/expandable,
 * and every rendered node is the caller's normal conversation row (so a click
 * opens that conversation as usual — nothing about navigation changes).
 */
export function ConversationTree({
  nodes,
  collapsedIds,
  onToggleCollapsed,
  renderConversationCard,
}: ConversationTreeProps) {
  if (nodes.length === 0) {
    return null;
  }
  return (
    <div className="space-y-0.5">
      {nodes.map((node) => (
        <NodeView
          key={node.conversation.id}
          node={node}
          collapsedIds={collapsedIds}
          onToggleCollapsed={onToggleCollapsed}
          renderConversationCard={renderConversationCard}
        />
      ))}
    </div>
  );
}
