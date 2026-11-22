import React from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

interface ConversationSubConversationGroupProps {
  conversation: AppConversation;
  subConversations: AppConversation[];
  expanded: boolean;
  onToggle: () => void;
  renderParent: (conversation: AppConversation) => React.ReactNode;
  renderChild: (conversation: AppConversation) => React.ReactNode;
}

/**
 * Wraps a parent conversation card with its server-linked children. The
 * toggle is rendered outside the parent's `NavigationLink` so clicking a
 * child navigates to the child (nested anchors are invalid HTML).
 */
export function ConversationSubConversationGroup({
  conversation,
  subConversations,
  expanded,
  onToggle,
  renderParent,
  renderChild,
}: ConversationSubConversationGroupProps) {
  const { t } = useTranslation("openhands");
  const count = subConversations.length;
  const label = t(I18nKey.CONVERSATION_PANEL$SUB_CONVERSATIONS, { count });
  const regionId = `conversation-sub-conversations-region-${conversation.id}`;

  return (
    <div>
      {renderParent(conversation)}
      {count > 0 ? (
        <>
          <button
            type="button"
            data-testid={`conversation-sub-conversations-toggle-${conversation.id}`}
            aria-expanded={expanded}
            aria-controls={regionId}
            aria-label={label}
            onClick={onToggle}
            className={cn(
              "ml-3 flex items-center gap-1 rounded px-1 py-0.5 text-2xs text-muted",
              "transition-colors hover:bg-surface hover:text-contrast",
            )}
          >
            {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            <span>{label}</span>
          </button>
          {expanded ? (
            <div
              id={regionId}
              role="group"
              aria-label={label}
              data-testid={`conversation-sub-conversations-${conversation.id}`}
              className="ml-4 border-l border-border pl-1"
            >
              {subConversations.map((child) => renderChild(child))}
            </div>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
