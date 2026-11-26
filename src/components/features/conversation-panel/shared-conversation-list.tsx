import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useNavigation } from "#/context/navigation-context";
import { useBackendScopedPath } from "#/hooks/use-backend-scoped-path";
import { useAutomationPermissions } from "#/hooks/use-automation-permissions";
import { useSharedConversations } from "#/hooks/query/use-shared-conversations";
import { useUpdateConversationPublicFlag } from "#/hooks/mutation/use-update-conversation-public-flag";
import { NavigationLink } from "#/components/shared/navigation-link";
import { uniqueById } from "#/utils/unique-by-id";
import { ConversationCard } from "./conversation-card/conversation-card";
import { ConversationCardSkeleton } from "./conversation-card/conversation-card-skeleton";

interface SharedConversationListProps {
  onClose?: () => void;
}

/**
 * The conversations that members shared with the active org. Each one opens
 * through the conversation route, which shows an owner their own
 * conversation and sends anyone else to the read-only view. Only the
 * creator's own rows have a menu, which lets them stop sharing.
 */
export function SharedConversationList({
  onClose,
}: SharedConversationListProps) {
  const { t } = useTranslation("openhands");
  const { conversationId: currentConversationId } = useNavigation();
  const backendScopedPath = useBackendScopedPath();
  const { userId } = useAutomationPermissions();
  const { mutate: updatePublicFlag } = useUpdateConversationPublicFlag();
  const [openContextMenuId, setOpenContextMenuId] = React.useState<
    string | null
  >(null);
  const { data, isLoading, hasNextPage, isFetchingNextPage, fetchNextPage } =
    useSharedConversations();

  // A row can come back on a later page when the list shifts between loads.
  const conversations = uniqueById(
    data?.pages.flatMap((page) => page.items) ?? [],
  );

  return (
    <div
      data-testid="shared-conversation-list"
      className="flex min-h-0 flex-1 flex-col overflow-y-auto overflow-x-hidden overscroll-contain custom-scrollbar-always conversation-panel-list-scroll"
    >
      {isLoading && <ConversationCardSkeleton />}

      {!isLoading && conversations.length === 0 && (
        <div
          data-testid="shared-conversation-list-empty-state"
          className="flex min-h-0 flex-1 flex-col items-center justify-center px-4 py-8"
        >
          <p className="text-xs text-muted">
            {t(I18nKey.CONVERSATION$NO_CONVERSATIONS)}
          </p>
        </div>
      )}

      <div className="space-y-0.5">
        {conversations.map((conversation) => (
          <NavigationLink
            key={conversation.id}
            to={backendScopedPath(`/conversations/${conversation.id}`)}
            onClick={onClose}
            className="block"
          >
            <ConversationCard
              title={conversation.title || t(I18nKey.CONVERSATION$UNTITLED)}
              selectedRepository={null}
              lastUpdatedAt={conversation.updated_at}
              showRepositoryMetadata={false}
              isActive={conversation.id === currentConversationId}
              contextMenuOpen={openContextMenuId === conversation.id}
              onContextMenuToggle={(isOpen) =>
                setOpenContextMenuId(isOpen ? conversation.id : null)
              }
              isShared
              onToggleShared={
                !!userId && conversation.created_by_user_id === userId
                  ? (isShared) =>
                      updatePublicFlag({
                        conversationId: conversation.id,
                        isPublic: isShared,
                      })
                  : undefined
              }
            />
          </NavigationLink>
        ))}
      </div>

      {hasNextPage &&
        (isFetchingNextPage ? (
          <div className="py-1">
            <ConversationCardSkeleton />
          </div>
        ) : (
          <div className="flex justify-center py-4">
            <button
              type="button"
              data-testid="load-more-shared-conversations"
              onClick={() => fetchNextPage()}
              className="text-xs text-muted hover:text-contrast"
            >
              {t(I18nKey.CONVERSATION$LOAD_MORE)}
            </button>
          </div>
        ))}
    </div>
  );
}
