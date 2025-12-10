import { useState } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { RecentConversationsSkeleton } from "./recent-conversations-skeleton";
import { RecentConversation } from "./recent-conversation";
import { usePaginatedConversations } from "#/hooks/query/use-paginated-conversations";
import { useInfiniteScroll } from "#/hooks/use-infinite-scroll";
import { cn } from "#/utils/utils";

export function RecentConversations() {
  const { t } = useTranslation();
  const [isExpanded, setIsExpanded] = useState(false);

  const {
    data: conversationsList,
    isFetching,
    isFetchingNextPage,
    error,
    hasNextPage,
    fetchNextPage,
  } = usePaginatedConversations(10);

  // Set up infinite scroll
  const scrollContainerRef = useInfiniteScroll({
    hasNextPage: !!hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
    threshold: 200, // Load more when 200px from bottom
  });

  const conversations =
    conversationsList?.pages.flatMap((page) => page.results) ?? [];

  // Get the conversations to display based on expansion state
  const displayLimit = isExpanded ? 10 : 3;
  const displayedConversations = conversations.slice(0, displayLimit);

  const hasConversations = conversations && conversations.length > 0;

  // Check if there are more conversations to show
  const hasMoreConversations =
    conversations && conversations.length > displayLimit;

  // Check if this is the initial load (no data yet)
  const isInitialLoading = isFetching && !conversationsList;

  const handleToggleExpansion = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <section
      data-testid="recent-conversations"
      className="flex flex-1 min-w-0 flex-col"
    >
      <div
        className={cn(
          "flex items-center gap-2",
          !hasConversations && "mb-[14px]",
        )}
      >
        <h3 className="text-xs leading-4 text-white font-bold py-[14px] pl-4">
          {t(I18nKey.COMMON$RECENT_CONVERSATIONS)}
        </h3>
      </div>

      {error && (
        <div className="flex flex-col items-center justify-center h-full pl-4">
          <p className="text-danger">{error.message}</p>
        </div>
      )}

      <div className="flex flex-col">
        {isInitialLoading && (
          <div className="pl-4">
            <RecentConversationsSkeleton />
          </div>
        )}
      </div>

      {!isInitialLoading && !error && displayedConversations?.length === 0 && (
        <span className="text-xs leading-4 text-white font-medium pl-4">
          {t(I18nKey.HOME$NO_RECENT_CONVERSATIONS)}
        </span>
      )}

      {!isInitialLoading &&
        displayedConversations &&
        displayedConversations.length > 0 && (
          <div className="flex flex-col">
            <div className="transition-all duration-300 ease-in-out overflow-y-auto custom-scrollbar">
              <div ref={scrollContainerRef} className="flex flex-col">
                {displayedConversations.map((conversation) => (
                  <RecentConversation
                    key={conversation.conversation_id}
                    conversation={conversation}
                  />
                ))}
              </div>
            </div>
          </div>
        )}

      {!isInitialLoading && (hasMoreConversations || isExpanded) && (
        <div className="flex justify-start mt-6 mb-8 ml-4">
          <button
            type="button"
            onClick={handleToggleExpansion}
            className="text-xs leading-4 text-[#FAFAFA] font-normal cursor-pointer hover:underline"
          >
            {isExpanded
              ? t(I18nKey.COMMON$VIEW_LESS)
              : t(I18nKey.COMMON$VIEW_MORE)}
          </button>
        </div>
      )}
    </section>
  );
}
