import RefreshIcon from "#/icons/u-refresh.svg?react";
import { useUnifiedGetGitChanges } from "#/hooks/query/use-unified-get-git-changes";

type ConversationTabTitleProps = {
  title: string;
  conversationKey: string;
};

export function ConversationTabTitle({
  title,
  conversationKey,
}: ConversationTabTitleProps) {
  const { refetch } = useUnifiedGetGitChanges();

  const handleRefresh = () => {
    refetch();
  };

  return (
    <div className="flex flex-row items-center justify-between border-b border-stroke py-2 px-3">
      <span className="text-xs font-medium text-content">{title}</span>
      {conversationKey === "editor" && (
        <button
          type="button"
          className="flex w-[26px] py-1 justify-center items-center gap-[10px] rounded-[7px] hover:bg-stroke cursor-pointer"
          onClick={handleRefresh}
        >
          <RefreshIcon width={12.75} height={15} className="text-content" />
        </button>
      )}
    </div>
  );
}
