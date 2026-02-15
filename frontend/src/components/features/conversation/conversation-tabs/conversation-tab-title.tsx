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
    <div className="flex flex-row items-center justify-between border-b border-[#27272A] py-2 px-3">
      <span className="text-xs font-medium text-[#FAFAFA]">{title}</span>
      {conversationKey === "editor" && (
        <button
          type="button"
          className="flex w-[26px] py-1 justify-center items-center gap-[10px] rounded-lg hover:bg-[#27272A] transition-colors duration-150 cursor-pointer"
          onClick={handleRefresh}
        >
          <RefreshIcon width={12.75} height={15} color="#ffffff" />
        </button>
      )}
    </div>
  );
}
