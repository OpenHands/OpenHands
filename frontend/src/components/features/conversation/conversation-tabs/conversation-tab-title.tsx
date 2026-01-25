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
  const { refetch, isFetching } = useUnifiedGetGitChanges();

  const handleRefresh = () => {
    refetch();
  };

  return (
    <div className="flex flex-row items-center justify-between border-b border-[#474A54] py-2 px-3">
      <span className="text-xs font-medium text-white">{title}</span>

      {conversationKey === "editor" && (
        <button
          type="button"
          onClick={handleRefresh}
          disabled={isFetching}
          className={cn(
            "flex w-[26px] py-1 items-center justify-center rounded-[7px] transition",
            isFetching
              ? "opacity-50 cursor-not-allowed"
              : "hover:bg-[#474A54] cursor-pointer",
          )}
        >
          <RefreshIcon
            width={12.75}
            height={15}
            color="#ffffff"
            className={isFetching ? "animate-spin" : ""}
          />
        </button>
      )}
    </div>
  );
}
