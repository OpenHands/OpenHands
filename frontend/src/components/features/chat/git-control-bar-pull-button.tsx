import { useTranslation } from "react-i18next";
import ArrowDownIcon from "#/icons/u-arrow-down.svg?react";
import { cn, getGitPullPrompt } from "#/utils/utils";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useUserProviders } from "#/hooks/use-user-providers";
import { I18nKey } from "#/i18n/declaration";
import { useTracking } from "#/hooks/use-tracking";

interface GitControlBarPullButtonProps {
  onSuggestionsClick: (value: string) => void;
  isConversationReady?: boolean;
}

export function GitControlBarPullButton({
  onSuggestionsClick,
  isConversationReady = true,
}: GitControlBarPullButtonProps) {
  const { t } = useTranslation();
  const { trackPullButtonClick } = useTracking();

  const { data: conversation } = useActiveConversation();
  const { providers } = useUserProviders();

  const providersAreSet = providers.length > 0;
  const hasRepository = conversation?.selected_repository;
  const isButtonEnabled =
    providersAreSet && hasRepository && isConversationReady;

  const handlePullClick = () => {
    trackPullButtonClick();
    onSuggestionsClick(getGitPullPrompt());
  };

  return (
    <button
      type="button"
      onClick={handlePullClick}
      disabled={!isButtonEnabled}
      className={cn(
        "flex flex-row gap-1 items-center justify-center px-0.5 py-1 rounded-[100px] w-[76px] min-w-[76px]",
        isButtonEnabled
          ? "bg-[#25272D] hover:bg-[#454545] cursor-pointer"
          : "bg-[rgba(71,74,84,0.50)] cursor-not-allowed",
      )}
    >
      <div className="w-3 h-3 flex items-center justify-center">
        <ArrowDownIcon width={12} height={12} color="white" />
      </div>
      <div
        className="font-normal text-white text-sm leading-5 max-w-[76px] truncate"
        title={t(I18nKey.COMMON$PULL)}
      >
        {t(I18nKey.COMMON$PULL)}
      </div>
    </button>
  );
}
