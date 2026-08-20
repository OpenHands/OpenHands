import { useTranslation } from "react-i18next";
import ArrowUpIcon from "#/icons/u-arrow-up.svg?react";
import { cn, getGitPushPrompt } from "#/utils/utils";
import {
  gitControlBarActionButtonClassName,
  gitControlBarActionIconColor,
  gitControlBarActionLabelClassName,
} from "#/utils/git-control-bar-classes";
import { I18nKey } from "#/i18n/declaration";
import { Provider } from "#/types/settings";
import { useTracking } from "#/hooks/use-tracking";

interface GitControlBarPushButtonProps {
  onSuggestionsClick: (value: string) => void;
  hasRepository: boolean;
  providerTokensReady: boolean;
  currentGitProvider: Provider;
  isConversationReady?: boolean;
}

export function GitControlBarPushButton({
  onSuggestionsClick,
  hasRepository,
  providerTokensReady,
  currentGitProvider,
  isConversationReady = true,
}: GitControlBarPushButtonProps) {
  const { t } = useTranslation("openhands");
  const { trackPushButtonClick } = useTracking();

  const isButtonEnabled =
    providerTokensReady && hasRepository && isConversationReady;

  const handlePushClick = () => {
    trackPushButtonClick();
    onSuggestionsClick(getGitPushPrompt(currentGitProvider));
  };

  return (
    <button
      type="button"
      onClick={handlePushClick}
      disabled={!isButtonEnabled}
      className={cn(
        gitControlBarActionButtonClassName(isButtonEnabled),
        "px-2 py-1 w-[77px] min-w-[77px]",
      )}
    >
      <div className="w-3 h-3 flex items-center justify-center">
        <ArrowUpIcon
          width={12}
          height={12}
          color={gitControlBarActionIconColor(isButtonEnabled)}
        />
      </div>
      <div
        className={cn(gitControlBarActionLabelClassName, "max-w-[77px]")}
        title={t(I18nKey.COMMON$PUSH)}
      >
        {t(I18nKey.COMMON$PUSH)}
      </div>
    </button>
  );
}
