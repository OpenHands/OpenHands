import { useTranslation } from "react-i18next";
import ArrowDownIcon from "#/icons/u-arrow-down.svg?react";
import { cn, getGitPullPrompt } from "#/utils/utils";
import {
  gitControlBarActionButtonClassName,
  gitControlBarActionIconColor,
  gitControlBarActionLabelClassName,
} from "#/utils/git-control-bar-classes";
import { I18nKey } from "#/i18n/declaration";
import { useTracking } from "#/hooks/use-tracking";

interface GitControlBarPullButtonProps {
  onSuggestionsClick: (value: string) => void;
  hasRepository: boolean;
  providerTokensReady: boolean;
  isConversationReady?: boolean;
}

export function GitControlBarPullButton({
  onSuggestionsClick,
  hasRepository,
  providerTokensReady,
  isConversationReady = true,
}: GitControlBarPullButtonProps) {
  const { t } = useTranslation("openhands");
  const { trackPullButtonClick } = useTracking();

  const isButtonEnabled =
    providerTokensReady && hasRepository && isConversationReady;

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
        gitControlBarActionButtonClassName(isButtonEnabled),
        "px-0.5 py-1 w-[76px] min-w-[76px]",
      )}
    >
      <div className="w-3 h-3 flex items-center justify-center">
        <ArrowDownIcon
          width={12}
          height={12}
          color={gitControlBarActionIconColor(isButtonEnabled)}
        />
      </div>
      <div
        className={cn(gitControlBarActionLabelClassName, "max-w-[76px]")}
        title={t(I18nKey.COMMON$PULL)}
      >
        {t(I18nKey.COMMON$PULL)}
      </div>
    </button>
  );
}
