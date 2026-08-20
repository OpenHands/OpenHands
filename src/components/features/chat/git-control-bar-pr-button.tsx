import { useTranslation } from "react-i18next";
import PRIcon from "#/icons/u-pr.svg?react";
import { cn, getCreatePRPrompt } from "#/utils/utils";
import {
  gitControlBarActionButtonClassName,
  gitControlBarActionIconColor,
  gitControlBarActionLabelClassName,
} from "#/utils/git-control-bar-classes";
import { I18nKey } from "#/i18n/declaration";
import { Provider } from "#/types/settings";
import { useTracking } from "#/hooks/use-tracking";

interface GitControlBarPrButtonProps {
  onSuggestionsClick: (value: string) => void;
  hasRepository: boolean;
  providerTokensReady: boolean;
  currentGitProvider: Provider;
  isConversationReady?: boolean;
}

export function GitControlBarPrButton({
  onSuggestionsClick,
  hasRepository,
  providerTokensReady,
  currentGitProvider,
  isConversationReady = true,
}: GitControlBarPrButtonProps) {
  const { t } = useTranslation("openhands");
  const { trackCreatePrButtonClick } = useTracking();

  const isButtonEnabled =
    providerTokensReady && hasRepository && isConversationReady;

  const handlePrClick = () => {
    trackCreatePrButtonClick();
    onSuggestionsClick(getCreatePRPrompt(currentGitProvider));
  };

  return (
    <button
      type="button"
      onClick={handlePrClick}
      disabled={!isButtonEnabled}
      className={cn(
        gitControlBarActionButtonClassName(isButtonEnabled),
        "px-2 py-1 w-[126px] min-w-[126px] h-7",
      )}
    >
      <div className="w-3 h-3 flex items-center justify-center">
        <PRIcon
          width={12}
          height={12}
          color={gitControlBarActionIconColor(isButtonEnabled)}
        />
      </div>
      <div
        className={cn(gitControlBarActionLabelClassName, "max-w-[126px]")}
        title={t(I18nKey.COMMON$PULL_REQUEST)}
      >
        {t(I18nKey.COMMON$PULL_REQUEST)}
      </div>
    </button>
  );
}
