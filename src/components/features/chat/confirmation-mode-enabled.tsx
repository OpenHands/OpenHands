import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import LockIcon from "#/icons/lock.svg?react";
import { useSettings } from "#/hooks/query/use-settings";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";

function ConfirmationModeEnabled() {
  const { t } = useTranslation("openhands");

  const { data: settings } = useSettings();

  if (!settings?.confirmation_mode) {
    return null;
  }

  return (
    <StyledTooltip
      closeDelay={100}
      content={t(I18nKey.COMMON$CONFIRMATION_MODE_ENABLED)}
      tooltipClassName="bg-white text-black hover:bg-transparent"
    >
      <div className="flex items-center justify-center w-6.5 h-6.5 rounded-lg bg-surface">
        <LockIcon width={15} height={15} />
      </div>
    </StyledTooltip>
  );
}

export default ConfirmationModeEnabled;
