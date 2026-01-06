import { useTranslation } from "react-i18next";
import { Tooltip } from "@heroui/react";
import { I18nKey } from "#/i18n/declaration";
import LockIcon from "#/icons/lock.svg?react";
import { useSettings } from "#/hooks/query/use-settings";

function ConfirmationModeEnabled() {
  const { t } = useTranslation();

  const { data: settings } = useSettings();

  if (!settings?.confirmation_mode) {
    return null;
  }

  return (
    <Tooltip
      content={t(I18nKey.COMMON$CONFIRMATION_MODE_ENABLED)}
      closeDelay={100}
      className="bg-white text-black hover:bg-transparent"
    >
      <div className="flex items-center justify-center w-[26px] h-[26px] rounded-lg bg-[#25272D]">
        <LockIcon width={15} height={15} />
      </div>
    </Tooltip>
  );
}

export default ConfirmationModeEnabled;
