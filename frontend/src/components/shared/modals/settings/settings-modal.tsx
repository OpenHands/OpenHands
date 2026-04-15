import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { ModalBackdrop } from "../modal-backdrop";
import { SettingsForm } from "./settings-form";
import { Settings } from "#/types/settings";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { HelpLink } from "#/ui/help-link";

interface SettingsModalProps {
  settings?: Settings;
  onClose: () => void;
}

export function SettingsModal({ onClose, settings }: SettingsModalProps) {
  const { t } = useTranslation();

  return (
    <ModalBackdrop>
      <div
        data-testid="ai-config-modal"
        className="bg-[#25272D] min-w-full max-w-[475px] m-4 p-6 rounded-xl flex flex-col gap-[17px] border border-tertiary api-configuration-modal"
      >
        <span className="text-5 leading-6 font-semibold -tracking-[0.2px]">
          {t(I18nKey.AI_SETTINGS$TITLE)}
        </span>
        <HelpLink
          testId="advanced-settings-link"
          text={`${t(I18nKey.SETTINGS$DESCRIPTION)}. ${t(I18nKey.SETTINGS$FOR_OTHER_OPTIONS)} ${t(I18nKey.COMMON$SEE)}`}
          linkText={t(I18nKey.COMMON$ADVANCED_SETTINGS)}
          href="/settings"
          suffix="."
          size="settings"
          linkColor="white"
          suffixClassName="text-white"
        />

        <SettingsForm
          settings={settings || DEFAULT_SETTINGS}
          onClose={onClose}
        />
      </div>
    </ModalBackdrop>
  );
}
