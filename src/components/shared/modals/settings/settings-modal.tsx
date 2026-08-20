import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { ModalBackdrop } from "../modal-backdrop";
import { MODAL_MAX_WIDTH_VIEWPORT, modalWidthClassName } from "../modal-body";
import { cn } from "#/utils/utils";
import { SettingsForm } from "./settings-form";
import { Settings } from "#/types/settings";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { HelpLink } from "#/ui/help-link";
import { modalTitleClassName } from "#/utils/modal-classes";
import { buildAgentCanvasPath } from "#/utils/base-path";

interface SettingsModalProps {
  settings?: Settings;
  onClose: () => void;
}

export function SettingsModal({ onClose, settings }: SettingsModalProps) {
  const { t } = useTranslation("openhands");

  return (
    <ModalBackdrop>
      <div
        data-testid="ai-config-modal"
        className={cn(
          "bg-[var(--oh-surface)] m-4 p-6 rounded-xl flex flex-col gap-[17px] border border-[var(--oh-border)] api-configuration-modal",
          modalWidthClassName("md"),
          MODAL_MAX_WIDTH_VIEWPORT,
        )}
      >
        <span className={modalTitleClassName}>
          {t(I18nKey.AI_SETTINGS$TITLE)}
        </span>
        <HelpLink
          testId="advanced-settings-link"
          text={`${t(I18nKey.SETTINGS$DESCRIPTION)}. ${t(I18nKey.SETTINGS$FOR_OTHER_OPTIONS)} ${t(I18nKey.COMMON$SEE)}`}
          linkText={t(I18nKey.COMMON$ADVANCED_SETTINGS)}
          href={buildAgentCanvasPath("/settings")}
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
