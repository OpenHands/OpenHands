import { useTranslation } from "react-i18next";
import type { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import {
  MODAL_MAX_WIDTH_VIEWPORT,
  modalWidthClassName,
} from "#/components/shared/modals/modal-body";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import { cn } from "#/utils/utils";
import { PluginPicker } from "./plugin-picker";

interface PluginPickerModalProps {
  selected: PluginSpec[];
  onChange: (next: PluginSpec[]) => void;
  onClose: () => void;
}

/** Modal shell around the reusable {@link PluginPicker} for the launcher flow. */
export function PluginPickerModal({
  selected,
  onChange,
  onClose,
}: PluginPickerModalProps) {
  const { t } = useTranslation("openhands");

  return (
    <ModalBackdrop onClose={onClose}>
      <div
        data-testid="plugin-picker-modal"
        className={cn(
          "relative flex max-h-[80vh] flex-col gap-4 rounded-xl border border-[var(--oh-border)] bg-base-secondary p-6",
          modalWidthClassName("lg"),
          MODAL_MAX_WIDTH_VIEWPORT,
        )}
      >
        <ModalCloseButton onClose={onClose} testId="plugin-picker-close" />
        <div className="pr-6">
          <Typography.H2>{t(I18nKey.PLUGINS$PICKER_TITLE)}</Typography.H2>
          <Typography.Text className="mt-1 block text-sm text-tertiary-light">
            {t(I18nKey.PLUGINS$PICKER_SUBTITLE)}
          </Typography.Text>
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar">
          <PluginPicker selected={selected} onChange={onChange} />
        </div>

        <div className="flex items-center justify-between border-t border-[var(--oh-border-subtle)] pt-4">
          <Typography.Text className="text-sm text-tertiary-light">
            {t(I18nKey.PLUGINS$PICKER_SELECTED_COUNT, {
              count: selected.length,
            })}
          </Typography.Text>
          <BrandButton
            testId="plugin-picker-done"
            type="button"
            variant="primary"
            onClick={onClose}
            className="px-4"
          >
            {t(I18nKey.PLUGINS$PICKER_DONE)}
          </BrandButton>
        </div>
      </div>
    </ModalBackdrop>
  );
}
