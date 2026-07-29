import { useTranslation } from "react-i18next";
import { BaseModalTitle } from "#/components/shared/modals/confirmation-modals/base-modal";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { I18nKey } from "#/i18n/declaration";

interface MetricsModalHeaderProps {
  onClose: () => void;
}

export function MetricsModalHeader({ onClose }: MetricsModalHeaderProps) {
  const { t } = useTranslation("openhands");

  return (
    <>
      <ModalCloseButton onClose={onClose} testId="close-metrics-modal" />
      <div className="w-full pr-6">
        <BaseModalTitle title={t(I18nKey.CONVERSATION$METRICS_INFO)} />
      </div>
    </>
  );
}
