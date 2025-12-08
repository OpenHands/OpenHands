import { useTranslation } from "react-i18next";
import { usePostHog } from "posthog-js/react";
import {
  BaseModalTitle,
  BaseModalDescription,
} from "#/components/shared/modals/confirmation-modals/base-modal";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { handleCaptureConsent } from "#/utils/handle-capture-consent";
import { BrandButton } from "../settings/brand-button";
import { I18nKey } from "#/i18n/declaration";

interface AnalyticsConsentFormModalProps {
  onClose: () => void;
}

export function AnalyticsConsentFormModal({
  onClose,
}: AnalyticsConsentFormModalProps) {
  const posthog = usePostHog();
  const { t } = useTranslation();
  const { mutate: saveUserSettings } = useSaveSettings();

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const analytics = formData.get("analytics") === "on";

    saveUserSettings(
      { user_consents_to_analytics: analytics },
      {
        onSuccess: () => {
          handleCaptureConsent(posthog, analytics);
          onClose();
        },
      },
    );
  };

  return (
    <ModalBackdrop>
      <form
        data-testid="user-capture-consent-form"
        onSubmit={handleSubmit}
        className="flex flex-col gap-2"
      >
        <ModalBody className="border border-tertiary">
          <BaseModalTitle title={t(I18nKey.ANALYTICS$TITLE)} />
          <BaseModalDescription>
            {t(I18nKey.ANALYTICS$DESCRIPTION)}
          </BaseModalDescription>

          <label className="flex gap-2 items-center self-start text-sm">
            <input name="analytics" type="checkbox" defaultChecked />
            {t(I18nKey.ANALYTICS$SEND_ANONYMOUS_DATA)}
          </label>

          <BrandButton
            testId="confirm-preferences"
            type="submit"
            variant="primary"
            className="w-full"
          >
            {t(I18nKey.ANALYTICS$CONFIRM_PREFERENCES)}
          </BrandButton>
        </ModalBody>
      </form>
    </ModalBackdrop>
  );
}
