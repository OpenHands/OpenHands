import React from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import {
  BaseModalDescription,
  BaseModalTitle,
} from "#/components/shared/modals/confirmation-modals/base-modal";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { I18nKey } from "#/i18n/declaration";
import type {
  AutomationDisableFeedback,
  AutomationDisableReason,
} from "#/types/automation-disable-feedback";

export type { AutomationDisableFeedback } from "#/types/automation-disable-feedback";

const AUTOMATION_DISABLE_REASON_RADIO_NAME = "automation-disable-reason";

const REASON_OPTIONS: Array<{
  value: AutomationDisableReason;
  label: I18nKey;
}> = [
  {
    value: "no_longer_needed",
    label: I18nKey.AUTOMATIONS$DISABLE_REASON_NO_LONGER_NEEDED,
  },
  {
    value: "unreliable",
    label: I18nKey.AUTOMATIONS$DISABLE_REASON_UNRELIABLE,
  },
  {
    value: "misconfigured",
    label: I18nKey.AUTOMATIONS$DISABLE_REASON_MISCONFIGURED,
  },
  {
    value: "too_noisy",
    label: I18nKey.AUTOMATIONS$DISABLE_REASON_TOO_NOISY,
  },
  {
    value: "too_expensive",
    label: I18nKey.AUTOMATIONS$DISABLE_REASON_TOO_EXPENSIVE,
  },
  {
    value: "low_quality",
    label: I18nKey.AUTOMATIONS$DISABLE_REASON_LOW_QUALITY,
  },
  {
    value: "other",
    label: I18nKey.AUTOMATIONS$DISABLE_REASON_OTHER,
  },
];

interface AutomationDisableFeedbackModalProps {
  onSubmit: (feedback: AutomationDisableFeedback) => void;
  onDismiss: () => void;
}

export function AutomationDisableFeedbackModal({
  onSubmit,
  onDismiss,
}: AutomationDisableFeedbackModalProps) {
  const { t } = useTranslation("openhands");
  const [reason, setReason] = React.useState<AutomationDisableReason | null>(
    null,
  );
  const [details, setDetails] = React.useState("");

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!reason) return;

    const trimmedDetails = details.trim();
    onSubmit({
      reason,
      details: trimmedDetails || undefined,
    });
  };

  return (
    <ModalBackdrop
      onClose={onDismiss}
      aria-label={t(I18nKey.AUTOMATIONS$DISABLE_FEEDBACK_TITLE)}
    >
      <ModalBody
        testID="automation-disable-feedback-modal"
        width="md"
        className="relative items-start border border-[var(--oh-border)]"
      >
        <ModalCloseButton
          onClose={onDismiss}
          testId="close-automation-disable-feedback"
        />

        <div className="flex flex-col gap-2 pr-8">
          <BaseModalTitle
            title={t(I18nKey.AUTOMATIONS$DISABLE_FEEDBACK_TITLE)}
          />
          <BaseModalDescription
            description={t(I18nKey.AUTOMATIONS$DISABLE_FEEDBACK_DESCRIPTION)}
          />
        </div>

        <form className="flex w-full flex-col gap-5" onSubmit={handleSubmit}>
          <fieldset className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            <legend className="mb-2 text-sm font-medium text-white sm:col-span-2">
              {t(I18nKey.AUTOMATIONS$DISABLE_FEEDBACK_REASON_LABEL)}
            </legend>
            {REASON_OPTIONS.map((option) => (
              <label
                key={option.value}
                className="flex cursor-pointer items-center gap-3 rounded-lg border border-[var(--oh-border)] px-3 py-2.5 text-sm text-white hover:bg-surface-raised"
              >
                <input
                  type="radio"
                  name={AUTOMATION_DISABLE_REASON_RADIO_NAME}
                  value={option.value}
                  checked={reason === option.value}
                  onChange={() => setReason(option.value)}
                />
                {t(option.label)}
              </label>
            ))}
          </fieldset>

          <label className="flex flex-col gap-2 text-sm font-medium text-white">
            {t(I18nKey.AUTOMATIONS$DISABLE_FEEDBACK_DETAILS_LABEL)}
            <textarea
              value={details}
              onChange={(event) => setDetails(event.target.value)}
              placeholder={t(
                I18nKey.AUTOMATIONS$DISABLE_FEEDBACK_DETAILS_PLACEHOLDER,
              )}
              rows={3}
              className="w-full resize-y rounded-lg border border-[var(--oh-border)] bg-base px-3 py-2 text-sm font-normal text-white placeholder:text-muted focus:border-primary focus:outline-none"
            />
          </label>

          <div className="flex justify-end gap-2">
            <BrandButton
              type="button"
              variant="secondary"
              onClick={onDismiss}
              testId="skip-automation-disable-feedback"
            >
              {t(I18nKey.AUTOMATIONS$DISABLE_FEEDBACK_SKIP)}
            </BrandButton>
            <BrandButton
              type="submit"
              variant="primary"
              isDisabled={!reason}
              testId="submit-automation-disable-feedback"
            >
              {t(I18nKey.AUTOMATIONS$DISABLE_FEEDBACK_SUBMIT)}
            </BrandButton>
          </div>
        </form>
      </ModalBody>
    </ModalBackdrop>
  );
}
