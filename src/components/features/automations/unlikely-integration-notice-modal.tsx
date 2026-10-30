import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import XMarkIcon from "#/icons/x-mark.svg?react";
import { modalTitleLgMediumClassName } from "#/utils/modal-classes";

/** Product names for the built-in integrations an instance reports. */
const INTEGRATION_NAMES: Record<string, string> = {
  github: "GitHub",
  gitlab: "GitLab",
  bitbucket: "Bitbucket Cloud",
  bitbucket_data_center: "Bitbucket Data Center",
  azure_devops: "Azure DevOps",
  jira: "Jira Cloud",
  jira_dc: "Jira Data Center",
  linear: "Linear",
  slack: "Slack",
};

interface UnlikelyIntegrationNoticeModalProps {
  automationName: string;
  /** Built-in integrations, by provider name, the instance does not report. */
  missing: readonly string[];
  onContinue: () => void;
  onCancel: () => void;
}

/**
 * Shown before setting up a template that needs a built-in integration the
 * instance does not report. The report can be incomplete, so the user may
 * continue anyway.
 */
export function UnlikelyIntegrationNoticeModal({
  automationName,
  missing,
  onContinue,
  onCancel,
}: UnlikelyIntegrationNoticeModalProps) {
  const { t, i18n } = useTranslation("openhands");
  const integrations = new Intl.ListFormat(i18n?.language ?? "en", {
    type: "conjunction",
  }).format(missing.map((id) => INTEGRATION_NAMES[id] ?? id));

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div
        className="absolute inset-0 bg-black/60"
        onClick={onCancel}
        onKeyDown={(e) => {
          if (e.key === "Escape") onCancel();
        }}
        role="presentation"
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="unlikely-integration-notice-title"
        data-testid="unlikely-integration-notice"
        className="relative w-full max-w-sm rounded-xl border border-border bg-surface p-6"
      >
        <button
          type="button"
          onClick={onCancel}
          className="absolute right-4 top-4 text-muted hover:text-foreground"
          aria-label={t(I18nKey.BUTTON$CLOSE)}
        >
          <XMarkIcon className="size-5" />
        </button>

        <h2
          id="unlikely-integration-notice-title"
          className={modalTitleLgMediumClassName}
        >
          {automationName}
        </h2>
        <p className="mt-2 text-sm text-muted">
          {t(I18nKey.RECOMMENDED_AUTOMATIONS$UNLIKELY_INTEGRATION_MESSAGE, {
            integrations,
          })}
        </p>

        <div className="mt-6 flex justify-end gap-3">
          <BrandButton
            type="button"
            variant="tertiary"
            testId="unlikely-integration-notice-cancel"
            onClick={onCancel}
          >
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
          <BrandButton
            type="button"
            variant="primary"
            className="bg-contrast text-contrast-foreground"
            testId="unlikely-integration-notice-continue"
            onClick={onContinue}
          >
            {t(I18nKey.BUTTON$CONTINUE)}
          </BrandButton>
        </div>
      </div>
    </div>
  );
}
