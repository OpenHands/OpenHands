import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { I18nKey } from "#/i18n/declaration";
import { modalTitleLgClassName } from "#/utils/modal-classes";

interface ProviderSetupNoticeModalProps {
  automationName: string;
  /** Display names of the git providers the instance has set up no way. */
  providers: readonly string[];
  onContinue: () => void;
  onCancel: () => void;
}

/**
 * Shown before setting up a template on a self-hosted instance that has none
 * of the template's git providers set up: no built-in integration configured,
 * no MCP server installed and no token secret saved. Continue goes on to set
 * up the provider's MCP server, whose token is saved as the secret the
 * template reads.
 */
export function ProviderSetupNoticeModal({
  automationName,
  providers,
  onContinue,
  onCancel,
}: ProviderSetupNoticeModalProps) {
  const { t, i18n } = useTranslation("openhands");
  const provider = new Intl.ListFormat(i18n?.language ?? "en", {
    type: "conjunction",
  }).format(providers);

  return (
    <ModalBackdrop onClose={onCancel} aria-label={automationName}>
      <div
        data-testid="provider-setup-notice"
        className="relative w-full max-w-sm rounded-xl border border-border bg-surface p-6"
      >
        <ModalCloseButton
          onClose={onCancel}
          testId="provider-setup-notice-close"
        />
        <h2 className={modalTitleLgClassName}>{automationName}</h2>
        <p className="mt-2 text-sm text-muted">
          {t(I18nKey.RECOMMENDED_AUTOMATIONS$PROVIDER_NOT_CONFIGURED, {
            provider,
          })}
        </p>
        <p className="mt-2 text-sm text-muted">
          {t(I18nKey.RECOMMENDED_AUTOMATIONS$PROVIDER_ALTERNATIVE, {
            provider,
          })}
        </p>
        <div className="mt-6 flex justify-end gap-3">
          <BrandButton
            type="button"
            variant="tertiary"
            testId="provider-setup-notice-cancel"
            onClick={onCancel}
          >
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
          <BrandButton
            type="button"
            variant="primary"
            className="bg-contrast text-contrast-foreground"
            testId="provider-setup-notice-continue"
            onClick={onContinue}
          >
            {t(I18nKey.BUTTON$CONTINUE)}
          </BrandButton>
        </div>
      </div>
    </ModalBackdrop>
  );
}
