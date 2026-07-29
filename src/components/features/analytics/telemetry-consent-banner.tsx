import React from "react";
import { useTranslation } from "react-i18next";
import { getLockedCloudHost } from "#/api/agent-server-config";
import { useActiveBackend } from "#/contexts/active-backend-context";
import type { Backend } from "#/api/backend-registry/types";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { useBackendsHealth } from "#/hooks/query/use-backends-health";
import { useSettings } from "#/hooks/query/use-settings";
import { I18nKey } from "#/i18n/declaration";
import { OPENHANDS_I18N_NAMESPACE } from "#/i18n";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import {
  BaseModalTitle,
  BaseModalDescription,
} from "#/components/shared/modals/confirmation-modals/base-modal";
import { BrandButton } from "#/components/features/settings/brand-button";
import { setTelemetryConsent } from "#/services/telemetry";

interface TelemetryConsentBannerProps {
  onChoice?: (granted: boolean) => void;
}

function LocalTelemetryConsentBanner({
  backend,
  onChoice,
}: TelemetryConsentBannerProps & {
  backend: Backend;
}) {
  const { t, ready } = useTranslation(OPENHANDS_I18N_NAMESPACE);
  const health = useBackendsHealth([backend]);
  const isBackendConnected = health[backend.id]?.isConnected === true;
  const { data: settings, isSuccess: hasLoadedSettings } = useSettings();
  const { mutateAsync: saveSettings, isPending: isSavingSettings } =
    useSaveSettings();
  const [isReady, setIsReady] = React.useState(false);
  const [hasSubmittedChoice, setHasSubmittedChoice] = React.useState(false);

  React.useEffect(() => {
    setHasSubmittedChoice(false);
  }, [backend.id]);

  const shouldShow =
    isBackendConnected &&
    hasLoadedSettings &&
    settings?.user_consents_to_analytics === null &&
    !hasSubmittedChoice;

  React.useEffect(() => {
    if (ready && shouldShow) {
      const timer = setTimeout(() => setIsReady(true), 50);
      return () => clearTimeout(timer);
    }
    setIsReady(false);
    return undefined;
  }, [ready, shouldShow]);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const granted = formData.get("analytics") === "on";

    try {
      await saveSettings({ user_consents_to_analytics: granted });
    } catch {
      return;
    }

    setHasSubmittedChoice(true);
    await setTelemetryConsent(granted ? "granted" : "denied");
    onChoice?.(granted);
  };

  if (!shouldShow || !isReady) return null;

  return (
    <ModalBackdrop elevated aria-label={t(I18nKey.TELEMETRY$CONSENT_TITLE)}>
      <form
        data-testid="telemetry-consent-form"
        onSubmit={handleSubmit}
        className="flex flex-col gap-2"
      >
        <ModalBody className="border border-[var(--oh-border)]">
          <BaseModalTitle title={t(I18nKey.TELEMETRY$CONSENT_TITLE)} />
          <BaseModalDescription>
            {t(I18nKey.TELEMETRY$CONSENT_DESCRIPTION)}
          </BaseModalDescription>
          <p className="text-xs text-neutral-400">
            {t(I18nKey.TELEMETRY$BACKEND_SCOPE, {
              name: backend.name,
              host: backend.host,
            })}
          </p>

          <label className="flex gap-2 items-center self-start text-sm cursor-pointer">
            <input
              name="analytics"
              type="checkbox"
              defaultChecked
              className="w-4 h-4 cursor-pointer"
            />
            {t(I18nKey.TELEMETRY$SEND_ANONYMOUS_DATA)}
          </label>

          <BrandButton
            testId="confirm-telemetry-preferences"
            type="submit"
            variant="primary"
            className="w-full"
            isDisabled={isSavingSettings}
          >
            {t(I18nKey.TELEMETRY$CONFIRM_PREFERENCES)}
          </BrandButton>
        </ModalBody>
      </form>
    </ModalBackdrop>
  );
}

export function TelemetryConsentBanner({
  onChoice,
}: TelemetryConsentBannerProps) {
  const { backend } = useActiveBackend();

  if (getLockedCloudHost() !== null || backend.kind !== "local") return null;

  return <LocalTelemetryConsentBanner backend={backend} onChoice={onChoice} />;
}
