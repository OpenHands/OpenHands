import { Info } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { mapProvider } from "#/utils/map-provider";
import {
  DEFAULT_ATTACHMENT_LIMIT_MB,
  getProviderImageLimitMb,
  hasProviderImageLimit,
} from "#/utils/provider-upload-limits";

/**
 * Shows the maximum attachment size that applies to the selected provider.
 *
 * The image ceiling is the provider's own limit, so the number the user is
 * held to is visible before they attach anything. A provider with no
 * documented limit falls back to the default, and the note says so, rather
 * than implying the fallback is the provider's limit.
 */
export function ProviderUploadLimitNote({
  provider,
}: {
  provider: string | null;
}) {
  const { t } = useTranslation("openhands");

  if (!provider) return null;

  const imageLimitMb = getProviderImageLimitMb(provider);
  const providerLabel = mapProvider(provider);

  if (!hasProviderImageLimit(provider)) {
    return (
      <p
        data-testid="llm-provider-upload-limit-note"
        className="flex items-start gap-2 text-xs text-content-2"
      >
        <Info className="mt-0.5 size-4 shrink-0" aria-hidden />
        <span>
          {t(I18nKey.SETTINGS$PROVIDER_UPLOAD_LIMIT_UNKNOWN, {
            provider: providerLabel,
            limitMb: imageLimitMb,
          })}
        </span>
      </p>
    );
  }

  return (
    <p
      data-testid="llm-provider-upload-limit-note"
      className="flex items-start gap-2 text-xs text-content-2"
    >
      <Info className="mt-0.5 size-4 shrink-0" aria-hidden />
      <span>
        {t(I18nKey.SETTINGS$PROVIDER_UPLOAD_LIMIT, {
          provider: providerLabel,
          limitMb: imageLimitMb,
          attachmentLimitMb: DEFAULT_ATTACHMENT_LIMIT_MB,
        })}
      </span>
    </p>
  );
}
