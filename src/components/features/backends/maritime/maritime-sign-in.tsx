import React from "react";
import { useTranslation } from "react-i18next";
import { ExternalLink, Loader2 } from "lucide-react";

import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { ErrorAlert } from "#/components/shared/error-alert";
import { I18nKey } from "#/i18n/declaration";

export const MARITIME_CREATE_API_KEY_URL =
  "https://maritime.sh/settings/api-keys";

interface MaritimeSignInProps {
  isSaving: boolean;
  error: string | null;
  onSaveApiKey: (apiKey: string) => void;
}

/** The Maritime API key form. Maritime has no OAuth app, so the key is the only path. */
export function MaritimeSignIn({
  isSaving,
  error,
  onSaveApiKey,
}: MaritimeSignInProps) {
  const { t } = useTranslation("openhands");
  const [apiKey, setApiKey] = React.useState("");

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    if (apiKey.trim() && !isSaving) onSaveApiKey(apiKey);
  };

  return (
    <div
      data-testid="maritime-sign-in"
      className="mx-auto flex w-full max-w-sm flex-col items-center gap-4 py-6"
    >
      <div className="flex flex-col items-center gap-1.5 text-center">
        <h3 className="text-balance text-base font-medium text-white">
          {t(I18nKey.MARITIME$SIGN_IN_TITLE)}
        </h3>
        <p className="text-pretty text-sm leading-relaxed text-muted">
          {t(I18nKey.MARITIME$SIGN_IN_DESCRIPTION)}
        </p>
      </div>

      <form onSubmit={submit} className="flex w-full flex-col gap-2">
        <SettingsInput
          testId="maritime-api-key"
          name="maritime-api-key"
          type="password"
          label={t(I18nKey.MARITIME$API_KEY_LABEL)}
          value={apiKey}
          onChange={setApiKey}
          // eslint-disable-next-line i18next/no-literal-string -- key format, not translatable
          placeholder="mk_..."
          className="w-full"
        />
        <p className="text-xs leading-relaxed text-muted">
          {t(I18nKey.MARITIME$API_KEY_HINT)}{" "}
          <a
            href={MARITIME_CREATE_API_KEY_URL}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 text-primary hover:underline"
          >
            {t(I18nKey.MARITIME$CREATE_API_KEY)}
            <ExternalLink className="size-3" aria-hidden />
          </a>
        </p>
        <BrandButton
          type="submit"
          variant="primary"
          isDisabled={!apiKey.trim() || isSaving}
          testId="maritime-api-key-submit"
          className="w-full"
          startContent={
            isSaving ? (
              <Loader2 className="size-4 animate-spin" aria-hidden />
            ) : null
          }
        >
          {t(I18nKey.MARITIME$API_KEY_SUBMIT)}
        </BrandButton>
      </form>

      {error ? (
        <ErrorAlert testId="maritime-auth-error" className="w-full">
          {error}
        </ErrorAlert>
      ) : null}
    </div>
  );
}
