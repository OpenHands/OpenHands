import React from "react";
import { useTranslation } from "react-i18next";
import { useQueryClient } from "@tanstack/react-query";
import { I18nKey } from "#/i18n/declaration";
import { AppLoginService } from "#/api/app-login-service";
import { APP_LOGIN_QUERY_KEYS } from "#/hooks/query/query-keys";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import {
  MODAL_MAX_WIDTH_VIEWPORT,
  modalWidthClassName,
} from "#/components/shared/modals/modal-body";
import { cn } from "#/utils/utils";

/**
 * Full-screen username/password gate for internal app login.
 */
export default function AppLoginScreen() {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
  const [username, setUsername] = React.useState("");
  const [password, setPassword] = React.useState("");
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const [errorMessage, setErrorMessage] = React.useState<string | null>(null);

  const canSubmit =
    username.trim().length > 0 && password.length > 0 && !isSubmitting;

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!canSubmit) return;

    setIsSubmitting(true);
    setErrorMessage(null);
    const result = await AppLoginService.login(username.trim(), password);
    if (!result.ok) {
      setErrorMessage(
        result.error.toLowerCase().includes("invalid")
          ? t(I18nKey.APP_LOGIN$INVALID_CREDENTIALS)
          : t(I18nKey.APP_LOGIN$ERROR),
      );
      setIsSubmitting(false);
      return;
    }

    await queryClient.invalidateQueries({
      queryKey: APP_LOGIN_QUERY_KEYS.session,
    });
    queryClient.setQueryData(APP_LOGIN_QUERY_KEYS.session, {
      authenticated: true,
      username: result.username,
    });
    setIsSubmitting(false);
  };

  return (
    <div
      data-testid="app-login-screen"
      className="flex min-h-screen items-center justify-center bg-base px-6"
    >
      <div
        className={cn(
          "relative rounded-xl border border-[var(--oh-border)] bg-base-secondary",
          modalWidthClassName("md"),
          MODAL_MAX_WIDTH_VIEWPORT,
        )}
      >
        <div className="px-6 pt-6 pb-2">
          <h2 className="text-lg font-semibold">
            {t(I18nKey.APP_LOGIN$TITLE)}
          </h2>
          <p className="mt-1 text-sm text-tertiary">
            {t(I18nKey.APP_LOGIN$SUBTITLE)}
          </p>
        </div>

        <form
          className="flex flex-col gap-4 px-6 pb-6 pt-2"
          onSubmit={handleSubmit}
        >
          <SettingsInput
            testId="app-login-username"
            name="username"
            label={t(I18nKey.APP_LOGIN$USERNAME)}
            type="text"
            value={username}
            onChange={setUsername}
            required
          />
          <SettingsInput
            testId="app-login-password"
            name="password"
            label={t(I18nKey.APP_LOGIN$PASSWORD)}
            type="password"
            value={password}
            onChange={setPassword}
            required
          />

          {errorMessage && (
            <p
              data-testid="app-login-error"
              className="text-sm text-danger"
              role="alert"
            >
              {errorMessage}
            </p>
          )}

          <BrandButton
            testId="app-login-submit"
            type="submit"
            variant="primary"
            isDisabled={!canSubmit}
            className="w-full"
          >
            {t(I18nKey.APP_LOGIN$SUBMIT)}
          </BrandButton>
        </form>
      </div>
    </div>
  );
}
