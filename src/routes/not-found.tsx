import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import ExclamationCircleIcon from "#/icons/exclamation-circle.svg?react";
import { BackNavButton } from "#/components/shared/buttons/back-nav-button";

/**
 * Catch-all for URLs no route matches. It renders inside the root layout so
 * the sidebar stays usable, instead of React Router's bare document-level 404
 * (which also failed hydration of the prerendered SPA shell).
 */
export default function NotFoundRoute() {
  const { t } = useTranslation("openhands");

  return (
    <div
      data-testid="not-found-screen"
      className="flex h-full flex-col items-center justify-center px-6 py-20 text-center"
    >
      <ExclamationCircleIcon className="size-12 text-muted" aria-hidden />
      <h1 className="mt-4 text-sm font-medium text-content">
        {t(I18nKey.NOT_FOUND$TITLE)}
      </h1>
      <p className="mt-2 text-sm text-muted">{t(I18nKey.NOT_FOUND$MESSAGE)}</p>
      <BackNavButton
        to="/"
        testId="not-found-home-link"
        className="mt-6 self-center"
      >
        {t(I18nKey.BUTTON$HOME)}
      </BackNavButton>
    </div>
  );
}
