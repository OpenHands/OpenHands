import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import ExclamationCircleIcon from "#/icons/exclamation-circle.svg?react";
import { BackLink } from "#/components/features/automations/detail/back-link";

/**
 * Shown when the automation backend answers the git-sync endpoints with 404:
 * it predates the API. Without this the page settles on the generic "something
 * went wrong" panel, which reads as a broken install rather than an old
 * backend.
 */
export function GitSyncUnsupportedState() {
  const { t } = useTranslation("openhands");

  return (
    <div className="flex flex-col items-center justify-center py-20">
      <ExclamationCircleIcon className="size-12 text-muted" />
      <p className="mt-4 text-sm font-medium text-content">
        {t(I18nKey.AUTOMATIONS$GIT_SYNC$UNSUPPORTED_TITLE)}
      </p>
      <p className="mt-2 text-sm text-muted">
        {t(I18nKey.AUTOMATIONS$GIT_SYNC$UNSUPPORTED_MESSAGE)}
      </p>
      <div className="mt-6">
        <BackLink />
      </div>
    </div>
  );
}
