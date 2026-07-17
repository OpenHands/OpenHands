import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";

const RELOAD_SENTINEL_KEY = "openhands:cross-app-reload";

export default function CrossAppRedirect() {
  const { t } = useTranslation();
  const [alreadyReloaded] = React.useState(() => {
    try {
      return (
        sessionStorage.getItem(RELOAD_SENTINEL_KEY) === window.location.href
      );
    } catch {
      return false;
    }
  });

  React.useEffect(() => {
    if (alreadyReloaded) {
      return;
    }

    try {
      sessionStorage.setItem(RELOAD_SENTINEL_KEY, window.location.href);
    } catch {
      // Ignore storage failures; the browser should still try the document load.
    }
    window.location.replace(window.location.href);
  }, [alreadyReloaded]);

  if (alreadyReloaded) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-base text-white">
        <div className="max-w-md text-center space-y-4">
          <h1 className="text-2xl font-semibold">{t(I18nKey.ERROR$GENERIC)}</h1>
          <p className="text-sm text-neutral-300">{t(I18nKey.ERROR$UNKNOWN)}</p>
          <button
            type="button"
            className="rounded bg-white px-4 py-2 text-sm font-medium text-black"
            onClick={() => {
              sessionStorage.removeItem(RELOAD_SENTINEL_KEY);
              window.location.replace(window.location.href);
            }}
          >
            {t(I18nKey.LAUNCH$TRY_AGAIN)}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-base">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white" />
    </div>
  );
}
