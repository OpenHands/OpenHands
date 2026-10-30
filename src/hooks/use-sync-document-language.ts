import React from "react";
import i18n from "#/i18n";

/**
 * Mirrors the active UI language onto `<html lang>` and `<html dir>` so
 * screen readers switch voice and right-to-left scripts get `dir="rtl"`.
 * Standalone app only: an embedding host owns its own document element.
 */
export function useSyncDocumentLanguage() {
  React.useEffect(() => {
    const applyLanguage = (language: string) => {
      const root = document.documentElement;
      root.lang = language;
      root.dir = i18n.dir(language);
    };

    if (i18n.language) {
      applyLanguage(i18n.language);
    }
    i18n.on("languageChanged", applyLanguage);
    return () => {
      i18n.off("languageChanged", applyLanguage);
    };
  }, []);
}
