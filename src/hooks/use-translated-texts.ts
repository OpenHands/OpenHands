import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  resolveUiTranslateTarget,
  TranslateService,
} from "#/api/integrations/translate-service";

/**
 * Translate a list of English strings into the active UI language.
 * While the request is in flight (or on failure), callers keep showing the
 * original text — never blank the security finding list.
 */
export function useTranslatedTexts(texts: readonly string[]): {
  translations: Map<string, string>;
  isTranslating: boolean;
} {
  const { i18n } = useTranslation();
  const target = resolveUiTranslateTarget(i18n.language);
  const fingerprint = useMemo(
    () =>
      [...new Set(texts.map((t) => t.trim()).filter(Boolean))]
        .sort()
        .join("\0"),
    [texts],
  );

  const [translations, setTranslations] = useState<Map<string, string>>(
    () => new Map(),
  );
  const [isTranslating, setIsTranslating] = useState(false);

  useEffect(() => {
    if (!target || !fingerprint) {
      setTranslations(new Map());
      setIsTranslating(false);
      return;
    }

    let cancelled = false;
    const sourceTexts = fingerprint.split("\0");
    setIsTranslating(true);

    void TranslateService.translateBatch(sourceTexts, i18n.language)
      .then((map) => {
        if (!cancelled) setTranslations(map);
      })
      .catch(() => {
        // Keep originals — translation is best-effort for client UX.
        if (!cancelled) setTranslations(new Map());
      })
      .finally(() => {
        if (!cancelled) setIsTranslating(false);
      });

    return () => {
      cancelled = true;
    };
  }, [fingerprint, i18n.language, target]);

  return { translations, isTranslating };
}

/** Resolve display text: translated when available, otherwise original. */
export function displayTranslatedText(
  original: string,
  translations: Map<string, string>,
): string {
  const trimmed = original.trim();
  return translations.get(trimmed) ?? translations.get(original) ?? original;
}
