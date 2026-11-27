// Translation resources for downstream library consumers.
//
// This module is intentionally separate from `src/i18n/index.ts` so that the
// 1 MB `translation.json` does not enter the app's eager dev/prod graph. The
// app initialises i18next with `i18next-http-backend`, which fetches
// `/locales/<lng>/openhands.json` at runtime; it never needs the embedded
// resources object.
//
// Library consumers (`@openhands/agent-canvas/i18n`) can still import
// `translationResources` from `src/i18n/index.ts`, which re-exports from here.
// The re-export is `export … from`, which rollup is able to drop when the
// app build does not reference `translationResources`.
import translationDefinitions from "./translation.json";

type TranslationDefinitionMap = Record<string, Partial<Record<string, string>>>;
export type TranslationResources = Record<string, Record<string, string>>;

const buildTranslationResources = (
  definitions: TranslationDefinitionMap,
): TranslationResources => {
  const resources: TranslationResources = {};

  Object.entries(definitions).forEach(([key, translations]) => {
    Object.entries(translations).forEach(([language, value]) => {
      if (typeof value !== "string") {
        return;
      }

      if (!resources[language]) {
        resources[language] = {};
      }

      resources[language][key] = value;
    });
  });

  return resources;
};

// `/* @__PURE__ */` lets rollup drop both this call *and* the
// `./translation.json` import when no consumer references
// `translationResources` — i.e. the standard app build path.
export const translationResources = /* @__PURE__ */ buildTranslationResources(
  translationDefinitions as TranslationDefinitionMap,
);
