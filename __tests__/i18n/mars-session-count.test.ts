import i18next from "i18next";
import { describe, expect, it } from "vitest";

import translations from "#/i18n/translation.json";
import { SESSION_COUNT_I18N_KEY } from "#/components/features/backends/managed-agents/managed-agents-labels";

type Translations = Record<string, Record<string, string>>;

async function sessionCount(lng: string, count: number) {
  const resources: Record<string, { translation: Record<string, string> }> = {};
  for (const [key, byLang] of Object.entries(translations as Translations)) {
    if (!key.startsWith(SESSION_COUNT_I18N_KEY)) continue;
    for (const [lang, value] of Object.entries(byLang)) {
      resources[lang] ??= { translation: {} };
      resources[lang].translation[key] = value;
    }
  }
  const instance = i18next.createInstance();
  await instance.init({ lng, fallbackLng: "en", resources });
  return instance.t(SESSION_COUNT_I18N_KEY, { count });
}

describe("DO_AGENTS$SESSION_COUNT plurals", () => {
  it("uses the singular for one session in English", async () => {
    expect(await sessionCount("en", 1)).toBe("1 session");
    expect(await sessionCount("en", 2)).toBe("2 sessions");
  });

  it("never falls back to English for Ukrainian or Arabic plural forms", async () => {
    expect(await sessionCount("uk", 1)).toBe("1 сесія");
    expect(await sessionCount("uk", 3)).toBe("3 сесії");
    expect(await sessionCount("uk", 5)).toBe("5 сесій");
    expect(await sessionCount("ar", 2)).toBe("2 جلستان");
    expect(await sessionCount("ar", 3)).toBe("3 جلسات");
    expect(await sessionCount("ar", 11)).toBe("11 جلسة");
  });

  it("keeps French zero singular", async () => {
    expect(await sessionCount("fr", 0)).toBe("0 session");
  });
});
