import i18next from "i18next";
import { describe, expect, it } from "vitest";

import translations from "#/i18n/translation.json";
import { I18nKey } from "#/i18n/declaration";

type Translations = Record<string, Record<string, string>>;

const KEY = I18nKey.RECOMMENDED_AUTOMATIONS$MISSING_CONNECT;

async function missingConnect(lng: string, count: number) {
  const resources: Record<string, { translation: Record<string, string> }> = {};
  for (const [key, byLang] of Object.entries(translations as Translations)) {
    if (!key.startsWith(KEY)) continue;
    for (const [lang, value] of Object.entries(byLang)) {
      resources[lang] ??= { translation: {} };
      resources[lang].translation[key] = value;
    }
  }
  const instance = i18next.createInstance();
  await instance.init({ lng, fallbackLng: "en", resources });
  return instance.t(KEY, { count });
}

describe("RECOMMENDED_AUTOMATIONS$MISSING_CONNECT plurals", () => {
  it("uses the singular for one MCP in English", async () => {
    expect(await missingConnect("en", 1)).toBe(
      "1 MCP to connect before launch",
    );
    expect(await missingConnect("en", 2)).toBe(
      "2 MCPs to connect before launch",
    );
  });

  it("never falls back to English for Ukrainian or Arabic plural forms", async () => {
    const english = await missingConnect("en", 3);
    expect(await missingConnect("uk", 3)).not.toBe(english);
    expect(await missingConnect("ar", 3)).not.toBe(english);
  });
});
