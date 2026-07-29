import { describe, expect, it } from 'vitest';
import fs from 'fs';
import path from 'path';
import translationChecker from '../../scripts/check-translation-completeness.cjs';

const { IDENTICAL_VALUE_ALLOWLIST, getSupportedLanguageCodes, checkTranslations } =
  translationChecker;

type TranslationMap = Record<string, Record<string, string>>;

const translationJson: TranslationMap = JSON.parse(
  fs.readFileSync(
    path.join(__dirname, '../../src/i18n/translation.json'),
    'utf-8',
  ),
);

const supportedLanguageCodes = getSupportedLanguageCodes();

const getPlaceholders = (value: string) =>
  (value.match(/{{\s*[^}]+}}/g) ?? []).sort();

describe('checkTranslations', () => {
  const languages = ['en', 'fr', 'ja'];

  it('reports languages missing from a key', () => {
    const { missingTranslations } = checkTranslations(
      { 'SOME$KEY': { en: 'Hello', fr: 'Bonjour' } },
      languages,
    );

    expect(missingTranslations).toEqual({ 'SOME$KEY': ['ja'] });
  });

  it('reports languages not in the supported list', () => {
    const { extraLanguages } = checkTranslations(
      { 'SOME$KEY': { en: 'Hello', fr: 'Bonjour', ja: 'こんにちは', xx: 'Hello' } },
      languages,
    );

    expect(extraLanguages).toEqual({ 'SOME$KEY': ['xx'] });
  });

  it('flags keys where every non-English value is the English value', () => {
    const { untranslatedKeys } = checkTranslations(
      { 'SOME$KEY': { en: 'Hello', fr: 'Hello', ja: 'Hello' } },
      languages,
    );

    expect(untranslatedKeys).toEqual({ 'SOME$KEY': 'Hello' });
  });

  it('does not flag allowlisted keys with identical values', () => {
    const { untranslatedKeys } = checkTranslations(
      { BRANDING$OPENHANDS: { en: 'OpenHands', fr: 'OpenHands', ja: 'OpenHands' } },
      languages,
    );

    expect(untranslatedKeys).toEqual({});
  });

  it('does not flag keys translated in at least one language', () => {
    // Cognates legitimately match English in some languages (e.g. "Terminal"
    // in French); only the copied-everywhere pattern signals a missing
    // translation.
    const { untranslatedKeys } = checkTranslations(
      { 'SOME$KEY': { en: 'Terminal', fr: 'Terminal', ja: 'ターミナル' } },
      languages,
    );

    expect(untranslatedKeys).toEqual({});
  });
});

describe('translation.json', () => {
  it('exposes the supported language codes including English', () => {
    expect(supportedLanguageCodes).toContain('en');
    expect(supportedLanguageCodes.length).toBeGreaterThan(1);
  });

  it('has complete, translated coverage for every key', () => {
    const { missingTranslations, extraLanguages, untranslatedKeys } =
      checkTranslations(translationJson, supportedLanguageCodes);

    expect(missingTranslations).toEqual({});
    expect(extraLanguages).toEqual({});
    // Regression for #1218: English values copied across all languages.
    expect(untranslatedKeys).toEqual({});
  });

  it('has no stale entries in the identical-value allowlist', () => {
    const staleKeys = [...IDENTICAL_VALUE_ALLOWLIST].filter(
      (key) => !(key in translationJson),
    );

    expect(staleKeys).toEqual([]);
  });

  it('keeps the keys from #1218 translated', () => {
    const keys = [
      'SETTINGS$SAVE_AND_RECONNECT',
      'SETTINGS$MCP_ERROR_URL_DUPLICATE',
      'SETTINGS$MCP_ERROR_ENV_INVALID_FORMAT',
    ];

    keys.forEach((key) => {
      const translations = translationJson[key];
      expect(translations, `missing key ${key}`).toBeDefined();
      ['ja', 'zh-CN', 'fr'].forEach((lang) => {
        expect(
          translations[lang],
          `${key} is not translated for ${lang}`,
        ).not.toBe(translations.en);
      });
    });
  });

  it('preserves interpolation placeholders in every translation', () => {
    const mismatches: string[] = [];

    Object.entries(translationJson).forEach(([key, translations]) => {
      const expected = getPlaceholders(translations.en ?? '').join(',');

      Object.entries(translations).forEach(([lang, value]) => {
        if (lang === 'en') {
          return;
        }
        if (getPlaceholders(value).join(',') !== expected) {
          mismatches.push(`${key} (${lang})`);
        }
      });
    });

    expect(mismatches).toEqual([]);
  });
});
