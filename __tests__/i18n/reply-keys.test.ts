import { describe, expect, it } from 'vitest';
import fs from 'fs';
import path from 'path';
import translationChecker from '../../scripts/check-translation-completeness.cjs';

const { getSupportedLanguageCodes } = translationChecker;

type TranslationMap = Record<string, Record<string, string>>;

const translationJson: TranslationMap = JSON.parse(
  fs.readFileSync(
    path.join(__dirname, '../../src/i18n/translation.json'),
    'utf-8',
  ),
);

const supportedLanguageCodes = getSupportedLanguageCodes();

const REPLY_KEYS = [
  'CHAT_INTERFACE$REPLY',
  'CHAT_INTERFACE$REPLY_YOU',
  'CHAT_INTERFACE$REPLY_AGENT',
  'CHAT_INTERFACE$REPLY_TRUNCATED',
  'CHAT_INTERFACE$REPLY_DISMISS',
  'CHAT_INTERFACE$REPLY_NOT_LOADED',
];

describe('reply-message i18n keys', () => {
  it('defines every reply key for all supported languages', () => {
    REPLY_KEYS.forEach((key) => {
      const translations = translationJson[key];
      expect(translations, `missing key ${key}`).toBeDefined();

      const missing = supportedLanguageCodes.filter(
        (langCode) => !(langCode in translations),
      );
      expect(missing, `${key} missing languages`).toEqual([]);
    });
  });

  it('translates every reply key into non-English languages', () => {
    REPLY_KEYS.forEach((key) => {
      const translations = translationJson[key];
      const copied = supportedLanguageCodes
        .filter((langCode) => langCode !== 'en')
        .filter((langCode) => translations[langCode] === translations.en);
      expect(copied, `${key} copied English into: ${copied.join(', ')}`).toEqual(
        [],
      );
    });
  });

  it('includes every reply key in the generated I18nKey enum', () => {
    // eslint-disable-next-line import/no-relative-packages
    const declaration = fs.readFileSync(
      path.join(__dirname, '../../src/i18n/declaration.ts'),
      'utf-8',
    );

    REPLY_KEYS.forEach((key) => {
      expect(declaration).toContain(`  ${key} = "${key}",`);
    });
  });
});
