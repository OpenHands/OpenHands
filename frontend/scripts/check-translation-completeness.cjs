#!/usr/bin/env node

/**
 * Pre-commit hook script to check for translation completeness
 * This script ensures that all translation keys have entries for all supported languages
 */

const fs = require('fs');
const path = require('path');

// Load the translation file
const translationJsonPath = path.join(__dirname, '../src/i18n/translation.json');
const translationJson = require(translationJsonPath);

// Load the available languages from the i18n index file
const i18nIndexPath = path.join(__dirname, '../src/i18n/index.ts');
const i18nIndexContent = fs.readFileSync(i18nIndexPath, 'utf8');

// Extract the language codes from the AvailableLanguages array
const languageCodesRegex = /\{ label: "[^"]+", value: "([^"]+)" \}/g;
const supportedLanguageCodes = [];
let match;

while ((match = languageCodesRegex.exec(i18nIndexContent)) !== null) {
  supportedLanguageCodes.push(match[1]);
}

// Track missing and extra translations
const missingTranslations = {};
const extraLanguages = {};
let hasErrors = false;

// Check each translation key
Object.entries(translationJson).forEach(([key, translations]) => {
  // Get the languages available for this key
  const availableLanguages = Object.keys(translations);

  // Find missing languages for this key
  // 找出该 key 缺失的语言代码
  const missing = supportedLanguageCodes.filter(
    (langCode) => !availableLanguages.includes(langCode)
  );

  if (missing.length > 0) {
    missingTranslations[key] = missing;
    hasErrors = true;
  }

  // Find extra languages for this key
  const extra = availableLanguages.filter(
    (langCode) => !supportedLanguageCodes.includes(langCode)
  );

  if (extra.length > 0) {
    extraLanguages[key] = extra;
    hasErrors = true;
  }
});

// Generate detailed error message if there are missing translations
if (Object.keys(missingTranslations).length > 0) {
  console.error('\x1b[31m%s\x1b[0m', 'ERROR: Missing translations detected');
  console.error(`Found ${Object.keys(missingTranslations).length} translation keys with missing languages:`);

  Object.entries(missingTranslations).forEach(([key, langs]) => {
    console.error(`- Key "${key}" is missing translations for: ${langs.join(', ')}`);
  });

  console.error('\nPlease add the missing translations before committing.');
}

// Generate detailed error message if there are extra languages
if (Object.keys(extraLanguages).length > 0) {
  console.error('\x1b[31m%s\x1b[0m', 'ERROR: Extra languages detected');
  console.error(`Found ${Object.keys(extraLanguages).length} translation keys with extra languages not in AvailableLanguages:`);

  Object.entries(extraLanguages).forEach(([key, langs]) => {
    console.error(`- Key "${key}" has translations for unsupported languages: ${langs.join(', ')}`);
  });

  console.error('\nPlease remove the extra languages before committing.');
}

// Exit with error code if there are issues
if (hasErrors) {
  process.exit(1);
} else {
  console.log('\x1b[32m%s\x1b[0m', 'All translation keys have complete language coverage!');
}
