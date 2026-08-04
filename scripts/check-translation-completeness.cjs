#!/usr/bin/env node

/**
 * Pre-commit hook script to check for translation completeness
 * This script ensures that all translation keys have entries for all supported languages
 * and that values are actually translated rather than English copied to every language.
 */

const fs = require('fs');
const path = require('path');

// Keys whose value is intentionally identical in every language (brand names,
// protocol/technical terms, placeholder-only format strings). Add a key here
// only when the English value is genuinely correct for all languages.
const IDENTICAL_VALUE_ALLOWLIST = new Set([
  'ACTION_MESSAGE$ACP_TOOL',
  'API$TAVILY_KEY_EXAMPLE',
  'API$TVLY_KEY_EXAMPLE',
  'AUTOMATIONS$DOWNLOAD_TARBALL',
  'AUTOMATIONS$CREATE_REPOSITORY_PLACEHOLDER',
  'AUTOMATIONS$CREATE_BRANCH_PLACEHOLDER',
  'AUTOMATIONS$WIZARD_AUTHOR_PLACEHOLDER',
  'AUTOMATIONS$WIZARD_EVENT_PUSH',
  'AUTOMATIONS$WIZARD_ACTION_CREATE_SCRIPT',
  'AUTOMATIONS$WIZARD_ACTION_CREATE_SCRIPT_DESC',
  'AUTOMATIONS$WIZARD_ACTION_INTRO',
  'AUTOMATIONS$WIZARD_ACTION_RUN_SCRIPT',
  'AUTOMATIONS$WIZARD_ACTION_RUN_SCRIPT_DESC',
  'AUTOMATIONS$WIZARD_ACTION_START_CONVERSATION',
  'AUTOMATIONS$WIZARD_ACTION_START_CONVERSATION_DESC',
  'AUTOMATIONS$WIZARD_ACTION_SUMMARY_CREATE_SCRIPT',
  'AUTOMATIONS$WIZARD_ACTION_SUMMARY_RUN_SCRIPT',
  'AUTOMATIONS$WIZARD_ACTION_SUMMARY_START_CONVERSATION',
  'AUTOMATIONS$WIZARD_ACTION_TYPE',
  'AUTOMATIONS$WIZARD_ADD_MORE',
  'AUTOMATIONS$WIZARD_ADVANCED_CRON',
  'AUTOMATIONS$WIZARD_CRON_PLACEHOLDER',
  'AUTOMATIONS$WIZARD_CONVERSATION_SETUP',
  'AUTOMATIONS$WIZARD_CONVERSATION_TITLE',
  'AUTOMATIONS$WIZARD_DESCRIBE_SCRIPT',
  'AUTOMATIONS$WIZARD_INTEGRATIONS_SECRETS',
  'AUTOMATIONS$WIZARD_PRESET_1M',
  'AUTOMATIONS$WIZARD_PRESET_5M',
  'AUTOMATIONS$WIZARD_PRESET_15M',
  'AUTOMATIONS$WIZARD_PRESET_1H',
  'AUTOMATIONS$WIZARD_PRESET_DAILY',
  'AUTOMATIONS$WIZARD_PRESETS',
  'AUTOMATIONS$WIZARD_QUICK_PRESETS',
  'AUTOMATIONS$WIZARD_REVIEW_DAILY',
  'AUTOMATIONS$WIZARD_REVIEW_MONTHLY',
  'AUTOMATIONS$WIZARD_REVIEW_WEEKLY',
  'AUTOMATIONS$WIZARD_RUNTIME',
  'AUTOMATIONS$WIZARD_SCHEDULE_AT_TIME',
  'AUTOMATIONS$WIZARD_SCHEDULE_DAILY',
  'AUTOMATIONS$WIZARD_SCHEDULE_FREQUENCY',
  'AUTOMATIONS$WIZARD_SCHEDULE_INTERVAL',
  'AUTOMATIONS$WIZARD_SCHEDULE_INTRO',
  'AUTOMATIONS$WIZARD_SCHEDULE_MONTHLY',
  'AUTOMATIONS$WIZARD_SCHEDULE_ON_DAY',
  'AUTOMATIONS$WIZARD_SCHEDULE_ON_DAY_OF_MONTH',
  'AUTOMATIONS$WIZARD_SCHEDULE_WEEKLY',
  'AUTOMATIONS$WIZARD_SCRIPT_CHANNELS',
  'AUTOMATIONS$WIZARD_SCRIPT_CHANNELS_PLACEHOLDER',
  'AUTOMATIONS$WIZARD_SCRIPT_CONFIGURATION',
  'AUTOMATIONS$WIZARD_SCRIPT_INPUTS',
  'AUTOMATIONS$WIZARD_SCRIPT_LANGUAGE',
  'AUTOMATIONS$WIZARD_SCRIPT_LAST_UPDATED',
  'AUTOMATIONS$WIZARD_SCRIPT_MATCH_CRITERIA',
  'AUTOMATIONS$WIZARD_SCRIPT_SECTION',
  'AUTOMATIONS$WIZARD_SELECT_SCRIPT',
  'AUTOMATIONS$WIZARD_STARTING_PROMPT',
  'AUTOMATIONS$WIZARD_SUMMARY_DAILY',
  'AUTOMATIONS$WIZARD_SUMMARY_MONTHLY',
  'AUTOMATIONS$WIZARD_SUMMARY_SCHEDULE',
  'AUTOMATIONS$WIZARD_SUMMARY_WEEKLY',
  'AUTOMATIONS$WIZARD_TRIGGER_SCHEDULE',
  'AUTOMATIONS$WIZARD_UNIT',
  'AUTOMATIONS$WIZARD_WEEKDAY_FRIDAY',
  'AUTOMATIONS$WIZARD_WEEKDAY_MONDAY',
  'AUTOMATIONS$WIZARD_WEEKDAY_SATURDAY',
  'AUTOMATIONS$WIZARD_WEEKDAY_SUNDAY',
  'AUTOMATIONS$WIZARD_WEEKDAY_THURSDAY',
  'AUTOMATIONS$WIZARD_WEEKDAY_TUESDAY',
  'AUTOMATIONS$WIZARD_WEEKDAY_WEDNESDAY',
  'AUTOMATIONS$WIZARD_WHAT_HAPPENS_NEXT',
  'BACKEND$CLOUD_TITLE',
  'BACKEND$VERSION_LABEL',
  'BRANDING$OPENHANDS',
  'COMMAND_MENU$SHORTCUT',
  'CONVERSATION$ACP_AGENT_GENERIC',
  'CONVERSATION$BUDGET_USAGE_FORMAT',
  'FILES$VSCODE',
  'GITHUB$AUTH_SCOPE',
  'LAUNCH$PLUGIN_PATH',
  'LAUNCH$PLUGIN_REF',
  'SCHEMA$LLM$SECTION_LABEL',
  'SCHEMA$LLM$TOP_K$LABEL',
  'SCHEMA$LLM$TOP_P$LABEL',
  'SCHEMA$SECURITY_ANALYZER$CHOICE$LLM',
  'SCHEMA$VERIFICATION$SECURITY_ANALYZER$CHOICE$LLM',
  'SETTINGS$AGENT_SERVER_URL_PLACEHOLDER',
  'SETTINGS$AGENT_TYPE_OPENHANDS',
  'SETTINGS$APP_UPDATE_CARD_TITLE',
  'SETTINGS$AZURE_DEVOPS',
  'SETTINGS$CLOUD_SETTINGS_LINK',
  'SETTINGS$GITHUB',
  'SETTINGS$GITLAB',
  'SETTINGS$MCP_AUTH_MODE_OAUTH',
  'SETTINGS$MCP_DEFAULT_CONFIG',
  'SETTINGS$MCP_HEADERS_PLACEHOLDER',
  'SETTINGS$MCP_OAUTH_CLIENT_ID_PLACEHOLDER',
  'SETTINGS$MCP_OAUTH_CLIENT_SECRET_PLACEHOLDER',
  'SETTINGS$MCP_OAUTH_SCOPES_PLACEHOLDER',
  'SETTINGS$MCP_SERVER_TYPE_SHTTP',
  'SETTINGS$MCP_SERVER_TYPE_SSE',
  'SETTINGS$MCP_SERVER_TYPE_STDIO',
  'SETTINGS$NAV_LLM',
  'SETTINGS$SKILLS_PILLS_MORE',
  'SETTINGS$SKILLS_VERSION',
  'SETTINGS$SLACK',
  'SETTINGS$TITLE_GENERATION_PROFILE_OPTION',
  'SETUP$REPOSITORY_PLACEHOLDER',
  'VSCODE$TITLE',
  'WORKSPACE$JUPYTER_TAB_LABEL',
]);

// Extract the language codes from the AvailableLanguages array in the i18n index file
function getSupportedLanguageCodes() {
  const i18nIndexPath = path.join(__dirname, '../src/i18n/index.ts');
  const i18nIndexContent = fs.readFileSync(i18nIndexPath, 'utf8');

  const languageCodesRegex = /\{ label: "[^"]+", value: "([^"]+)" \}/g;
  const supportedLanguageCodes = [];
  let match;

  while ((match = languageCodesRegex.exec(i18nIndexContent)) !== null) {
    supportedLanguageCodes.push(match[1]);
  }

  return supportedLanguageCodes;
}

// Check each translation key for missing languages, extra languages, and
// untranslated (English-copied) values
function checkTranslations(translationJson, supportedLanguageCodes) {
  const missingTranslations = {};
  const extraLanguages = {};
  const untranslatedKeys = {};

  const nonEnglishLanguageCodes = supportedLanguageCodes.filter(
    (langCode) => langCode !== 'en'
  );

  Object.entries(translationJson).forEach(([key, translations]) => {
    // Get the languages available for this key
    const availableLanguages = Object.keys(translations);

    // Find missing languages for this key
    const missing = supportedLanguageCodes.filter(
      (langCode) => !availableLanguages.includes(langCode)
    );

    if (missing.length > 0) {
      missingTranslations[key] = missing;
    }

    // Find extra languages for this key
    const extra = availableLanguages.filter(
      (langCode) => !supportedLanguageCodes.includes(langCode)
    );

    if (extra.length > 0) {
      extraLanguages[key] = extra;
    }

    // Flag keys where every non-English value is the English value copied
    // verbatim — a strong signal the key was never translated. Keys whose value
    // is legitimately identical everywhere belong in IDENTICAL_VALUE_ALLOWLIST.
    if (
      !IDENTICAL_VALUE_ALLOWLIST.has(key) &&
      translations.en !== undefined &&
      nonEnglishLanguageCodes.every(
        (langCode) => translations[langCode] === translations.en
      )
    ) {
      untranslatedKeys[key] = translations.en;
    }
  });

  return { missingTranslations, extraLanguages, untranslatedKeys };
}

module.exports = {
  IDENTICAL_VALUE_ALLOWLIST,
  getSupportedLanguageCodes,
  checkTranslations,
};

if (require.main === module) {
  // Load the translation file
  const translationJsonPath = path.join(__dirname, '../src/i18n/translation.json');
  const translationJson = require(translationJsonPath);

  const { missingTranslations, extraLanguages, untranslatedKeys } =
    checkTranslations(translationJson, getSupportedLanguageCodes());

  const hasErrors =
    Object.keys(missingTranslations).length > 0 ||
    Object.keys(extraLanguages).length > 0 ||
    Object.keys(untranslatedKeys).length > 0;

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

  // Generate detailed error message if there are untranslated keys
  if (Object.keys(untranslatedKeys).length > 0) {
    console.error('\x1b[31m%s\x1b[0m', 'ERROR: Untranslated keys detected');
    console.error(`Found ${Object.keys(untranslatedKeys).length} translation keys where the English value is copied to every language:`);

    Object.entries(untranslatedKeys).forEach(([key, value]) => {
      console.error(`- Key "${key}" has the same value ("${value}") for all languages`);
    });

    console.error('\nPlease translate the values before committing. If a value is intentionally identical in every language (brand name, technical term, format string), add the key to IDENTICAL_VALUE_ALLOWLIST in scripts/check-translation-completeness.cjs.');
  }

  // Exit with error code if there are issues
  if (hasErrors) {
    process.exit(1);
  } else {
    console.log('\x1b[32m%s\x1b[0m', 'All translation keys have complete language coverage!');
  }
}
