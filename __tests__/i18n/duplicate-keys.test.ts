import { describe, expect, it } from 'vitest';
import fs from 'fs';
import path from 'path';

describe('translation.json', () => {
  it('should not have duplicate translation keys', () => {
    // Read the translation.json file
    const translationPath = path.join(__dirname, '../../src/i18n/translation.json');
    const translationContent = fs.readFileSync(translationPath, 'utf-8');

    // First, let's check for exact string matches of key definitions
    const keyRegex = /"([^"]+)": {/g;
    const matches = translationContent.matchAll(keyRegex);
    const keyOccurrences = new Map<string, number>();
    const duplicateKeys: string[] = [];

    for (const match of matches) {
      const key = match[1];
      const count = (keyOccurrences.get(key) || 0) + 1;
      keyOccurrences.set(key, count);
      if (count > 1) {
        duplicateKeys.push(key);
      }
    }

    // Remove duplicates from duplicateKeys array
    const uniqueDuplicates = [...new Set(duplicateKeys)];

    // If there are duplicates, create a helpful error message
    if (uniqueDuplicates.length > 0) {
      const errorMessage = `Found duplicate translation keys:\n${uniqueDuplicates
        .map((key) => `  - "${key}" appears ${keyOccurrences.get(key)} times`)
        .join('\n')}`;
      throw new Error(errorMessage);
    }

    // Expect no duplicates (this will pass if we reach here)
    expect(uniqueDuplicates).toHaveLength(0);
  });

  it('should have consistent translations for each key', () => {
    // Read the translation.json file
    const translationPath = path.join(__dirname, '../../src/i18n/translation.json');
    const translationContent = fs.readFileSync(translationPath, 'utf-8');
    const translations = JSON.parse(translationContent);

    // Create a map to store English translations for each key
    const englishTranslations = new Map<string, string>();
    const inconsistentKeys: string[] = [];

    // Check each key's English translation
    Object.entries(translations).forEach(([key, value]: [string, any]) => {
      if (typeof value === 'object' && value.en !== undefined) {
        const currentEn = value.en.toLowerCase();
        const existingEn = englishTranslations.get(key)?.toLowerCase();

        if (existingEn !== undefined && existingEn !== currentEn) {
          inconsistentKeys.push(key);
        } else {
          englishTranslations.set(key, value.en);
        }
      }
    });

    // If there are inconsistencies, create a helpful error message
    if (inconsistentKeys.length > 0) {
      const errorMessage = `Found inconsistent translations for keys:\n${inconsistentKeys
        .map((key) => `  - "${key}" has multiple different English translations`)
        .join('\n')}`;
      throw new Error(errorMessage);
    }

    // Expect no inconsistencies
    expect(inconsistentKeys).toHaveLength(0);
  });
});
