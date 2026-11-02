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
});
