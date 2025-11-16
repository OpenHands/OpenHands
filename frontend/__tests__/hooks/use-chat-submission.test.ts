import { describe, it, expect } from 'vitest';

/**
 * Test the regex for stripping @ from file paths while preserving code decorators
 */
describe('useChatSubmission @ stripping regex', () => {
  // The regex from use-chat-submission.ts
  const stripFilePathAt = (message: string) => {
    return message.replace(
      /(^|\s)@((?:\.\/|\.\.\/|~\/)[^\s]*|[^\s]*\/[^\s]*|[^\s]+\.(?:ts|tsx|js|jsx|py|java|cpp|c|h|hpp|cs|rb|go|rs|md|txt|json|yaml|yml|xml|html|css|scss|sass|less|vue|svelte)(?:\s|$))/gi,
      "$1$2"
    );
  };

  describe('should strip @ from file paths', () => {
    it('strips @ from paths with slashes', () => {
      expect(stripFilePathAt('@src/components/Button.tsx')).toBe('src/components/Button.tsx');
      expect(stripFilePathAt('Check @path/to/file.py for details')).toBe('Check path/to/file.py for details');
      expect(stripFilePathAt('@OpenHands/frontend/app.tsx')).toBe('OpenHands/frontend/app.tsx');
    });

    it('strips @ from relative paths', () => {
      expect(stripFilePathAt('@./file.ts')).toBe('./file.ts');
      expect(stripFilePathAt('@../parent/file.py')).toBe('../parent/file.py');
      expect(stripFilePathAt('@~/home/user/file.rb')).toBe('~/home/user/file.rb');
    });

    it('strips @ from files with extensions', () => {
      expect(stripFilePathAt('@file.py')).toBe('file.py');
      expect(stripFilePathAt('@component.tsx')).toBe('component.tsx');
      expect(stripFilePathAt('Look at @README.md')).toBe('Look at README.md');
      expect(stripFilePathAt('@config.json')).toBe('config.json');
    });

    it('handles multiple @ mentions in one message', () => {
      expect(stripFilePathAt('Check @src/file.ts and @docs/README.md')).toBe('Check src/file.ts and docs/README.md');
      expect(stripFilePathAt('@./local.py and @../parent.js')).toBe('./local.py and ../parent.js');
    });
  });

  describe('should preserve @ in code decorators and annotations', () => {
    it('preserves Python decorators', () => {
      expect(stripFilePathAt('@property')).toBe('@property');
      expect(stripFilePathAt('@dataclass')).toBe('@dataclass');
      expect(stripFilePathAt('@staticmethod')).toBe('@staticmethod');
      expect(stripFilePathAt('@lru_cache')).toBe('@lru_cache');
      expect(stripFilePathAt('Use @property decorator')).toBe('Use @property decorator');
    });

    it('preserves Java/TypeScript annotations', () => {
      expect(stripFilePathAt('@Override')).toBe('@Override');
      expect(stripFilePathAt('@Component')).toBe('@Component');
      expect(stripFilePathAt('@Injectable')).toBe('@Injectable');
      expect(stripFilePathAt('@Deprecated')).toBe('@Deprecated');
    });

    it('preserves @ in email addresses', () => {
      expect(stripFilePathAt('Contact alona@gmail.com')).toBe('Contact alona@gmail.com');
      expect(stripFilePathAt('Email: user@example.org')).toBe('Email: user@example.org');
    });

    it('preserves @ in social media handles', () => {
      expect(stripFilePathAt('@username')).toBe('@username');
      expect(stripFilePathAt('Follow @johndoe')).toBe('Follow @johndoe');
    });
  });

  describe('handles mixed content correctly', () => {
    it('strips file paths but preserves decorators in code snippets', () => {
      const input = `
Look at @src/models/user.py for the implementation.

\`\`\`python
@dataclass
class User:
    @property
    def name(self):
        return self._name
\`\`\`

Also check @docs/README.md`;

      const expected = `
Look at src/models/user.py for the implementation.

\`\`\`python
@dataclass
class User:
    @property
    def name(self):
        return self._name
\`\`\`

Also check docs/README.md`;

      expect(stripFilePathAt(input)).toBe(expected);
    });

    it('handles edge cases correctly', () => {
      // @ at end of line shouldn't match
      expect(stripFilePathAt('Just @')).toBe('Just @');
      // @ with numbers only shouldn't match
      expect(stripFilePathAt('@123')).toBe('@123');
      // @ with special chars shouldn't match unless it's a path
      expect(stripFilePathAt('@#$%')).toBe('@#$%');
    });
  });
});
