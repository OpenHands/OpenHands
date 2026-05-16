/**
 * Split a command line into tokens, respecting single/double quotes so
 * paths and arguments with spaces survive a round-trip — e.g.
 * `npx -y "my pkg"` becomes `["npx", "-y", "my pkg"]`.
 *
 * Intentionally narrow: no shell expansion, no backslash escapes, no
 * variable substitution. Just enough to keep the textarea honest about
 * what the backend will receive.
 */
export function tokenizeCommand(value: string): string[] {
  const tokens: string[] = [];
  let current = "";
  let inToken = false;
  let quote: '"' | "'" | null = null;

  for (let i = 0; i < value.length; i += 1) {
    const ch = value[i];
    if (quote !== null) {
      if (ch === quote) {
        quote = null;
      } else {
        current += ch;
      }
    } else if (ch === '"' || ch === "'") {
      quote = ch;
      inToken = true;
    } else if (/\s/.test(ch)) {
      if (inToken) {
        tokens.push(current);
        current = "";
        inToken = false;
      }
    } else {
      current += ch;
      inToken = true;
    }
  }
  if (inToken) tokens.push(current);
  return tokens;
}

/**
 * Format tokens back into a command line. Tokens with whitespace are
 * wrapped in double quotes (single quotes if the token already contains
 * a double quote) so `tokenizeCommand` parses them back to the same list.
 */
export function formatCommand(command: string[]): string {
  return command
    .map((part) => {
      if (part === "") return '""';
      if (!/\s/.test(part)) return part;
      return part.includes('"') ? `'${part}'` : `"${part}"`;
    })
    .join(" ");
}
