const MASKED_PLACEHOLDER = "<secret-hidden>";

/**
 * Defensive backstop: redact any unmasked value inside a `<CUSTOM_SECRETS>`
 * block before showing dynamic context in the UI, in case backend masking
 * regresses. Text outside the block is untouched.
 */
export function redactCustomSecrets(text: string): string {
  // Regexes are local: the `g` flag makes them stateful, and keeping them
  // scoped to this call avoids cross-call `lastIndex` surprises.

  // Closing tag is optional so a truncated block is still redacted, not leaked.
  const customSecretsBlock =
    /(<CUSTOM_SECRETS>)([\s\S]*?)(<\/CUSTOM_SECRETS>|$)/gi;

  // `KEY: value` / `KEY=value`, capturing key + separator so only the value changes.
  const secretLine = /^(\s*[^=:\n]+?\s*[:=]\s*)(.+?)\s*$/gm;

  return text.replace(
    customSecretsBlock,
    (_match, open: string, body: string, close: string) => {
      const redactedBody = body.replace(
        secretLine,
        (lineMatch, prefix: string, value: string) =>
          value === MASKED_PLACEHOLDER
            ? lineMatch
            : `${prefix}${MASKED_PLACEHOLDER}`,
      );
      return `${open}${redactedBody}${close}`;
    },
  );
}
