const CONTENT_EDITABLE_FORMATTING_CHARS = /[\u200B-\u200D\u2060\uFEFF]/g;

/** Remove invisible formatting characters that contentEditable can retain. */
export const stripSlashCommandFormatting = (text: string): string =>
  text.replace(CONTENT_EDITABLE_FORMATTING_CHARS, "");

/** Normalize an exact slash-command submission without changing normal text. */
export const normalizeUiCommand = (message: string): string =>
  stripSlashCommandFormatting(message).trim();
