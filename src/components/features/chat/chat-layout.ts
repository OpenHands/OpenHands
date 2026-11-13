export const CHAT_BODY_CLASS_NAME =
  "relative flex h-full min-h-0 flex-col justify-between px-4";

// The trailing `pb-12` replaces the `last:mb-4` the message rows used to carry:
// a row is now the `:last-child` of its own wrapper, so that margin fired on
// every row rather than only the final message. The column's own bottom padding
// gives the last row the same 16px of breathing room (the old margin) plus the
// existing 32px, whatever trails the list.
export const CHAT_SCROLL_CLASS_NAME =
  "custom-scrollbar-always flex min-h-0 grow flex-col gap-2 overflow-x-hidden overflow-y-auto px-0 pt-4 pb-12 md:px-4";

export const CHAT_COMPOSER_CLASS_NAME = "flex shrink-0 flex-col gap-1.5 pb-4";
