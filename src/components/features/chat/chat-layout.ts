export const CHAT_BODY_CLASS_NAME =
  "relative flex h-full min-h-0 flex-col justify-between px-4";

// The message rows used to carry `last:mb-4` for the final message's breathing
// room. A row is now the `:last-child` of its own wrapper, so that margin would
// fire on every row; the chat's scroll column adds it back as extra bottom
// padding instead (see `CHAT_SCROLL_BOTTOM_PADDING_CLASS_NAME`). The shared
// class keeps only the padding the launch view already had.
export const CHAT_SCROLL_CLASS_NAME =
  "custom-scrollbar-always flex min-h-0 grow flex-col gap-2 overflow-x-hidden overflow-y-auto px-0 pt-4 pb-8 md:px-4";

// The message list adds the old `last:mb-4` (16px) on top of the column's own
// `pb-8`, whatever trails the list. Applied only where the message list renders,
// not on the launch view, which never had the per-message margin.
export const CHAT_SCROLL_BOTTOM_PADDING_CLASS_NAME = "pb-12";

export const CHAT_COMPOSER_CLASS_NAME = "flex shrink-0 flex-col gap-1.5 pb-4";
