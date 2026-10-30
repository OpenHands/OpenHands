/**
 * Margin corrections that make a virtualized row lay out like the plain list.
 *
 * The plain list's row wrapper is `display: contents`, so a row's children are
 * direct children of the scroll column and `last:mb-4` fires only on the very
 * last message. A virtualized row wraps each message in its own box, so the
 * message becomes a `:last-child` of that box and its `last:mb-4` would fire on
 * every row instead of only the final one.
 *
 * Both selectors are scoped to the elements that actually carry `last:mb-4`
 * (`ChatMessage`'s bubble and `CorrectiveNudgeMessage`), so the reset does not
 * also strip unrelated margins such as `EventGroup`'s `my-1`. Written as
 * complete string literals: the production CSS scanner cannot discover a
 * candidate built from concatenated fragments.
 */

/** Plain wrapper: neutralize `last:mb-4` on every row but the last. */
export const PLAIN_ROW_MARGIN_RESET_CLASSNAME =
  '[&>*:not(:last-child)>*:last-child[class~="last:mb-4"]]:mb-0';

/** Virtual row: same correction, for the row's own single message box. */
export const VIRTUAL_ROW_MARGIN_RESET_CLASSNAME =
  '[&>*:last-child[class~="last:mb-4"]]:mb-0';
