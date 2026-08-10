/**
 * Display helpers for conversation tag chips. Pure (no React) so fit/truncate
 * behavior can be unit-tested without laying out the sidebar card.
 */

import { I18nKey } from "#/i18n/declaration";

/** Max characters shown on a chip before hard truncation with an ellipsis. */
export const TAG_CHIP_VALUE_MAX_LENGTH = 14;

/** Horizontal gap between chips (matches Tailwind ``gap-1`` = 4px). */
export const TAG_CHIP_GAP_PX = 4;

/**
 * Reserved width for the ``+N`` overflow control when deciding how many chips
 * fit on one row. Slightly generous so the button never wraps under a chip.
 */
export const TAG_CHIP_OVERFLOW_WIDTH_PX = 36;

/** Known tag keys that already have a dedicated hovercard label. */
export type ConversationTagLabelKind =
  | "git"
  | "repo"
  | "branch"
  | "workspace"
  | "app_mode"
  | "work_tools"
  | "work_wsid"
  | "other";

/**
 * Map a server tag key to a hovercard label kind. ``archiveworkspacepath`` is
 * treated as "workspace" — the sandbox working directory for the conversation.
 * ACM / Work stamps (``Appmode``, ``Worktools``, ``Workwsid``) get dedicated
 * kinds so chips and hovercards show friendly labels instead of the wire key.
 *
 * ``origin`` / ``source`` deliberately stay "other" (humanized to "Origin" /
 * "Source"): they name where a conversation came from — Slack, an API call, an
 * automation — which is not a git fact. Only ``git_provider`` is "Git".
 */
export function getConversationTagLabelKind(
  key: string,
): ConversationTagLabelKind {
  switch (key.trim().toLowerCase()) {
    case "git_provider":
      return "git";
    case "repo_name":
    case "repo":
    case "repository":
      return "repo";
    case "selected_branch":
    case "branch":
      return "branch";
    case "archiveworkspacepath":
    case "workspace":
    case "working_dir":
      return "workspace";
    case "appmode":
    case "app_mode":
    case "mode":
      return "app_mode";
    case "worktools":
    case "work_tools":
    case "tools":
      return "work_tools";
    case "workwsid":
    case "work_wsid":
    case "wsid":
    case "workspace_id":
      return "work_wsid";
    default:
      return "other";
  }
}

/**
 * Soften unknown snake_case / kebab-case keys for tooltips and overflow rows
 * (``selected_branch`` is mapped above; this covers free-form keys like
 * ``env`` → ``Env``).
 */
export function humanizeConversationTagKey(key: string): string {
  const trimmed = key.trim();
  if (!trimmed) {
    return trimmed;
  }
  const words = trimmed.replace(/[_-]+/g, " ").replace(/\s+/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/**
 * Localized label for a tag key (chip tooltip, overflow popover, hovercard).
 * Known keys use the preview copy; everything else is humanized.
 */
export function getConversationTagLabel(
  key: string,
  t: (key: I18nKey) => string,
): string {
  switch (getConversationTagLabelKind(key)) {
    case "git":
      return t(I18nKey.CONVERSATION_PANEL$PREVIEW_GIT);
    case "repo":
      return t(I18nKey.CONVERSATION_PANEL$PREVIEW_REPO);
    case "branch":
      return t(I18nKey.CONVERSATION_PANEL$PREVIEW_BRANCH);
    case "workspace":
      return t(I18nKey.CONVERSATION_PANEL$PREVIEW_WORKSPACE);
    case "app_mode":
      return t(I18nKey.CONVERSATION_PANEL$PREVIEW_APP_MODE);
    case "work_tools":
      return t(I18nKey.CONVERSATION_PANEL$PREVIEW_WORK_TOOLS);
    case "work_wsid":
      return t(I18nKey.CONVERSATION_PANEL$PREVIEW_WORK_WSID);
    default:
      return humanizeConversationTagKey(key);
  }
}

/** ``Branch: main`` — used by chip ``title`` tooltips. */
export function formatConversationTagTooltip(
  key: string,
  value: string,
  t: (key: I18nKey) => string,
): string {
  return `${getConversationTagLabel(key, t)}: ${value}`;
}

/**
 * Hard-truncate a tag value for the chip label. The full ``key: value`` string
 * stays available via tooltip / overflow popover.
 *
 * Measures and slices by code point rather than UTF-16 code unit: values
 * stamped by Slack / Discord automations carry emoji, and cutting between the
 * halves of a surrogate pair leaves a lone surrogate that browsers draw as a
 * replacement glyph.
 */
export function truncateTagChipValue(
  value: string,
  maxLength: number = TAG_CHIP_VALUE_MAX_LENGTH,
): string {
  const characters = Array.from(value);
  if (characters.length <= maxLength) {
    return value;
  }
  if (maxLength <= 1) {
    return "…";
  }
  return `${characters.slice(0, maxLength - 1).join("")}…`;
}

/**
 * How many chips fit in ``containerWidth`` while keeping a single nowrap row
 * and reserving space for a ``+N`` overflow control when any chips would hide.
 *
 * Returns ``widths.length`` when ``containerWidth <= 0`` (not laid out yet /
 * jsdom) so callers can show every chip until a real measurement arrives.
 * Returns ``0`` when the row is too narrow for even one chip + overflow — the
 * UI then shows only the ``+N`` control with the full list in the popover.
 */
export function computeVisibleTagChipCount(
  widths: number[],
  containerWidth: number,
  options: {
    gapPx?: number;
    overflowWidthPx?: number;
  } = {},
): number {
  const gapPx = options.gapPx ?? TAG_CHIP_GAP_PX;
  const overflowWidthPx = options.overflowWidthPx ?? TAG_CHIP_OVERFLOW_WIDTH_PX;

  if (widths.length === 0) {
    return 0;
  }
  if (containerWidth <= 0) {
    return widths.length;
  }

  let used = 0;
  for (let i = 0; i < widths.length; i += 1) {
    const width = widths[i]!;
    const gap = i > 0 ? gapPx : 0;
    const remaining = widths.length - i - 1;
    const reserve = remaining > 0 ? overflowWidthPx + gapPx : 0;
    if (used + gap + width + reserve > containerWidth) {
      return i;
    }
    used += gap + width;
  }
  return widths.length;
}
