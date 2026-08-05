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
  | "other";

/**
 * Map a server tag key to a hovercard label kind. ``archiveworkspacepath`` is
 * treated as "workspace" — the sandbox working directory for the conversation.
 */
export function getConversationTagLabelKind(
  key: string,
): ConversationTagLabelKind {
  switch (key.trim().toLowerCase()) {
    case "git_provider":
    case "origin":
    case "source":
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

export interface PreviewTagCoverage {
  hasRepository: boolean;
  hasBranch: boolean;
  hasDirectory: boolean;
  hasGitProvider: boolean;
}

/**
 * Drop tags that duplicate fields already rendered in the conversation
 * hovercard (repository / branch / directory / provider), so the list stays
 * one row per fact.
 */
export function filterPreviewConversationTags(
  tags: Array<[string, string]>,
  coverage: PreviewTagCoverage,
): Array<[string, string]> {
  return tags.filter(([key]) => {
    const kind = getConversationTagLabelKind(key);
    if (kind === "git" && coverage.hasGitProvider) {
      return false;
    }
    if (kind === "repo" && coverage.hasRepository) {
      return false;
    }
    if (kind === "branch" && coverage.hasBranch) {
      return false;
    }
    if (kind === "workspace" && coverage.hasDirectory) {
      return false;
    }
    return true;
  });
}

/**
 * Hard-truncate a tag value for the chip label. The full ``key: value`` string
 * stays available via tooltip / overflow popover.
 */
export function truncateTagChipValue(
  value: string,
  maxLength: number = TAG_CHIP_VALUE_MAX_LENGTH,
): string {
  if (value.length <= maxLength) {
    return value;
  }
  if (maxLength <= 1) {
    return "…";
  }
  return `${value.slice(0, maxLength - 1)}…`;
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
