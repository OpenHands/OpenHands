/**
 * Shared raised-pill class for conversation-card metadata chips (LLM model,
 * tags, repo, branch, workspace folder). Keep these visually identical so the
 * footer reads as one chip system.
 */
export const CONVERSATION_CARD_META_CHIP_CLASSNAME =
  "inline-flex max-w-full min-w-0 shrink-0 items-center gap-0.5 rounded-sm bg-[var(--oh-surface-raised)] px-1 py-px text-[10px] leading-4 text-[var(--oh-muted)]";

/**
 * Fixed line-height icon slot matching ``leading-4`` chip text so Lucide /
 * react-icons / local SVGs share the same optical center.
 */
export const CONVERSATION_CARD_META_CHIP_ICON_SLOT_CLASSNAME =
  "inline-flex h-4 w-3 shrink-0 items-center justify-center [&_svg]:block";

export const CONVERSATION_CARD_META_CHIP_ICON_CLASSNAME = "h-3 w-3";
