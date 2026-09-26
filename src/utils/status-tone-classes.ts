/**
 * Theme-aware status colors for the light palettes. Call sites retain their
 * established `dark:` classes so OpenHands Neutral remains visually stable.
 * The unprefixed semantic classes repair Light+ and Solarized Light without
 * turning this accessibility follow-up into a dark-theme redesign.
 */
export type StatusTone = "success" | "warning" | "danger" | "info";

export const statusToneTextClassName: Record<StatusTone, string> = {
  success: "text-success",
  warning: "text-warning",
  danger: "text-danger",
  info: "text-info",
};

/** Compact pill: soft tone fill with an ink that clears 4.5:1 on every panel. */
export const statusToneBadgeClassName: Record<StatusTone, string> = {
  success: "bg-success-soft text-success-soft-foreground",
  warning: "bg-warning-soft text-warning-soft-foreground",
  danger: "bg-danger-soft text-danger-soft-foreground",
  info: "bg-info-soft text-info-soft-foreground",
};

/** Bordered notice panel for inline errors and warnings. */
export const statusToneBannerClassName: Record<StatusTone, string> = {
  success:
    "border border-success/40 bg-success-soft text-success-soft-foreground",
  warning:
    "border border-warning/40 bg-warning-soft text-warning-soft-foreground",
  danger: "border border-danger/40 bg-danger-soft text-danger-soft-foreground",
  info: "border border-info/40 bg-info-soft text-info-soft-foreground",
};
