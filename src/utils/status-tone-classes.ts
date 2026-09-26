/**
 * Theme-aware status colors. Each tone resolves through semantic tokens, so
 * light palettes get dark ink on a pale tint and dark palettes keep light ink
 * on a deep tint. Prefer these over Tailwind palette literals such as
 * `text-red-300` or `bg-green-900/50`, which only work on dark backgrounds,
 * and over ad-hoc `bg-<tone>/10 text-<tone>` pairs, whose contrast depends on
 * whichever panel the badge happens to sit on.
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
