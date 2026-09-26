/**
 * Theme-aware status colors. Each tone resolves through semantic tokens, so
 * light palettes get dark ink on a pale tint and dark palettes keep light ink
 * on a deep tint. Prefer these over Tailwind palette literals such as
 * `text-red-300` or `bg-green-900/50`, which only work on dark backgrounds.
 */
export type StatusTone = "success" | "warning" | "danger" | "info";

export const statusToneTextClassName: Record<StatusTone, string> = {
  success: "text-success",
  warning: "text-warning",
  danger: "text-danger",
  info: "text-info",
};

/** Compact pill: 10% tinted fill with tone-colored text. */
export const statusToneBadgeClassName: Record<StatusTone, string> = {
  success: "bg-success/10 text-success",
  warning: "bg-warning/10 text-warning",
  danger: "bg-danger/10 text-danger",
  info: "bg-info/10 text-info",
};

/** Bordered notice panel for inline errors and warnings. */
export const statusToneBannerClassName: Record<StatusTone, string> = {
  success: "border border-success/40 bg-success/10 text-success",
  warning: "border border-warning/40 bg-warning/10 text-warning",
  danger: "border border-danger/40 bg-danger/10 text-danger",
  info: "border border-info/40 bg-info/10 text-info",
};
