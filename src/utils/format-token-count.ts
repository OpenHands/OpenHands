/** Compact token counts for context-window UI (e.g. 198.5k, 1.0M). */
export function formatCompactTokenCount(value: number): string {
  if (value >= 1_000_000) {
    const millions = value / 1_000_000;
    if (millions >= 10 && Number.isInteger(millions)) {
      return `${millions.toFixed(0)}M`;
    }
    return `${millions.toFixed(1)}M`;
  }

  if (value >= 1_000) {
    const thousands = value / 1_000;
    if (Number.isInteger(thousands)) {
      return `${thousands.toFixed(0)}k`;
    }
    const rounded = thousands.toFixed(1);
    // Rounding can carry the value into the next unit (999_999 -> "1000.0k"),
    // which reads as a four-digit count. Promote it to M instead.
    if (Number(rounded) >= 1_000) {
      return `${(Number(rounded) / 1_000).toFixed(1)}M`;
    }
    return `${rounded}k`;
  }

  return value.toLocaleString();
}

export function getContextWindowUsagePercentage(
  perTurnToken: number,
  contextWindow: number,
): number {
  if (contextWindow <= 0) {
    return 0;
  }

  return Math.min(100, (perTurnToken / contextWindow) * 100);
}
