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
    return `${thousands.toFixed(1)}k`;
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
