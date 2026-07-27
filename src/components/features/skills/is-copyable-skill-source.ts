/** True when `source` is a filesystem path or URL worth copying (not a scope label like "global"). */
export function isCopyableSkillSource(
  source: string | null | undefined,
): boolean {
  const trimmed = source?.trim();
  if (!trimmed) return false;

  if (/^https?:\/\//i.test(trimmed)) return true;
  if (
    trimmed.startsWith("/") ||
    trimmed.startsWith("./") ||
    trimmed.startsWith("../") ||
    trimmed.startsWith("~")
  ) {
    return true;
  }
  if (/^[A-Za-z]:[\\/]/.test(trimmed)) return true;
  if (trimmed.includes("/") || trimmed.includes("\\")) return true;

  return false;
}
