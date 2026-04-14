/**
 * Coerce unknown API values to string[] for list endpoints (models, agents, etc.).
 */
export function normalizeStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string");
}
