import type { ProfileScopeMode } from "#/constants/profile-scope";
import type { SettingsValue } from "#/types/settings";

/** A tool spec as stored on an agent profile and sent on the wire. */
export type ProfileToolSpec = {
  name: string;
  params: Record<string, SettingsValue>;
};

function toParams(value: unknown): Record<string, SettingsValue> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, SettingsValue>)
    : {};
}

/**
 * Read a stored profile's `tools` into picker state.
 *
 * Same tri-state as the scope fields: `null`/absent = the server's standard
 * set, an array = exactly those tools (`[]` = a deliberately bare agent).
 * Stored params ride along untouched so a round-trip through the editor cannot
 * drop configuration the editor does not model.
 */
export function readProfileTools(value: unknown): {
  mode: ProfileScopeMode;
  selected: string[];
  params: Record<string, Record<string, SettingsValue>>;
} {
  if (!Array.isArray(value))
    return { mode: "standard", selected: [], params: {} };
  const selected: string[] = [];
  const params: Record<string, Record<string, SettingsValue>> = {};
  value.forEach((entry) => {
    const name = (entry as { name?: unknown })?.name;
    if (typeof name !== "string" || name in params) return;
    params[name] = toParams((entry as { params?: unknown }).params);
    selected.push(name);
  });
  return { mode: "custom", selected, params };
}

/** Build the `tools` value to persist: `null` for standard, else the picks. */
export function buildProfileToolsValue({
  mode,
  selected,
  params = {},
}: {
  mode: ProfileScopeMode;
  selected: string[];
  params?: Record<string, Record<string, SettingsValue>>;
}): ProfileToolSpec[] | null {
  if (mode === "standard") return null;
  return selected.map((name) => ({ name, params: params[name] ?? {} }));
}
