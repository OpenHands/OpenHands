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

/** Read stored `tools` into picker state: absent = standard, `[]` = bare. */
export function readProfileTools(value: unknown): {
  mode: ProfileScopeMode;
  selected: string[];
  params: Record<string, Record<string, SettingsValue>>;
} {
  if (!Array.isArray(value))
    return { mode: "standard", selected: [], params: {} };
  const params = new Map<string, Record<string, SettingsValue>>();
  value.forEach((entry) => {
    const name = (entry as { name?: unknown })?.name;
    if (typeof name !== "string") return;
    if (!params.has(name))
      params.set(name, toParams((entry as { params?: unknown }).params));
  });
  return {
    mode: "custom",
    selected: [...params.keys()],
    params: Object.fromEntries(params),
  };
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
  return selected.map((name) => ({
    name,
    params: Object.hasOwn(params, name) ? params[name] : {},
  }));
}
