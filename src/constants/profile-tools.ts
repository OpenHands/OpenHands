import type { ProfileScopeMode } from "#/constants/profile-scope";
import type { SettingsValue } from "#/types/settings";

/** A tool spec as stored on an agent profile and sent on the wire. */
export type ProfileToolSpec = {
  name: string;
  params: Record<string, SettingsValue>;
};

const SWITCH_LLM_TOOL_NAME = "switch_llm";

/** Selectable built-ins a profile may store under their class name. */
const BUILT_IN_TOOL_NAMES: ReadonlyMap<string, string> = new Map([
  ["SwitchLLMTool", SWITCH_LLM_TOOL_NAME],
]);

/** The tool name a stored spec resolves to. */
function canonicalToolName(name: string): string {
  return BUILT_IN_TOOL_NAMES.get(name) ?? name;
}

function toParams(value: unknown): Record<string, SettingsValue> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, SettingsValue>)
    : {};
}

function readStoredProfileTools(value: unknown): ProfileToolSpec[] | null {
  if (!Array.isArray(value)) return null;
  return value.flatMap((entry) => {
    const name = (entry as { name?: unknown })?.name;
    return typeof name === "string"
      ? [{ name, params: toParams((entry as { params?: unknown }).params) }]
      : [];
  });
}

/** Read stored `tools` into picker state: absent = standard, `[]` = bare. */
export function readProfileTools(value: unknown): {
  mode: ProfileScopeMode;
  selected: string[];
  params: Record<string, Record<string, SettingsValue>>;
} {
  const stored = readStoredProfileTools(value);
  if (stored === null) return { mode: "standard", selected: [], params: {} };
  const params = new Map<string, Record<string, SettingsValue>>();
  stored.forEach((spec) => {
    const name = canonicalToolName(spec.name);
    if (!params.has(name)) params.set(name, spec.params);
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
