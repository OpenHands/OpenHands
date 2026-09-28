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

/** The legacy tool switches a `tools` selection implies, for servers without a picker. */
export function toolSwitchesFromProfileTools(value: unknown): {
  enable_sub_agents: boolean;
  enable_switch_llm_tool: boolean;
} {
  if (!Array.isArray(value))
    return { enable_sub_agents: false, enable_switch_llm_tool: true };
  const { selected } = readProfileTools(value);
  return {
    enable_sub_agents: selected.includes("task_tool_set"),
    enable_switch_llm_tool:
      selected.includes("switch_llm") || selected.includes("SwitchLLMTool"),
  };
}
