import type { ProfileScopeMode } from "#/constants/profile-scope";
import type { SettingsValue } from "#/types/settings";

/** A tool spec as stored on an agent profile and sent on the wire. */
export type ProfileToolSpec = {
  name: string;
  params: Record<string, SettingsValue>;
};

/** Form-seed key carrying the stored profile's `schema_version`. */
export const PROFILE_SCHEMA_VERSION_KEY = "profile_schema_version";

/** Whether a profile schema reads `switch_llm` from `tools` rather than a switch. */
export function profileToolsCarrySwitchLlm(schemaVersion: unknown): boolean {
  return typeof schemaVersion === "number" && schemaVersion >= 3;
}

const SUB_AGENT_TOOL_NAME = "task_tool_set";
const SWITCH_LLM_TOOL_NAME = "switch_llm";

/** Selectable built-ins a profile may store under their class name. */
const BUILT_IN_TOOL_NAMES: ReadonlyMap<string, string> = new Map([
  ["SwitchLLMTool", SWITCH_LLM_TOOL_NAME],
]);

/** The tool name a stored spec resolves to. */
export function canonicalToolName(name: string): string {
  return BUILT_IN_TOOL_NAMES.get(name) ?? name;
}

function toParams(value: unknown): Record<string, SettingsValue> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, SettingsValue>)
    : {};
}

/** Stored `tools` as specs, or `null` when unset. */
export function readStoredProfileTools(
  value: unknown,
): ProfileToolSpec[] | null {
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

function withTool(
  tools: ProfileToolSpec[],
  name: string,
  enabled: boolean,
): ProfileToolSpec[] {
  const without = tools.filter((spec) => canonicalToolName(spec.name) !== name);
  if (!enabled) return without;
  return without.length === tools.length
    ? [...tools, { name, params: {} }]
    : tools;
}

/**
 * Apply the legacy switches to an explicit `tools` list. `switchLlm` is left
 * alone when undefined: older servers attach that tool from the switch alone.
 */
export function applyToolSwitchesToProfileTools(
  tools: ProfileToolSpec[],
  { subAgents, switchLlm }: { subAgents: boolean; switchLlm?: boolean },
): ProfileToolSpec[] {
  const withSubAgents = withTool(tools, SUB_AGENT_TOOL_NAME, subAgents);
  return switchLlm === undefined
    ? withSubAgents
    : withTool(withSubAgents, SWITCH_LLM_TOOL_NAME, switchLlm);
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
    enable_sub_agents: selected.includes(SUB_AGENT_TOOL_NAME),
    enable_switch_llm_tool: selected.includes(SWITCH_LLM_TOOL_NAME),
  };
}
