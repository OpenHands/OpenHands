import { I18nKey } from "#/i18n/declaration";

/** Tool spec as it is stored on an agent profile and sent on the wire. */
export interface ProfileToolSpec {
  name: string;
  params: Record<string, unknown>;
}

/** Picker state for a profile's `tools` field. */
export type ProfileToolsMode = "standard" | "custom";

/**
 * The SDK's deterministic default tool set (`DEFAULT_EXEC_TOOL_NAMES`), used
 * whenever a profile carries no explicit `tools`.
 */
export const DEFAULT_TOOL_NAMES = [
  "terminal",
  "file_editor",
  "task_tracker",
] as const;

export const BROWSER_TOOL_NAME = "browser_tool_set";

/**
 * Sub-agent delegation. Kept out of the picker: `enable_sub_agents` is the
 * single control for it, and `create_agent` honours that flag only while
 * `tools` is unset, so the save re-adds this name for an explicit list.
 */
export const SUB_AGENT_TOOL_NAME = "task_tool_set";

/**
 * The tools a profile may scope, in picker order, each with a description.
 *
 * Deliberately an allow-list. The backend's `usable_tools` is a dump of whatever
 * the server process imported — `tool_router.py` registers the default,
 * builtin-agent, Gemini and planning presets unconditionally at import time — so
 * it answers "can this server run X?", not "should a user pick X?". Treating it
 * as a menu surfaced the Gemini file family (a parallel set to `file_editor`
 * that no product path builds an agent from), the planning agent's internal
 * PLAN.md editor (which needs a per-launch `plan_path` a stored profile cannot
 * supply), and the low-level half of the workflow pair whose own docstring says
 * to prefer the set.
 *
 * A new SDK tool therefore needs a line here before it appears. That is the
 * intended trade: an undescribed entry in a picker that decides what an agent
 * can do is worse than a missing one, and canvas already pins an agent-server
 * version.
 */
export const KNOWN_PROFILE_TOOL_DESCRIPTIONS: Record<string, I18nKey> = {
  terminal: I18nKey.SETTINGS$TOOL_DESC_TERMINAL,
  file_editor: I18nKey.SETTINGS$TOOL_DESC_FILE_EDITOR,
  task_tracker: I18nKey.SETTINGS$TOOL_DESC_TASK_TRACKER,
  glob: I18nKey.SETTINGS$TOOL_DESC_GLOB,
  grep: I18nKey.SETTINGS$TOOL_DESC_GREP,
  [BROWSER_TOOL_NAME]: I18nKey.SETTINGS$TOOL_DESC_BROWSER,
};

const KNOWN_PROFILE_TOOL_NAMES = Object.keys(KNOWN_PROFILE_TOOL_DESCRIPTIONS);

function isUsable(name: string, usableTools: string[] | null): boolean {
  return usableTools === null || usableTools.includes(name);
}

/**
 * Ordered tool names the picker offers: the allow-listed tools this backend can
 * run, plus any the stored profile already carries.
 *
 * A backend that advertises no `usable_tools` (cloud serves no `/server_info`)
 * gets the whole allow-list.
 */
export function buildProfileToolCatalog({
  usableTools,
  storedToolNames = [],
}: {
  usableTools: string[] | null;
  storedToolNames?: string[];
}): string[] {
  const catalog: string[] = [];
  const push = (name: string) => {
    if (!catalog.includes(name)) catalog.push(name);
  };
  KNOWN_PROFILE_TOOL_NAMES.filter((name) =>
    isUsable(name, usableTools),
  ).forEach(push);
  // Stored names ride along even when they fall outside the allow-list or the
  // backend no longer advertises them: the save is a whole-profile overwrite,
  // so hiding a tool the profile was given through the API would silently strip
  // it. Visible means clearable. The sub-agent tool is the one exception — the
  // `enable_sub_agents` toggle owns it and the save re-adds it.
  storedToolNames.filter((name) => name !== SUB_AGENT_TOOL_NAME).forEach(push);
  return catalog;
}

/**
 * What a `tools: null` profile actually launches with — shown as the read-only
 * preview of "standard".
 *
 * Browser is absent from the SDK default because it is environment-dependent;
 * the agent-server appends it on a profile launch when it is usable and the
 * profile sets no explicit `tools`. Mirrored here so the preview matches.
 */
export function standardProfileToolNames({
  usableTools,
  subAgentsEnabled,
}: {
  usableTools: string[] | null;
  subAgentsEnabled: boolean;
}): string[] {
  const names: string[] = [...DEFAULT_TOOL_NAMES];
  if (isUsable(BROWSER_TOOL_NAME, usableTools)) names.push(BROWSER_TOOL_NAME);
  if (subAgentsEnabled && isUsable(SUB_AGENT_TOOL_NAME, usableTools)) {
    names.push(SUB_AGENT_TOOL_NAME);
  }
  return names;
}

function toRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

/**
 * Read a stored profile's `mcp_server_refs` into picker state.
 *
 * Same tri-state as `tools`: `null`/absent = every configured server, an array
 * = only those keys.
 */
export function readProfileMcpRefs(value: unknown): {
  mode: ProfileToolsMode;
  selected: string[];
} {
  if (!Array.isArray(value)) return { mode: "standard", selected: [] };
  return {
    mode: "custom",
    selected: value.filter((name): name is string => typeof name === "string"),
  };
}

/**
 * Read a stored profile's `tools` into picker state.
 *
 * `params` keeps every stored spec's params — including the sub-agent tool's,
 * which the picker hides — so a round-trip through the editor preserves them.
 * Anything that is not an array of named specs reads as "standard", the value
 * the field defaults to.
 */
export function readProfileTools(value: unknown): {
  mode: ProfileToolsMode;
  selected: string[];
  params: Record<string, Record<string, unknown>>;
} {
  if (!Array.isArray(value)) {
    return { mode: "standard", selected: [], params: {} };
  }
  const selected: string[] = [];
  const params: Record<string, Record<string, unknown>> = {};
  value.forEach((entry) => {
    const name = (entry as { name?: unknown })?.name;
    if (typeof name !== "string" || name in params) return;
    params[name] = toRecord((entry as { params?: unknown }).params);
    if (name !== SUB_AGENT_TOOL_NAME) selected.push(name);
  });
  return { mode: "custom", selected, params };
}

/**
 * Read a stored profile's `secret_refs` into picker state.
 *
 * Same tri-state as `tools`, minus the params: `null`/absent = every secret,
 * an array = only those names. The ACP provider credentials the server unions
 * back in are deliberately not modelled here — they are not the user's to
 * deselect.
 */
export function readProfileSecretRefs(value: unknown): {
  mode: ProfileToolsMode;
  selected: string[];
} {
  if (!Array.isArray(value)) return { mode: "standard", selected: [] };
  return {
    mode: "custom",
    selected: value.filter((name): name is string => typeof name === "string"),
  };
}

/**
 * Build the `tools` value to persist: `null` for standard, otherwise the
 * selection plus the sub-agent tool when that toggle is on and the backend can
 * run it.
 */
export function buildProfileToolsValue({
  mode,
  selected,
  params = {},
  subAgentsEnabled,
  usableTools,
}: {
  mode: ProfileToolsMode;
  selected: string[];
  params?: Record<string, Record<string, unknown>>;
  subAgentsEnabled: boolean;
  usableTools: string[] | null;
}): ProfileToolSpec[] | null {
  if (mode === "standard") return null;
  const names = selected.filter((name) => name !== SUB_AGENT_TOOL_NAME);
  if (subAgentsEnabled && isUsable(SUB_AGENT_TOOL_NAME, usableTools)) {
    names.push(SUB_AGENT_TOOL_NAME);
  }
  return names.map((name) => ({ name, params: params[name] ?? {} }));
}
