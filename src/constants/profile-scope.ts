/** Picker state for a profile's scope fields: server default, or an explicit list. */
export type ProfileScopeMode = "standard" | "custom";

/**
 * Read a stored profile's `mcp_server_refs` into picker state.
 *
 * Tri-state on the wire: `null`/absent = every configured server, an array =
 * only those keys (`[]` = none).
 */
export function readProfileMcpRefs(value: unknown): {
  mode: ProfileScopeMode;
  selected: string[];
} {
  if (!Array.isArray(value)) return { mode: "standard", selected: [] };
  return {
    mode: "custom",
    selected: value.filter((name): name is string => typeof name === "string"),
  };
}
