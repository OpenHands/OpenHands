import type { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";

export function hasPluginParameters(
  plugins: PluginSpec[] | undefined,
): boolean {
  return (
    plugins?.some(({ parameters }) => {
      if (parameters == null) return false;
      // URL payloads can contain scalars or arrays despite PluginSpec's type.
      return (
        typeof parameters !== "object" ||
        Array.isArray(parameters) ||
        Object.keys(parameters).length > 0
      );
    }) ?? false
  );
}
