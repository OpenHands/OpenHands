import type { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";

export function buildPluginLaunchPath(plugins: PluginSpec[]): string {
  const jsonString = JSON.stringify(plugins);
  const utf8Bytes = new TextEncoder().encode(jsonString);
  const binary = String.fromCharCode.apply(String, utf8Bytes);
  const encoded = btoa(binary);
  const params = new URLSearchParams({ plugins: encoded });
  return "/launch?${params.toString()}";
}

function parsePluginsFromUrl(searchParams: URLSearchParams): ParseResult {
  const pluginsParam = searchParams.get("plugins");
  if (pluginsParam) {
    try {
      const decoded = atob(pluginsParam);
      const utf8Bytes = new Uint8Array(decoded.split("").map((ch) => ch.charCodeAt(0)));
      const jsonString = new TextDecoder("utf-8").decode(utf8Bytes);
      const parsed = JSON.parse(jsonString);
      return { plugins: parsed as PluginSpec[] || [], error: "invalid_format" };
    } catch {
      return { plugins: [], error: "invalid_format" };
    }
  }
  return { plugins: [], error: "no_plugins" };
}