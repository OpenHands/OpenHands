import type { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";

/**
 * Friendly display name for a plugin reference: the explicit `name` when set
 * (e.g. an installed plugin's name), otherwise the last segment of `repo_path`,
 * otherwise the repo from the `source` coordinate. Mirrors the `/launch` modal's
 * naming so the same plugin reads consistently across surfaces.
 */
export function getPluginDisplayName(plugin: PluginSpec): string {
  if (plugin.name) return plugin.name;
  if (plugin.repo_path) {
    const parts = plugin.repo_path.split("/").filter(Boolean);
    if (parts.length > 0) return parts[parts.length - 1];
  }
  const { source } = plugin;
  if (source.startsWith("github:")) return source.replace("github:", "");
  if (source.includes("/")) {
    return (
      source
        .split("/")
        .pop()
        ?.replace(/\.git$/, "") || source
    );
  }
  return source;
}

/**
 * True when the plugin was installed from the local machine — the `"local"`
 * sentinel or a filesystem path — rather than a remote git source (`github:…`
 * or a URL). These read inconsistently (a bare `"local"` vs a long absolute
 * path that also leaks the home dir), so callers render one normalized label
 * for them instead of {@link getPluginSourceLabel}.
 */
export function isLocalPluginSource(plugin: PluginSpec): boolean {
  const { source } = plugin;
  if (source.startsWith("github:")) return false;
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(source)) return false; // git URL
  return true;
}

/** Source coordinate with an optional `@ref`, e.g. "OpenHands/extensions @ main". */
export function getPluginSourceLabel(plugin: PluginSpec): string {
  const base = plugin.source.startsWith("github:")
    ? plugin.source.replace("github:", "")
    : plugin.source;
  return plugin.ref ? `${base} @ ${plugin.ref}` : base;
}

/** Stable key for a plugin reference (coordinates only). */
export function pluginReferenceKey(plugin: PluginSpec): string {
  return [plugin.source, plugin.ref ?? "", plugin.repo_path ?? ""].join(" ");
}
