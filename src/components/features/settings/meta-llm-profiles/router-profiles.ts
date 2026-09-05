/**
 * Helpers for creating the LLM profiles a meta-profile (model router) needs.
 *
 * A router's ``model_table`` lists the models the classifier may pick from, one
 * per line as ``- <name> <description...>``. The router matches the classifier's
 * answer against saved LLM *profile names*, so each ``<name>`` must exist as a
 * profile. We create the missing ones by pairing the name with a chosen provider
 * connection: the endpoint is derived by convention as
 * ``<provider>.toLowerCase()/<name>`` and the profile links the connection for
 * its credentials instead of cloning them.
 */

/**
 * Extract model names from a router ``model_table``.
 *
 * Each row looks like ``- <name> <anything else>`` (the trailing text is human
 * description we ignore). A ``:`` directly after the name (``- <name>: stats``)
 * is also stripped. Blank lines and non-list lines are skipped, and names are
 * de-duplicated while preserving first-seen order.
 */
export function parseModelTableNames(
  modelTable: string | null | undefined,
): string[] {
  if (!modelTable) return [];
  const names: string[] = [];
  const seen = new Set<string>();
  for (const line of modelTable.split("\n")) {
    const match = line.match(/^\s*-\s+(\S+)/);
    if (!match) continue;
    const name = match[1].replace(/:+$/, "");
    const key = name.toLowerCase();
    if (!name || seen.has(key)) continue;
    seen.add(key);
    names.push(name);
  }
  return names;
}

/**
 * Derive the LLM endpoint for a router model under a provider connection.
 *
 * Convention (intentionally simple so this can ship; extend per-provider in a
 * later PR if needed): ``<provider>.toLowerCase()/<name>``. This holds for
 * openhands, anthropic, gemini, openai and similar litellm providers.
 */
export function buildRouterModel(provider: string, name: string): string {
  return `${provider.toLowerCase()}/${name}`;
}

/**
 * The set of profile names a router config depends on: every model in the
 * table plus the classifier and default models. De-duplicated, first-seen
 * order, table names first.
 */
export function collectRequiredRouterModelNames(config: {
  classifier_model?: string | null;
  default_model?: string | null;
  model_table?: string | null;
}): string[] {
  const names = parseModelTableNames(config.model_table);
  const seen = new Set(names.map((name) => name.toLowerCase()));
  for (const extra of [config.classifier_model, config.default_model]) {
    const name = (extra ?? "").trim();
    if (name && !seen.has(name.toLowerCase())) {
      seen.add(name.toLowerCase());
      names.push(name);
    }
  }
  return names;
}
