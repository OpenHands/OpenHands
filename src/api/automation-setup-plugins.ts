/**
 * Plugin list for the automation setup form.
 *
 * The form still keeps `pluginSource` and `pluginRef` so the agent tool and
 * older drafts can fill the first plugin. Every plugin, including later ones
 * and an optional ref, lives in `pluginList` (a JSON string, so it survives
 * the string-only draft patch). When those disagree, the scalar fields win
 * for the first row and the rest of the list stays put.
 */

export interface AutomationSetupPluginEntry {
  source: string;
  ref: string;
}

/** Git ref prefilled on a newly added plugin card. */
export const DEFAULT_AUTOMATION_PLUGIN_REF = "main";

interface PluginFormFields {
  pluginList?: string;
  pluginSource?: string;
  pluginRef?: string;
}

/**
 * Parse the persisted plugin list.
 *
 * Invalid JSON and non-array values become an empty list so a corrupt draft
 * cannot throw while the panel is opening.
 */
export function parseAutomationSetupPluginList(
  value: string | undefined,
): AutomationSetupPluginEntry[] {
  if (!value?.trim()) return [];
  try {
    const parsed = JSON.parse(value) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.flatMap((entry) => {
      if (!entry || typeof entry !== "object") return [];
      const record = entry as Record<string, unknown>;
      return [
        {
          source: typeof record.source === "string" ? record.source : "",
          ref: typeof record.ref === "string" ? record.ref : "",
        },
      ];
    });
  } catch {
    return [];
  }
}

/** Serialize plugin rows for the string form field. */
export function serializeAutomationSetupPluginList(
  entries: AutomationSetupPluginEntry[],
): string {
  return JSON.stringify(
    entries.map((entry) => ({ source: entry.source, ref: entry.ref })),
  );
}

function overlayFirstPlugin(
  entries: AutomationSetupPluginEntry[],
  form: PluginFormFields,
): AutomationSetupPluginEntry[] {
  if (entries.length === 0) {
    if (!form.pluginSource && !form.pluginRef) return [];
    return [{ source: form.pluginSource ?? "", ref: form.pluginRef ?? "" }];
  }

  const first = entries[0];
  const source = form.pluginSource ?? first.source;
  const ref = form.pluginRef ?? first.ref;
  if (source === first.source && ref === first.ref) return entries;
  return [{ source, ref }, ...entries.slice(1)];
}

/**
 * The plugins the form should show and save.
 *
 * `pluginList` is canonical once it has been written. Older drafts that only
 * stored `plugins: string[]` or a single source still expand into rows.
 */
export function resolveAutomationSetupPlugins(
  form: PluginFormFields,
  draftPlugins: string[] = [],
): AutomationSetupPluginEntry[] {
  if (typeof form.pluginList === "string") {
    return overlayFirstPlugin(
      parseAutomationSetupPluginList(form.pluginList),
      form,
    );
  }

  if (draftPlugins.length > 1) {
    return overlayFirstPlugin(
      draftPlugins.map((source) => ({ source, ref: "" })),
      form,
    );
  }

  const source = form.pluginSource ?? draftPlugins[0] ?? "";
  const ref = form.pluginRef ?? "";
  if (!source && !ref) return [];
  return [{ source, ref }];
}
