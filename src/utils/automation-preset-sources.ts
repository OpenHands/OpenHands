import type {
  Automation,
  AutomationPluginSpec,
  AutomationRepository,
} from "#/types/automation";

type UnknownRecord = Record<string, unknown>;

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readRecords(value: unknown): UnknownRecord[] {
  return Array.isArray(value) ? value.filter(isRecord) : [];
}

function readNonEmptyString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim().length > 0
    ? value
    : undefined;
}

/** The repository entries in `value` that name a URL. */
export function readAutomationRepositories(
  value: unknown,
): AutomationRepository[] {
  return readRecords(value).flatMap((repo) => {
    const url = readNonEmptyString(repo.url);
    if (!url) return [];
    const ref = readNonEmptyString(repo.ref);
    const provider = readNonEmptyString(repo.provider);
    return [{ url, ...(ref && { ref }), ...(provider && { provider }) }];
  });
}

function readPresetPlugins(metadata: UnknownRecord): string[] {
  return readRecords(metadata.plugins).flatMap((plugin) => {
    const source = readNonEmptyString(plugin.source);
    return source ? [source] : [];
  });
}

/**
 * Read the plugin-spec payload only when the service marks this automation as
 * plugin_spec. Invalid or older payloads keep using the existing script path.
 */
export function getAutomationPluginSpec(
  automation: Pick<Automation, "preset_metadata" | "plugin_spec">,
): AutomationPluginSpec | null {
  const metadata = isRecord(automation.preset_metadata)
    ? automation.preset_metadata
    : {};
  if (metadata.preset_type !== "plugin_spec") return null;

  const payload = automation.plugin_spec;
  if (!isRecord(payload) || !isRecord(payload.plugin)) return null;
  const source = readNonEmptyString(payload.plugin.source);
  if (!source) return null;

  const ref = readNonEmptyString(payload.plugin.ref);
  const repoPath = readNonEmptyString(payload.plugin.repo_path);
  return {
    plugin: {
      source,
      ...(ref && { ref }),
      ...(repoPath && { repo_path: repoPath }),
    },
    config: isRecord(payload.config) ? payload.config : {},
  };
}

/**
 * The automation service keeps an automation's repositories and plugins only
 * inside `preset_metadata` (`repos: [{url, ref?, provider?}]`,
 * `plugins: [{source, ref?, repo_path?}]`); its response has no top-level
 * fields for them. Lift them onto the `repositories` and `plugins` fields the
 * UI reads, falling back to the top-level `repository`/`branch`/`plugins` a
 * record without that metadata may carry. Fields with nothing to show stay
 * absent.
 */
export function withPresetSources(automation: Automation): Automation {
  const metadata = isRecord(automation.preset_metadata)
    ? automation.preset_metadata
    : {};

  const presetRepositories = readAutomationRepositories(metadata.repos);
  const legacyRepositories: AutomationRepository[] = automation.repository
    ? [
        {
          url: automation.repository,
          ...(automation.branch && { ref: automation.branch }),
        },
      ]
    : [];
  const repositories =
    presetRepositories.length > 0 ? presetRepositories : legacyRepositories;

  const presetPlugins = readPresetPlugins(metadata);

  return {
    ...automation,
    ...(repositories.length > 0 && { repositories }),
    ...(presetPlugins.length > 0 && { plugins: presetPlugins }),
  };
}
