import type { Automation, AutomationRepository } from "#/types/automation";

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

function readPresetRepositories(
  metadata: UnknownRecord,
): AutomationRepository[] {
  return readRecords(metadata.repos).flatMap((repo) => {
    const url = readNonEmptyString(repo.url);
    if (!url) return [];
    const ref = readNonEmptyString(repo.ref);
    return [{ url, ...(ref && { ref }) }];
  });
}

function readPresetPlugins(metadata: UnknownRecord): string[] {
  return readRecords(metadata.plugins).flatMap((plugin) => {
    const source = readNonEmptyString(plugin.source);
    return source ? [source] : [];
  });
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

  const presetRepositories = readPresetRepositories(metadata);
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
