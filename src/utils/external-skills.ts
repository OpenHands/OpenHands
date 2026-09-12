import type { SkillInfo } from "#/types/settings";

/**
 * External skills supplied by the launcher's repeatable `--skills <source>`
 * option (OpenHands/OpenHands#17247).
 *
 * The launcher resolves each source — a local directory, a git URL, or
 * `host/path` shorthand — into a manifest at startup and the serving layer
 * inlines it as `window.__AGENT_CANVAS_EXTERNAL_SKILLS__` (a JSON string,
 * the same contract as `__AGENT_CANVAS_RUNTIME_SERVICES_INFO__`). Vite dev
 * mode gets it from the `VITE_EXTERNAL_SKILLS_FILE` entry-module transform;
 * `static-server.mjs` injects it from `--external-skills-file`. Nothing here
 * fetches from the agent-server: the manifest is launcher-provided config,
 * so this module is a pure read of a window global.
 */

const EXTERNAL_SKILLS_WINDOW_KEY = "__AGENT_CANVAS_EXTERNAL_SKILLS__";

/** One valid Agent Skill from a `--skills` source, as resolved at launch. */
export interface ExternalSkillEntry {
  name: string;
  description?: string | null;
  triggers?: string[];
  content: string;
  license?: string | null;
  compatibility?: string | null;
  version?: string | null;
  allowed_tools?: string[] | null;
  /**
   * Absolute SKILL.md path on the launcher host. Passed to the agent-server
   * as the skill's `source` so it can resolve relative resources
   * (scripts/, references/) when the skill is opted into a conversation.
   */
  path?: string | null;
  /** The `--skills` argument as supplied (display only). */
  source: string;
  /**
   * Stable source-qualified identity seed (`<slug>-<hash8>`). Combined with
   * the skill name into the enablement key by
   * `externalSkillEnablementKey()` so same-named skills from different
   * sources never share toggle state.
   */
  source_id: string;
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((v) => typeof v === "string");
}

function isExternalSkillEntry(value: unknown): value is ExternalSkillEntry {
  if (typeof value !== "object" || value === null) return false;
  const entry = value as Record<string, unknown>;
  return (
    typeof entry.name === "string" &&
    entry.name.length > 0 &&
    typeof entry.source_id === "string" &&
    entry.source_id.length > 0 &&
    typeof entry.content === "string" &&
    (entry.triggers === undefined || isStringArray(entry.triggers))
  );
}

/**
 * The resolved external skills manifest entries, or `[]` when the launcher
 * passed no `--skills` sources (the default). Tolerates the injected value
 * arriving either as the documented JSON string or already-parsed.
 */
export function getExternalSkillEntries(): ExternalSkillEntry[] {
  if (typeof window === "undefined") return [];
  const raw = (window as unknown as Record<string, unknown>)[
    EXTERNAL_SKILLS_WINDOW_KEY
  ];
  if (raw === undefined || raw === null || raw === "") return [];

  let parsed: unknown = raw;
  if (typeof raw === "string") {
    try {
      parsed = JSON.parse(raw);
    } catch {
      return [];
    }
  }

  const skills = (parsed as { skills?: unknown })?.skills;
  if (!Array.isArray(skills)) return [];
  return skills.filter(isExternalSkillEntry);
}

/**
 * Manifest entry → `SkillInfo` for listing surfaces. `source` carries the
 * human-supplied `--skills` value so the card shows where the skill came
 * from; `source_id` marks it as launcher-external for scope grouping and
 * collision-safe enablement keys.
 */
export function externalSkillToSkillInfo(entry: ExternalSkillEntry): SkillInfo {
  return {
    name: entry.name,
    type: "agentskills",
    source: entry.source ?? null,
    source_id: entry.source_id,
    description: entry.description ?? null,
    triggers: entry.triggers ?? [],
    content: entry.content,
    license: entry.license ?? null,
    compatibility: entry.compatibility ?? null,
    version: entry.version ?? undefined,
    allowed_tools: entry.allowed_tools ?? null,
    is_agentskills_format: true,
  };
}

/**
 * The external skill a message invokes by name, if any — `/<skill-name>` or a
 * declared "/" trigger, mirroring `findInvokedCatalogSkill` for the bundled
 * catalog. Returns the manifest entry rather than a bare name because two
 * `--skills` sources may ship the same skill name; callers compare
 * `source_id` + `name`.
 */
export function findInvokedExternalSkill(
  query?: string,
): ExternalSkillEntry | undefined {
  const firstToken = query?.trim().split(/\s+/, 1)[0];
  if (!firstToken?.startsWith("/")) return undefined;
  const command = firstToken.toLowerCase();

  return getExternalSkillEntries().find((entry) =>
    [`/${entry.name}`, ...(entry.triggers ?? [])]
      .filter((trigger) => trigger.startsWith("/"))
      .some((trigger) => trigger.toLowerCase() === command),
  );
}

/**
 * The `agent_context.skills` payload for one external skill. Mirrors the SDK
 * `Skill` shape `buildBundledSkills()` emits — the server uses it for trigger
 * matching, activation, and system-prompt injection — except `source` is the
 * absolute SKILL.md path (or a qualified fallback) rather than a bundled
 * catalog path.
 */
export function externalSkillToAgentSkill(entry: ExternalSkillEntry): {
  name: string;
  content: string;
  trigger: { type: "keyword"; keywords: string[] } | null;
  source: string;
  description: string | null;
  is_agentskills_format: true;
  license?: string;
  compatibility?: string;
} {
  return {
    name: entry.name,
    content: entry.content,
    trigger:
      entry.triggers && entry.triggers.length > 0
        ? { type: "keyword", keywords: entry.triggers }
        : null,
    source: entry.path || `external:${entry.source_id}`,
    description: entry.description ?? null,
    is_agentskills_format: true,
    ...(entry.license ? { license: entry.license } : {}),
    ...(entry.compatibility ? { compatibility: entry.compatibility } : {}),
  };
}
