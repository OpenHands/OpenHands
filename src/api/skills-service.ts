import {
  AgentServerClient,
  SkillsClient,
} from "@openhands/typescript-client/clients";
import {
  SKILLS_CATALOG,
  type SkillCatalogEntry,
} from "@openhands/extensions/skills";
import { SkillInfo, SkillOrigin } from "#/types/settings";
import { getAgentServerWorkingDir } from "./agent-server-config";
import { getActiveBackend } from "./backend-registry/active-store";
import {
  fetchCloudConversationSkills,
  fetchCloudSkills,
} from "./cloud/skills-service.api";
import { getAgentServerClientOptions } from "./agent-server-client-options";

function catalogEntryToSkillInfo(entry: SkillCatalogEntry): SkillInfo {
  return {
    name: entry.name,
    type: "knowledge",
    source: "public",
    description: entry.description,
    triggers: entry.triggers,
    category: entry.category,
    content: entry.content,
    license: entry.license ?? null,
    compatibility: entry.compatibility ?? null,
  };
}

/**
 * Public skills loaded from the `@openhands/extensions` npm package.
 *
 * This is an **immutable build-time snapshot**: the catalog is baked into the
 * bundle at `npm run build` / `vite build` time and does not change at
 * runtime. Updating the catalog requires bumping the `@openhands/extensions`
 * dependency and rebuilding.
 */
const PUBLIC_SKILLS: SkillInfo[] = SKILLS_CATALOG.map(catalogEntryToSkillInfo);

const BUNDLED_BY_NAME = new Map(SKILLS_CATALOG.map((e) => [e.name, e]));

export const SKILL_CATALOG_PATH = "/api/skills/catalog";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isOrigin(value: unknown): value is SkillOrigin {
  return value === "public" || value === "user";
}

/** A server entry, with the bundled catalog's display metadata for public skills. */
function toCatalogSkill(entry: unknown): SkillInfo {
  if (
    !isRecord(entry) ||
    typeof entry.name !== "string" ||
    !isOrigin(entry.origin)
  ) {
    throw new Error("The agent server returned a malformed skill catalog.");
  }
  const skill = entry as unknown as SkillInfo;
  if (skill.origin !== "public") return skill;
  // Shown as the catalog, as the bundled entries are, rather than a server path.
  const listed = { ...skill, source: "public" };
  const bundled = BUNDLED_BY_NAME.get(skill.name);
  if (!bundled) return listed;
  return {
    ...listed,
    category: bundled.category,
    license: skill.license ?? bundled.license ?? null,
    compatibility: skill.compatibility ?? bundled.compatibility ?? null,
  };
}

function hasStatus(error: unknown, status: number): boolean {
  return isRecord(error) && error.status === status;
}

class SkillsService {
  static async getSkills(projectDir?: string): Promise<SkillInfo[]> {
    if (getActiveBackend().backend.kind === "cloud") {
      return fetchCloudSkills();
    }

    // Public skills come from the bundled @openhands/extensions npm package —
    // no agent-server round-trip or GitHub fetch needed. Only ask the agent-
    // server for user and project skills so local .agents/skills/ content is
    // still picked up.
    let localSkills: SkillInfo[] = [];
    try {
      const response = await new SkillsClient(
        getAgentServerClientOptions(),
      ).getSkills({
        load_public: false,
        load_user: true,
        load_project: true,
        load_org: false,
        project_dir: projectDir ?? getAgentServerWorkingDir(),
      });
      localSkills = (response.skills ?? []) as SkillInfo[];
    } catch {
      // Agent-server may not support the skills endpoint or may be
      // unreachable; fall back to the bundled public catalog alone.
    }

    return [...localSkills, ...PUBLIC_SKILLS];
  }

  /**
   * Skills an agent profile can select on this server. Repo skills load only
   * when a conversation opens its workspace, so they are not listed.
   */
  static async getSkillCatalog(): Promise<SkillInfo[]> {
    const { host, apiKey } = getAgentServerClientOptions();
    try {
      const data = await new AgentServerClient({ host, apiKey }).get<unknown>(
        SKILL_CATALOG_PATH,
        { acceptableStatusCodes: new Set([200]), responseType: "json" },
      );
      const skills = isRecord(data) ? data.skills : undefined;
      if (!Array.isArray(skills)) {
        throw new Error("The agent server returned a malformed skill catalog.");
      }
      return skills.map(toCatalogSkill);
    } catch (error) {
      if (!hasStatus(error, 404)) throw error;
    }
    // Servers before the catalog endpoint: the bundled catalog plus user skills.
    const response = await new SkillsClient(
      getAgentServerClientOptions(),
    ).getSkills({
      load_public: false,
      load_user: true,
      load_project: false,
      load_org: false,
    });
    const userSkills = ((response.skills ?? []) as SkillInfo[]).map(
      (skill) => ({ ...skill, origin: "user" as const }),
    );
    const userNames = new Set(userSkills.map((skill) => skill.name));
    return [
      ...PUBLIC_SKILLS.filter((skill) => !userNames.has(skill.name)).map(
        (skill) => ({ ...skill, origin: "public" as const }),
      ),
      ...userSkills,
    ];
  }

  /**
   * Skills loaded into a running cloud conversation (see
   * `fetchCloudConversationSkills`). Cloud-only: local conversations keep
   * using `getSkills(projectDir)`, whose agent-server call already scopes to
   * the conversation's workspace.
   */
  static getConversationSkills(conversationId: string): Promise<SkillInfo[]> {
    return fetchCloudConversationSkills(conversationId);
  }
}

export default SkillsService;
