import { SkillsClient } from "@openhands/typescript-client/clients";
import {
  SKILLS_CATALOG,
  type SkillCatalogEntry,
} from "@openhands/extensions/skills";
import { SkillInfo } from "#/types/settings";
import { getAgentServerWorkingDir } from "./agent-server-config";
import { getActiveBackend } from "./backend-registry/active-store";
import {
  fetchCloudConversationSkills,
  fetchCloudSkills,
} from "./cloud/skills-service.api";
import { getAgentServerClientOptions } from "./agent-server-client-options";
import type { LoadedResources } from "#/types/slash-command";
import { toLoadedSkillResources } from "#/utils/loaded-resources";

function catalogEntryToSkillInfo(entry: SkillCatalogEntry): SkillInfo {
  return {
    name: entry.name,
    type: "knowledge",
    source: "public",
    description: entry.description,
    triggers: entry.triggers,
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
const PUBLIC_SKILLS_BY_NAME = new Map(
  PUBLIC_SKILLS.map((skill) => [skill.name, skill]),
);

function addBundledDescription(skill: SkillInfo): SkillInfo {
  // Cloud has used both `global` and `public` for its bundled catalog scope.
  // Never enrich a same-named user/project/org skill with unrelated public
  // metadata.
  if (
    skill.description ||
    (skill.source !== "global" && skill.source !== "public")
  ) {
    return skill;
  }

  const bundledSkill = PUBLIC_SKILLS_BY_NAME.get(skill.name);
  if (!bundledSkill?.description) return skill;

  return { ...skill, description: bundledSkill.description };
}

class SkillsService {
  static async getSkills(projectDir?: string): Promise<SkillInfo[]> {
    if (getActiveBackend().backend.kind === "cloud") {
      const cloudSkills = await fetchCloudSkills();

      // Cloud skill search intentionally returns lightweight records without
      // descriptions. Preserve that list and its scopes, but restore metadata
      // for exact matches from the public catalog already bundled in Canvas.
      return cloudSkills.map(addBundledDescription);
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

  static async getConversationLoadedSkills(
    conversationId: string,
  ): Promise<LoadedResources["skills"]> {
    if (getActiveBackend().backend.kind !== "cloud") {
      throw new Error("Loaded Cloud skills require a cloud backend.");
    }

    const skills = await fetchCloudConversationSkills(conversationId);
    return toLoadedSkillResources(skills);
  }
}

export default SkillsService;
