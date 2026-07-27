import type { SkillInfo } from "#/types/settings";

export type SkillScope = "project" | "personal" | "public";

export const SKILL_SCOPE_ORDER: SkillScope[] = [
  "project",
  "personal",
  "public",
];

const USER_SKILL_DIR_MARKERS = [
  "/.agents/skills/",
  "/.openhands/skills/",
  "/.openhands/microagents/",
] as const;

function normalizePath(path: string): string {
  return path.replace(/\\/g, "/");
}

function isPublicSource(source: string): boolean {
  const norm = normalizePath(source).toLowerCase();
  if (norm === "public") {
    return true;
  }
  return (
    norm.includes("public-skills") || norm.includes("/.openhands/cache/skills/")
  );
}

function isUserHomeSkillPath(source: string): boolean {
  const norm = normalizePath(source);
  if (/^\/Users\/[^/]+\/\.(agents|openhands)\//.test(norm)) {
    return true;
  }
  if (/^\/home\/[^/]+\/\.(agents|openhands)\//.test(norm)) {
    return true;
  }

  return USER_SKILL_DIR_MARKERS.some((marker) => {
    const markerIndex = norm.indexOf(marker);
    if (markerIndex === -1) {
      return false;
    }
    const prefix = norm.slice(0, markerIndex);
    return /^\/Users\/[^/]+$/.test(prefix) || /^\/home\/[^/]+$/.test(prefix);
  });
}

function isProjectSkillPath(
  source: string,
  projectDir?: string | null,
): boolean {
  const norm = normalizePath(source);
  const hasProjectMarker = USER_SKILL_DIR_MARKERS.some((marker) =>
    norm.includes(marker),
  );
  if (!hasProjectMarker) {
    return false;
  }
  if (isUserHomeSkillPath(source)) {
    return false;
  }

  if (projectDir) {
    const projectNorm = normalizePath(projectDir).replace(/\/$/, "");
    if (norm.startsWith(projectNorm)) {
      return true;
    }
  }

  return true;
}

export function getSkillScope(
  skill: SkillInfo,
  projectDir?: string | null,
): SkillScope {
  const source = skill.source?.trim();
  if (!source) {
    return skill.type === "repo" ? "project" : "public";
  }

  const lower = source.toLowerCase();
  if (isPublicSource(source)) {
    return "public";
  }
  if (lower === "user" || lower === "global") {
    return "personal";
  }
  if (lower === "project" || lower === "repo" || lower === "sandbox") {
    return "project";
  }
  if (isUserHomeSkillPath(source)) {
    return "personal";
  }
  if (isProjectSkillPath(source, projectDir)) {
    return "project";
  }

  return "public";
}

export function groupSkillsByScope(
  skills: SkillInfo[],
  projectDir?: string | null,
): Record<SkillScope, SkillInfo[]> {
  const groups: Record<SkillScope, SkillInfo[]> = {
    project: [],
    personal: [],
    public: [],
  };

  for (const skill of skills) {
    groups[getSkillScope(skill, projectDir)].push(skill);
  }

  for (const scope of SKILL_SCOPE_ORDER) {
    groups[scope].sort((left, right) => left.name.localeCompare(right.name));
  }

  return groups;
}
