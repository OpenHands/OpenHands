import type { SkillInfo } from "#/types/settings";

export const CUSTOM_SKILLS_STORAGE_KEY = "openhands_custom_skills";
export const CUSTOM_SKILLS_UPDATED_EVENT = "openhands:custom_skills_updated";

const FRONTMATTER_REGEX = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/;

export interface CustomSkillInput {
  name: string;
  description?: string | null;
  triggers?: string[];
  content?: string | null;
}

export interface ParsedSkillFrontmatter {
  name?: string;
  description?: string | null;
  triggers?: string[];
  body: string;
}

export function parseSkillFrontmatter(content: string): ParsedSkillFrontmatter {
  const match = content.match(FRONTMATTER_REGEX);
  if (!match) {
    return { body: content.trim() };
  }

  const rawFrontmatter = match[1] ?? "";
  const body = content.slice(match[0].length).trim();

  let name: string | undefined;
  let description: string | undefined;
  let triggers: string[] | undefined;

  const nameMatch = rawFrontmatter.match(
    /^name:\s*['"]?([a-z0-9_-]+)['"]?\s*$/im,
  );
  if (nameMatch) {
    name = nameMatch[1]?.trim().toLowerCase();
  }

  const descBlockMatch = rawFrontmatter.match(
    /^description:\s*\|\s*\r?\n((?:[ \t].*(?:\r?\n|$))+)/m,
  );
  if (descBlockMatch) {
    description = descBlockMatch[1]!.replace(/^[ \t]+/gm, "").trim();
  } else {
    const descQuotedMatch = rawFrontmatter.match(
      /^description:\s*['"](.+?)['"]\s*$/m,
    );
    if (descQuotedMatch) {
      description = descQuotedMatch[1]!.trim();
    } else {
      const descInlineMatch = rawFrontmatter.match(/^description:\s*(.+)\s*$/m);
      if (descInlineMatch) {
        description = descInlineMatch[1]!.trim();
      }
    }
  }

  const triggersListMatch = rawFrontmatter.match(
    /^triggers:\s*\r?\n((?:[ \t]*-[ \t]+.*(?:\r?\n|$))+)/m,
  );
  if (triggersListMatch) {
    triggers = triggersListMatch[1]!
      .split(/\r?\n/)
      .map((line) =>
        line.replace(/^[ \t]*-[ \t]+['"]?|['"]?[ \t]*$/g, "").trim(),
      )
      .filter((t) => t.length > 0);
  } else {
    const inlineTriggersMatch = rawFrontmatter.match(/^triggers:\s*\[(.*?)\]/m);
    if (inlineTriggersMatch) {
      triggers = inlineTriggersMatch[1]!
        .split(",")
        .map((t) => t.replace(/['"]/g, "").trim())
        .filter((t) => t.length > 0);
    } else {
      const inlineCsvMatch = rawFrontmatter.match(/^triggers:\s*(.+)$/m);
      if (inlineCsvMatch) {
        triggers = inlineCsvMatch[1]!
          .split(",")
          .map((t) => t.replace(/['"]/g, "").trim())
          .filter((t) => t.length > 0);
      }
    }
  }

  return { name, description, triggers, body };
}

export function formatSkillMarkdown(params: CustomSkillInput): string {
  const trimmedName = params.name.trim().toLowerCase();
  const description = params.description?.trim();
  const rawContent = (params.content ?? "").trim();

  const parsed = parseSkillFrontmatter(rawContent);
  const body = parsed.body || rawContent;

  const triggers =
    params.triggers && params.triggers.length > 0
      ? params.triggers.map((t) => t.trim()).filter(Boolean)
      : [`/${trimmedName}`];

  const frontmatterLines = ["---", `name: ${trimmedName}`];

  if (description) {
    if (description.includes("\n")) {
      frontmatterLines.push("description: |");
      for (const line of description.split("\n")) {
        frontmatterLines.push(`  ${line}`);
      }
    } else {
      frontmatterLines.push(`description: ${description}`);
    }
  }

  if (triggers.length > 0) {
    frontmatterLines.push("triggers:");
    for (const trigger of triggers) {
      frontmatterLines.push(`  - ${trigger}`);
    }
  }

  frontmatterLines.push("---");
  frontmatterLines.push("");
  frontmatterLines.push(body);

  return frontmatterLines.join("\n");
}

export function loadCustomSkills(): SkillInfo[] {
  if (typeof window === "undefined" || !window.localStorage) {
    return [];
  }

  try {
    const raw = window.localStorage.getItem(CUSTOM_SKILLS_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];

    return parsed
      .filter(
        (item): item is CustomSkillInput =>
          typeof item === "object" &&
          item !== null &&
          typeof (item as CustomSkillInput).name === "string",
      )
      .map((item) => {
        const parsedFm = parseSkillFrontmatter(item.content || "");
        const name = item.name.trim().toLowerCase();
        const description =
          item.description?.trim() || parsedFm.description || null;
        const triggers = item.triggers?.length
          ? item.triggers
          : parsedFm.triggers?.length
            ? parsedFm.triggers
            : [`/${name}`];
        return {
          name,
          type: "knowledge" as const,
          source: "custom",
          origin: "user" as const,
          description,
          triggers,
          content: item.content || formatSkillMarkdown({ ...item, name }),
        };
      });
  } catch {
    return [];
  }
}

export function saveCustomSkill(input: CustomSkillInput): SkillInfo {
  const name = input.name.trim().toLowerCase();
  if (!name) {
    throw new Error("Skill name is required");
  }
  if (!/^[a-z0-9_-]+$/.test(name)) {
    throw new Error(
      "Skill name can only contain lowercase alphanumeric characters, hyphens, and underscores",
    );
  }

  const formattedContent = formatSkillMarkdown(input);
  const parsedFm = parseSkillFrontmatter(formattedContent);
  const triggers = input.triggers?.length
    ? input.triggers
    : parsedFm.triggers?.length
      ? parsedFm.triggers
      : [`/${name}`];
  const description = input.description?.trim() || parsedFm.description || null;

  const currentSkills = loadCustomSkills();
  const existingIndex = currentSkills.findIndex((s) => s.name === name);

  const updatedCustomSkill: SkillInfo = {
    name,
    type: "knowledge",
    source: "custom",
    origin: "user",
    description,
    triggers,
    content: formattedContent,
  };

  const nextSkills = [...currentSkills];
  if (existingIndex >= 0) {
    nextSkills[existingIndex] = updatedCustomSkill;
  } else {
    nextSkills.unshift(updatedCustomSkill);
  }

  if (typeof window !== "undefined" && window.localStorage) {
    window.localStorage.setItem(
      CUSTOM_SKILLS_STORAGE_KEY,
      JSON.stringify(nextSkills),
    );
    window.dispatchEvent(
      new CustomEvent(CUSTOM_SKILLS_UPDATED_EVENT, {
        detail: { skill: updatedCustomSkill },
      }),
    );
  }

  return updatedCustomSkill;
}

export function deleteCustomSkill(name: string): boolean {
  const trimmedName = name.trim().toLowerCase();
  const currentSkills = loadCustomSkills();
  const filtered = currentSkills.filter((s) => s.name !== trimmedName);
  if (filtered.length === currentSkills.length) return false;

  if (typeof window !== "undefined" && window.localStorage) {
    window.localStorage.setItem(
      CUSTOM_SKILLS_STORAGE_KEY,
      JSON.stringify(filtered),
    );
    window.dispatchEvent(
      new CustomEvent(CUSTOM_SKILLS_UPDATED_EVENT, {
        detail: { name: trimmedName },
      }),
    );
  }

  return true;
}

export function getCustomSkill(name: string): SkillInfo | null {
  const trimmedName = name.trim().toLowerCase();
  const skills = loadCustomSkills();
  return skills.find((s) => s.name === trimmedName) ?? null;
}
