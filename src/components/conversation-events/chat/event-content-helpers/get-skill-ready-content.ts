import { TextContent } from "#/types/agent-server/core/base/common";

/**
 * Represents a single activated skill with its name and associated content.
 */
export interface SkillReadyItem {
  name: string;
  content: string;
}

/**
 * Extracts all text content from an array of TextContent items.
 */
function extractAllText(extendedContent: TextContent[]): string {
  return extendedContent
    .filter((c) => c.type === "text")
    .map((c) => c.text)
    .join("");
}

/**
 * Extracts all <EXTRA_INFO> blocks from the given text.
 * Returns an array of content strings (without the wrapper tags).
 */
const extractExtraInfoBlocks = (text: string): string[] => {
  const blockRegex = /<EXTRA_INFO>([\s\S]*?)<\/EXTRA_INFO>/gi;
  return Array.from(text.matchAll(blockRegex), (match) =>
    match[1].trim(),
  ).filter(Boolean);
};

/**
 * Formats a single skill with its corresponding content block.
 */
const formatSkillWithContent = (
  skill: string,
  contentBlock: string | undefined,
): string => {
  let formatted = `\n\n- **${skill}**`;

  if (contentBlock) {
    formatted += `\n\n${contentBlock}`;
  }

  return formatted;
};

/**
 * Formats skills paired with their corresponding extended content blocks.
 */
const formatSkillKnowledge = (
  activatedSkills: string[],
  extraInfoBlocks: string[],
): string => {
  let content = `\n\n**Triggered Skill Knowledge:**`;

  activatedSkills.forEach((skill, index) => {
    const contentBlock = extraInfoBlocks[index];
    content += formatSkillWithContent(skill, contentBlock);
  });

  return content;
};

/**
 * Formats extended content blocks when no skills are present.
 */
const formatExtendedContentOnly = (extraInfoBlocks: string[]): string => {
  let content = `\n\n**Extended Content:**`;

  extraInfoBlocks.forEach((block) => {
    content += `\n\n${block}`;
  });

  return content;
};

/**
 * Extracts EXTRA_INFO blocks from extended content.
 */
const getExtraInfoBlocks = (extendedContent?: TextContent[]): string[] => {
  if (!extendedContent) return [];
  return extractExtraInfoBlocks(extractAllText(extendedContent));
};

/**
 * Formats activated skills and extended content into markdown for display.
 * Each skill is paired with its corresponding <EXTRA_INFO> block by index.
 */
export const getSkillReadyContent = (
  activatedSkills: string[],
  extendedContent: TextContent[],
): string => {
  const extraInfoBlocks = getExtraInfoBlocks(extendedContent);

  if (activatedSkills && activatedSkills.length > 0) {
    return formatSkillKnowledge(activatedSkills, extraInfoBlocks);
  }

  if (extraInfoBlocks.length > 0) {
    return formatExtendedContentOnly(extraInfoBlocks);
  }

  return "";
};

/**
 * Returns structured skill items with their names and associated content.
 * Each skill is paired with its corresponding <EXTRA_INFO> block by index.
 */
export const getSkillReadyItems = (
  activatedSkills: string[],
  extendedContent: TextContent[],
): SkillReadyItem[] => {
  const extraInfoBlocks = getExtraInfoBlocks(extendedContent);

  if (activatedSkills && activatedSkills.length > 0) {
    return activatedSkills.map((skill, index) => ({
      name: skill,
      content: extraInfoBlocks[index] ?? "",
    }));
  }

  return extraInfoBlocks.map((block, index) => ({
    name: `Extended Content ${index + 1}`,
    content: block,
  }));
};
