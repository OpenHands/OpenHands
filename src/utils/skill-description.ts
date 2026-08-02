/** Strip common inline Markdown syntax from skill descriptions. */
export function stripMarkdown(text: string): string {
  return text
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\*{3}(.+?)\*{3}/g, "$1")
    .replace(/\*{2}(.+?)\*{2}/g, "$1")
    .replace(/\*(.+?)\*/g, "$1")
    .replace(/_{3}(.+?)_{3}/g, "$1")
    .replace(/_{2}(.+?)_{2}/g, "$1")
    .replace(/_(.+?)_/g, "$1")
    .replace(/`(.+?)`/g, "$1")
    .replace(/~~(.+?)~~/g, "$1");
}

/**
 * Extract a short plain-text description from skill content. AgentSkills
 * frontmatter is preferred; legacy content falls back to its first meaningful
 * body line.
 */
export function getSkillDescription(content: string): string | null {
  let body = content;
  const frontmatterMatch = content.match(/^---\s*\n([\s\S]*?)\n---/);

  if (frontmatterMatch) {
    const descriptionMatch = frontmatterMatch[1].match(
      /^description:\s*(.+)$/m,
    );
    if (descriptionMatch) {
      let description = descriptionMatch[1].trim();
      if (
        (description.startsWith('"') && description.endsWith('"')) ||
        (description.startsWith("'") && description.endsWith("'"))
      ) {
        description = description.slice(1, -1);
      }
      return stripMarkdown(description);
    }
    body = content.slice(frontmatterMatch[0].length);
  }

  const meaningfulLine = body
    .split("\n")
    .map((line) => line.trim())
    .find((line) => line.length > 0 && !line.startsWith("#") && line !== "---");

  if (!meaningfulLine) return null;

  const plainText = stripMarkdown(meaningfulLine);
  return plainText.match(/^[^.!?\n]*[.!?]/)?.[0] ?? plainText;
}
