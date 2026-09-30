/** Strip common inline Markdown syntax so descriptions render as plain text. */
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

/** Extract a short plain-text description from skill Markdown. */
export function getSkillDescription(content: string): string | null {
  let body = content;
  const frontmatterMatch = content.match(/^---\s*\n([\s\S]*?)\n---/);
  if (frontmatterMatch) {
    const descMatch = frontmatterMatch[1].match(/^description:\s*(.+)$/m);
    if (descMatch) {
      let desc = descMatch[1].trim();
      if (
        (desc.startsWith('"') && desc.endsWith('"')) ||
        (desc.startsWith("'") && desc.endsWith("'"))
      ) {
        desc = desc.slice(1, -1);
      }
      return stripMarkdown(desc);
    }
    body = content.slice(frontmatterMatch[0].length);
  }

  const meaningful = body
    .split("\n")
    .map((line) => line.trim())
    .find((line) => line.length > 0 && !line.startsWith("#") && line !== "---");
  if (!meaningful) return null;

  const stripped = stripMarkdown(meaningful);
  const sentence = stripped.match(/^[^.!?\n]*[.!?]/);
  return sentence?.[0] || stripped;
}
