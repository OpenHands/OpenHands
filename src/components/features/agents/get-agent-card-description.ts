import type { AgentInfo } from "#/types/settings";

const FRONTMATTER_PATTERN = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/;

function stripFrontmatter(content: string): string {
  const match = content.match(FRONTMATTER_PATTERN);
  if (!match) return content.trim();
  return content.slice(match[0].length).trim();
}

function extractBodyPreview(body: string): string {
  const withoutTitle = body.replace(/^#+\s+[^\n]+\n+/, "").trim();
  const paragraph =
    withoutTitle
      .split(/\n\s*\n/)
      .map((part) => part.replace(/\s+/g, " ").trim())
      .find((part) => part.length > 0) ??
    withoutTitle.replace(/\s+/g, " ").trim();

  return paragraph;
}

/**
 * Subtitle text for agent cards: prefer the API `description` (parsed from the
 * agent's frontmatter), then fall back to the first paragraph of the
 * `system_prompt` body.
 */
export function getAgentCardDescription(agent: AgentInfo): string {
  const description = agent.description?.trim();
  if (description) return description;

  const prompt = agent.system_prompt?.trim();
  if (!prompt) return "";

  return extractBodyPreview(stripFrontmatter(prompt));
}
