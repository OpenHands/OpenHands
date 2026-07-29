import {
  TextContent,
  ImageContent,
} from "#/types/agent-server/core/base/common";

/**
 * Joins the text parts of a tool observation's `content` array, dropping image
 * parts. Mirrors the extraction the markdown helpers do so migrated tools show
 * the same text the fallback path would have.
 */
export const textFromContent = (
  content: Array<TextContent | ImageContent>,
): string =>
  content
    .filter((c) => c.type === "text")
    .map((c) => c.text)
    .join("\n");
