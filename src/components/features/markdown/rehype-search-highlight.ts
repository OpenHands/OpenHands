import type { Plugin } from "unified";
import type { ElementContent, Root } from "hast";
import { SKIP, visit } from "unist-util-visit";

/** Text inside these is never marked (same guard as fade-in words). */
const SKIP_TAGS = new Set(["pre", "code", "svg", "math", "script", "style"]);

interface SearchHighlightOptions {
  terms: string[];
}

/**
 * Wraps search-term occurrences in `mark` elements so a message jumped to
 * from content-search results shows why it matched. Substring on purpose:
 * the backend matches tokens by prefix, so "doo" marks "door" too. Terms
 * shorter than 2 characters would mark noise (every "a", "i") and are
 * dropped.
 */
export const rehypeSearchHighlight: Plugin<[SearchHighlightOptions], Root> = ({
  terms: inputTerms = [],
}) => {
  const terms = inputTerms.filter((term) => term.length >= 2);
  if (terms.length === 0) {
    return undefined;
  }
  // Longest-first so "garden door" wins over "door" at the same spot.
  const pattern = new RegExp(
    terms
      .map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
      .sort((a, b) => b.length - a.length)
      .join("|"),
    "gi",
  );

  return (tree) => {
    visit(tree, (node, index, parent) => {
      if (node.type !== "text" || index == null || parent == null) return;
      if (parent.type === "element" && SKIP_TAGS.has(parent.tagName)) {
        return SKIP;
      }
      const value = node.value;
      if (!pattern.test(value)) {
        pattern.lastIndex = 0;
        return;
      }
      pattern.lastIndex = 0;

      const parts: ElementContent[] = [];
      let cursor = 0;
      for (const match of value.matchAll(pattern)) {
        const at = match.index ?? 0;
        if (at > cursor) {
          parts.push({ type: "text", value: value.slice(cursor, at) });
        }
        parts.push({
          type: "element",
          tagName: "mark",
          properties: {},
          children: [{ type: "text", value: match[0] }],
        });
        cursor = at + match[0].length;
      }
      if (cursor < value.length) {
        parts.push({ type: "text", value: value.slice(cursor) });
      }
      (parent as { children: ElementContent[] }).children.splice(
        index,
        1,
        ...parts,
      );
      return [SKIP, index + parts.length];
    });
  };
};
