import type { Plugin } from "unified";
import type { Root } from "mdast";
import { visit } from "unist-util-visit";

export const ALERT_TYPES = [
  "note",
  "tip",
  "important",
  "warning",
  "caution",
] as const;
export type AlertType = (typeof ALERT_TYPES)[number];

// GitHub's alert marker has the form `[!TYPE]` on the first line of a
// blockquote, optionally followed by a newline before the body. We match
// case-insensitively because GitHub does, and we swallow the trailing
// `\r?\n` so the body content can start on the next visual line without
// inheriting a leading break.
const ALERT_REGEX = /^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\][ \t]*\r?\n?/i;

/**
 * Remark plugin that recognises GitHub-style alert blockquotes:
 *
 *     > [!WARNING]
 *     > Body content
 *
 * When the first text node of a blockquote matches `[!TYPE]`, the marker
 * is stripped and `markdown-alert markdown-alert-<type>` classes are
 * attached to the blockquote via `hProperties`. A downstream renderer
 * (see `blockquote.tsx`) inspects those classes to render the styled
 * alert layout. Blockquotes that don't match the marker are left alone.
 *
 * The plugin tolerates running before OR after `remark-breaks`:
 *   - Before: the first text node is `"[!WARNING]\nBody"`, the regex
 *     consumes the marker plus its newline, leaving `"Body"`.
 *   - After:  the first text node is just `"[!WARNING]"`, followed by a
 *     `break` node. We drop the now-empty text node and the leading
 *     break so the body paragraph reads naturally.
 */
function toClassList(value: unknown): string[] {
  if (Array.isArray(value)) return value as string[];
  if (typeof value === "string") return value.split(/\s+/).filter(Boolean);
  return [];
}

export const remarkGithubAlerts: Plugin<[], Root> = () => (tree) => {
  // Remark plugins mutate the AST in place by design — that's the visitor
  // pattern `unist-util-visit` is built around. Aliasing each node to a
  // local variable keeps the mutations out of `no-param-reassign`'s view
  // without changing semantics.
  visit(tree, "blockquote", (node) => {
    const block = node;
    const firstChild = block.children[0];
    if (firstChild?.type !== "paragraph") return;
    const paragraph = firstChild;
    const firstText = paragraph.children[0];
    if (firstText?.type !== "text") return;

    const match = firstText.value.match(ALERT_REGEX);
    if (!match) return;

    const type = match[1].toLowerCase() as AlertType;
    const text = firstText;
    text.value = text.value.replace(ALERT_REGEX, "");

    if (text.value === "") {
      paragraph.children.shift();
      if (paragraph.children[0]?.type === "break") {
        paragraph.children.shift();
      }
      if (paragraph.children.length === 0) {
        block.children.shift();
      }
    }

    block.data = block.data ?? {};
    const existingHProps = (block.data.hProperties ?? {}) as Record<
      string,
      unknown
    >;
    const baseClasses = toClassList(existingHProps.className);
    block.data.hProperties = {
      ...existingHProps,
      className: [...baseClasses, "markdown-alert", `markdown-alert-${type}`],
    };
  });
};
