import Markdown, { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";
import { code } from "./code";
import { ul, ol } from "./list";
import { paragraph } from "./paragraph";
import { anchor } from "./anchor";
import { h1, h2, h3, h4, h5, h6 } from "./headings";

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
}: MarkdownRendererProps) {
  // Build the components object with defaults and optional additions
  const components: Components = {
    code,
    ul,
    ol,
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

  return (
    <div data-testid="markdown-renderer">
      <Markdown
        components={components}
        remarkPlugins={[remarkGfm, remarkBreaks]}
      >
        {markdownContent}
      </Markdown>
    </div>
  );
}
