import React, { useMemo } from "react";
import Markdown, { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";
import { code } from "./code";
import { ul, ol } from "./list";
import { paragraph } from "./paragraph";
import { anchor } from "./anchor";
import { h1, h2, h3, h4, h5, h6 } from "./headings";
import { table, th, td } from "./table";

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

// The default component set is fully static, so hoist it to module scope so it
// keeps a stable reference across renders. react-markdown diffing keys off the
// `components` identity; a fresh object each render forces every subtree to
// re-render and re-tokenize, which during streaming means the whole accumulated
// string is re-parsed on every token.
const DEFAULT_COMPONENTS: Partial<Components> = {
  code,
  ul,
  ol,
  table,
  th,
  td,
};

const STANDARD_COMPONENTS: Partial<Components> = {
  a: anchor,
  p: paragraph,
};

const HEADING_COMPONENTS: Partial<Components> = {
  h1,
  h2,
  h3,
  h4,
  h5,
  h6,
};

// remark plugins are static too; a stable array avoids re-processing config.
const REMARK_PLUGINS = [remarkGfm, remarkBreaks];

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
export const MarkdownRenderer = React.memo(
  ({
    children,
    content,
    components: customComponents,
    includeStandard = false,
    includeHeadings = false,
  }: MarkdownRendererProps) => {
    // Build the components object with defaults and optional additions. The
    // result is memoized on the flags/customComponents so the identity stays
    // stable unless something that actually changes the mapping changes.
    const components: Components = useMemo(() => {
      if (!includeStandard && !includeHeadings && !customComponents) {
        return DEFAULT_COMPONENTS;
      }
      return {
        ...DEFAULT_COMPONENTS,
        ...(includeStandard && STANDARD_COMPONENTS),
        ...(includeHeadings && HEADING_COMPONENTS),
        ...customComponents, // Custom components override defaults
      };
    }, [includeStandard, includeHeadings, customComponents]);

    const markdownContent = content ?? children ?? "";

    return (
      <div data-testid="markdown-renderer">
        <Markdown components={components} remarkPlugins={REMARK_PLUGINS}>
          {markdownContent}
        </Markdown>
      </div>
    );
  },
);

MarkdownRenderer.displayName = "MarkdownRenderer";
