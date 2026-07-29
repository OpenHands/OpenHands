import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";

import { SyntaxHighlighter } from "#/components/features/markdown/syntax-highlighter";
import { getPrismLanguageForFile } from "#/utils/file-language";

interface HighlightedSourceViewProps {
  path: string;
  text: string;
  mimeType?: string;
}

/**
 * Renders the raw bytes of a workspace text file with Prism syntax
 * highlighting. Used both in:
 *   - Rich mode for actual source files (.ts, .py, .yaml, …) — there is
 *     no "rich" rendering of source code, so highlighted source IS the
 *     rich view.
 *   - Plain mode for source code AND for the source form of markdown /
 *     HTML files (so users can inspect the markup behind a rich preview).
 *
 * When we don't have a Prism grammar for the file we fall through to a
 * plain `<pre>` so the bytes still show. The wrapper styling matches the
 * right-pane background so the highlighted block reads as part of the
 * surrounding chrome instead of a floating card.
 */
export function HighlightedSourceView({
  path,
  text,
  mimeType,
}: HighlightedSourceViewProps) {
  const language = getPrismLanguageForFile(path, mimeType);

  if (!language) {
    return (
      <pre
        data-testid="file-content-viewer-plain"
        className="h-full w-full overflow-auto whitespace-pre-wrap break-words bg-[var(--oh-surface)] p-4 text-xs leading-5 text-white custom-scrollbar-always"
      >
        {text}
      </pre>
    );
  }

  return (
    <div
      data-testid="file-content-viewer-highlighted"
      data-language={language}
      className="h-full w-full overflow-auto bg-[var(--oh-surface)] custom-scrollbar-always"
    >
      <SyntaxHighlighter
        language={language}
        style={vscDarkPlus}
        showLineNumbers
        wrapLongLines={false}
        // Override the theme's hard-coded background so the highlighter
        // blends with the right-pane chrome instead of painting a slab
        // of a slightly-different dark color.
        customStyle={{
          margin: 0,
          padding: "1rem",
          background: "transparent",
          fontSize: "0.75rem",
          lineHeight: "1.25rem",
          minHeight: "100%",
        }}
        codeTagProps={{
          style: { background: "transparent", fontFamily: "inherit" },
        }}
        lineNumberStyle={{
          color: "var(--oh-border)",
          minWidth: "2.5em",
          paddingRight: "1em",
          userSelect: "none",
        }}
      >
        {text}
      </SyntaxHighlighter>
    </div>
  );
}
