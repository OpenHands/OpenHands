import React from "react";
import { ExtraProps } from "react-markdown";
import { CopyableContentWrapper } from "#/components/shared/buttons/copyable-content-wrapper";
import { useColorTheme } from "#/hooks/use-color-theme";
import { getSyntaxHighlighterTheme } from "#/themes/syntax-highlighter-themes";
import { cn } from "#/utils/utils";
import { SyntaxHighlighter } from "./syntax-highlighter";
import { MermaidDiagram } from "./mermaid-diagram";

/** Fenced-block languages rendered as an inline diagram instead of code. */
export const MERMAID_FENCE_LANGUAGES = new Set(["mermaid", "mmd"]);

// See https://github.com/remarkjs/react-markdown?tab=readme-ov-file#use-custom-components-syntax-highlight

/**
 * Component to render code blocks in markdown.
 *
 * Named `Code` rather than lowercase like its sibling markdown components
 * because it calls a hook (`useColorTheme`), and `react-hooks/rules-of-hooks`
 * only treats capitalized functions as components. It is exported as `code`
 * so the `components` map in markdown-renderer reads like the others.
 */
function Code({
  children,
  className,
}: React.ClassAttributes<HTMLElement> &
  React.HTMLAttributes<HTMLElement> &
  ExtraProps) {
  const colorTheme = useColorTheme();
  const match = /language-(\w+)/.exec(className || ""); // get the language
  const codeString = String(children).replace(/\n$/, "");

  // Mermaid fenced blocks render as an inline diagram, not highlighted code.
  if (match && MERMAID_FENCE_LANGUAGES.has(match[1].toLowerCase())) {
    return <MermaidDiagram source={codeString} />;
  }

  if (!match) {
    const isMultiline = String(children).includes("\n");

    if (!isMultiline) {
      return (
        <code
          className={cn(
            className,
            "bg-surface-raised text-foreground border border-surface-raised rounded px-[0.4em] py-[0.2em]",
          )}
        >
          {children}
        </code>
      );
    }

    return (
      <CopyableContentWrapper text={codeString}>
        <pre className="bg-surface-raised text-foreground border border-surface-raised rounded p-[1em] overflow-auto">
          <code className={className}>{codeString}</code>
        </pre>
      </CopyableContentWrapper>
    );
  }

  return (
    <CopyableContentWrapper text={codeString}>
      <SyntaxHighlighter
        className="rounded-lg"
        style={getSyntaxHighlighterTheme(colorTheme)}
        language={match?.[1]}
        PreTag="div"
      >
        {codeString}
      </SyntaxHighlighter>
    </CopyableContentWrapper>
  );
}

export { Code as code };
